"""Isolated LMDeploy extension with executed packed-precision receipts."""
import os
import torch
import navigation_stage2_runtime as base
from navigation_precision_guard import validate_config, inspect_awq_modules
from navigation_precision_kernels import precision_linear
from lmdeploy.pytorch.backends.awq_modules import LinearW4A16Impl

WB = int(os.environ['PROJECT2_PRECISION_WBITS'])
AB = int(os.environ['PROJECT2_PRECISION_ABITS'])
assert (WB, AB) in ((4,4), (4,8), (8,16))


class NavigationPrecisionImpl(LinearW4A16Impl):
    def __init__(self, in_features, out_features, w_bit, group_size):
        assert w_bit == WB and group_size == 128
        self.in_features, self.out_features = in_features, out_features
        self.w_bit, self.group_size = w_bit, group_size
        self.calls = 0

    def forward(self, x, qweight, scales, qzeros, bias=None, all_reduce=False, group=None):
        if all_reduce:
            raise ValueError('Precision extension only validated for TP1')
        shape = x.shape[:-1] + (self.out_features,)
        out = precision_linear(x.flatten(0, -2), qweight, scales, qzeros, WB, AB)
        if bias is not None:
            out = out + bias
        out = out.reshape(shape)
        self.calls += 1
        return out.unsqueeze(0) if out.ndim == 2 else out


def build(in_features, out_features, w_bit, group_size, bias=False, dtype=None):
    return NavigationPrecisionImpl(in_features, out_features, w_bit, group_size)


# Process-local extension points only: installed package and frozen files stay unchanged.
from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Builder
AwqLinearW4A16Builder.build = staticmethod(build)
base.validate_config = validate_config
base.inspect_awq_modules = inspect_awq_modules


class PrecisionQwen3VLForConditionalGeneration(base.Stage2Qwen3VLForConditionalGeneration):
    def _precision_evidence(self):
        return dict(algorithm=self._method, weight_quantization_algorithm=self._method,
                    weight_dtype=f'int{WB}_packed+float16_scales', activation_dtype=f'int{AB}' if AB<16 else 'float16',
                    weight_bits=WB, activation_bits=AB,
                    activation_recipe='dynamic_per_token_symmetric_maxabs_narrow_range' if AB<16 else None,
                    integer_compute_bits=8 if AB<16 else None,
                    activation_storage_bits=AB, intermediate_dtype='float16',
                    deployment_format='project2_precision_extension', stock_backend=False,
                    w4a4_native_int4_mma=False if AB==4 else None,
                    quantization_config=dict(quant_method='awq', version='gemm', bits=WB, group_size=128, zero_point=True))

    def forward(self, *args, **kwargs):
        self._precision_before = {name: module.impl.calls for name, module in self._awq_objects.items()}
        return super().forward(*args, **kwargs)

    def _awq_execution(self, lower_rows, upper_rows):
        receipt = super()._awq_execution(lower_rows, upper_rows)
        counts = {name: module.impl.calls-self._precision_before[name] for name, module in self._awq_objects.items()}
        assert len(counts) == 144 and set(counts.values()) == {1}, 'Actual custom kernel execution missing'
        receipt.update(precision_extension_executed=True, weight_bits=WB, activation_bits=AB,
                       compute='int8_mma' if AB<16 else 'fp16_mma',
                       quantized_projection_calls=sum(counts.values()), activation_packed=AB==4,
                       no_fp16_activation_fallback=AB<16)
        return receipt
