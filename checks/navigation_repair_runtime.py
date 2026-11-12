"""New repair cohort runtime; frozen diagnostic/runtime files stay unchanged."""
import os
from types import MethodType

import torch

WB = int(os.environ.get('PROJECT2_REPAIR_WBITS', '16'))
AB = int(os.environ.get('PROJECT2_REPAIR_ABITS', '16'))
EXTENSION = (WB, AB) in ((4, 4), (4, 8), (8, 16))

if EXTENSION:
    import navigation_stage2_runtime as stage2
    from navigation_precision_guard import validate_config, inspect_awq_modules
    from lmdeploy.pytorch.backends.awq_modules import LinearW4A16Impl
    from lmdeploy.pytorch.backends.cuda.awq_modules import AwqLinearW4A16Builder

    class NavigationPrecisionImpl(LinearW4A16Impl):
        def __init__(self, in_features, out_features, w_bit, group_size):
            assert w_bit == WB and group_size == 128
            self.in_features, self.out_features = in_features, out_features
            self.w_bit, self.group_size, self.calls = w_bit, group_size, 0

        def forward(self, x, qweight, scales, qzeros, bias=None, all_reduce=False, group=None):
            from navigation_precision_repair_kernels import precision_linear
            assert not all_reduce
            shape = x.shape[:-1] + (self.out_features,)
            value = precision_linear(x.flatten(0, -2).contiguous(), qweight, scales, qzeros,
                                     WB, AB)
            if bias is not None:
                value = value + bias
            value = value.reshape(shape)
            self.calls += 1
            return value.unsqueeze(0) if value.ndim == 2 else value

    def build(in_features, out_features, w_bit, group_size, bias=False, dtype=None):
        return NavigationPrecisionImpl(in_features, out_features, w_bit, group_size)

    AwqLinearW4A16Builder.build = staticmethod(build)
    stage2.validate_config, stage2.inspect_awq_modules = validate_config, inspect_awq_modules

from navigation_w4a16_fix_runtime import W4A16FixModel


class NavigationRepairModel(W4A16FixModel):
    def _fix_before_step(self, kwargs):
        # Only fixed, explicit first two smoke requests collect GPU input SHA.
        # All subsequent variable navigation lengths retain lightweight timing.
        if kwargs.get('pixel_values') is not None and self._fix_index >= 1:
            self._fix_seen_inputs.add(kwargs['input_ids'].numel())
        return super()._fix_before_step(kwargs)

    def _precision_evidence(self):
        if not EXTENSION:
            return super()._precision_evidence()
        return dict(algorithm=self._method, weight_bits=WB, activation_bits=AB,
            activation_recipe='dynamic_token_group128_maxabs' if AB == 4 else
                              'dynamic_token_maxabs' if AB == 8 else None,
            integer_compute_bits=8 if AB < 16 else None, activation_storage_bits=AB,
            intermediate_dtype='float16', stock_backend=False,
            w4a4_native_int4_mma=False if AB == 4 else None)

    def forward(self, *args, **kwargs):
        first = not self._fix_execution_verified
        if first and self._is_quant and not EXTENSION:
            from navigation_w4a16_repair_kernels import install_for_model
            self._repair_kernel_table = install_for_model(self)
        before = {name: module.impl.calls for name, module in self._awq_objects.items()} if first and EXTENSION else {}
        value = super().forward(*args, **kwargs)
        if first:
            counts = {name: module.impl.calls-before[name] for name, module in self._awq_objects.items()} if EXTENSION else {}
            if EXTENSION:
                assert len(counts) == 144 and set(counts.values()) == {1}
            self._fix_write('repair_execution_audit.json', dict(weight_bits=WB, activation_bits=AB,
                extension=EXTENSION, actual_extension_calls=counts,
                frozen_launch_table=getattr(self, '_repair_kernel_table', None),
                input_sha_scope='Explicit first two fixed smoke requests only',
                native_forward=True, formal_online_autotune=False))
        return value
