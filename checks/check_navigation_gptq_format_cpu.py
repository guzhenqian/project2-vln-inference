"""Bounded synthetic CPU GPTQ/INT4 format check; never loads a model checkpoint."""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import hashlib
import importlib.metadata
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/navigation_gptq_format_cpu_20261109.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(not REPORT.exists(), 'Preserve existing report; do not overwrite evidence')
    import torch
    from loguru import logger
    from compressed_tensors.quantization import ActivationOrdering, QuantizationArgs
    from compressed_tensors.compressors.quantized_compressors.pack_quantized import (
        PackedQuantizationCompressor, unpack_from_int32,
    )
    from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
        accumulate_hessian, make_empty_hessian, quantize_weight,
    )

    torch.set_num_threads(1)
    torch.manual_seed(7)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    args = QuantizationArgs(num_bits=4, type='int', symmetric=False,
                            strategy='group', group_size=128, actorder='static')
    require(args.actorder == ActivationOrdering.WEIGHT, 'Static activation-order alias changed')

    def check_group_mapping(actorder, g_idx, width):
        require(actorder in ('static', 'weight'), 'Only static GPTQ is accepted')
        if g_idx is not None:
            require(g_idx.shape == (width,), 'Invalid g_idx shape')
            require(g_idx.dtype in (torch.int32, torch.int64), 'Invalid g_idx dtype')
            expected = torch.arange(width, dtype=g_idx.dtype) // 128
            require(torch.equal(g_idx.cpu(), expected), 'Non-contiguous group mapping')

    def pack_awq(value):
        require(value.dtype == torch.int32 and value.ndim == 2, 'Invalid INT4 input')
        require(value.shape[1] % 8 == 0, 'Output channels must align to 8')
        require(bool(((value >= 0) & (value <= 15)).all()), 'INT4 range violation')
        order = [0, 2, 4, 6, 1, 3, 5, 7]
        grouped = value.reshape(value.shape[0], -1, 8)[:, :, order]
        return (grouped << (torch.arange(8, dtype=torch.int32) * 4)).sum(-1, dtype=torch.int32)

    def unpack_awq(value):
        # Independent fixed shift schedule also used by the existing CUDA reference check.
        return torch.stack([(value >> s) & 15 for s in (0, 16, 4, 20, 8, 24, 12, 28)], -1).reshape(value.shape[0], -1)

    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level='WARNING')
    checks = []
    try:
        with torch.no_grad():
            for input_features, output_features in ((256, 32), (512, 16)):
                layer = torch.nn.Linear(input_features, output_features, bias=False, device='cpu')
                layer.weight.copy_(layer.weight.half().float())
                # Correlated, full-rank synthetic activations exercise Hessian updates.
                x = torch.randn(1, input_features * 2, input_features)
                x[:, :, 1:] += 0.25 * x[:, :, :-1].clone()
                x *= torch.linspace(0.5, 2.0, input_features)
                hessian, samples = accumulate_hessian(x, layer, make_empty_hessian(layer), 0)
                damped = hessian + torch.eye(input_features) * (0.01 * hessian.diag().mean())
                torch.linalg.cholesky(damped)  # Fail explicitly instead of allowing RTN fallback.
                loss, quantized, scale, zero, g_idx = quantize_weight(layer, args, {layer: hessian})
                require(not warnings, 'GPTQ emitted warning; result may contain RTN fallback')
                require(g_idx is None, 'Static GPTQ unexpectedly retained a group mapping')
                check_group_mapping(args.actorder, g_idx, input_features)
                compressor = PackedQuantizationCompressor()
                packed = compressor.compress_weight(quantized, scale, args, zero_point=zero, g_idx=g_idx)
                restored = compressor.decompress_weight({**packed, 'weight_scale': scale}, args)
                torch.testing.assert_close(restored, quantized, rtol=0, atol=1e-7)
                signed = unpack_from_int32(packed['weight_packed'], 4, quantized.shape).to(torch.int32)
                zeros = unpack_from_int32(packed['weight_zero_point'], 4, scale.shape, packed_dim=0).to(torch.int32)
                qweight = pack_awq((signed + 8).t().contiguous())
                qzeros = pack_awq((zeros + 8).t().contiguous())
                require(torch.equal(unpack_awq(qweight), (signed + 8).t()), 'Weight integers changed')
                require(torch.equal(unpack_awq(qzeros), (zeros + 8).t()), 'Zero points changed')
                delta = unpack_awq(qweight) - unpack_awq(qzeros).repeat_interleave(128, 0)
                reference = (delta.float() * scale.t().repeat_interleave(128, 0)).t()
                torch.testing.assert_close(reference, restored, rtol=0, atol=0)
                half_scale = scale.half()
                require(bool(torch.isfinite(half_scale).all() and (half_scale > 0).all()), 'Invalid FP16 scale')
                native_math = (delta.float() * half_scale.float().t().repeat_interleave(128, 0)).t()
                scale_error = (native_math - reference).abs().max().item()
                bound = 15 * (half_scale.float() - scale).abs().max().item() + 1e-7
                require(scale_error <= bound, 'Error exceeds FP16 scale rounding bound')
                checks.append(dict(shape=list(layer.weight.shape), synthetic_batches=samples,
                    g_idx_absent=True, gptq_loss=loss, integer_roundtrip_exact=True,
                    float32_dequantization_exact=True,
                    compressed_roundtrip_max_abs=(restored-quantized).abs().max().item(),
                    fp16_scale_max_abs=(half_scale.float()-scale).abs().max().item(),
                    fp16_scale_weight_max_abs=scale_error, fp16_scale_weight_error_bound=bound))
    finally:
        logger.remove(sink)

    identity = torch.arange(256, dtype=torch.int32) // 128
    check_group_mapping('static', identity, 256)
    rejected = []
    for label, order, mapping in (
        ('dynamic_actorder', 'dynamic', None),
        ('group_actorder', 'group', None),
        ('non_contiguous_g_idx', 'static', identity.flip(0)),
        ('wrong_g_idx_dtype', 'static', identity.float()),
        ('wrong_g_idx_shape', 'static', identity[:128]),
    ):
        try:
            check_group_mapping(order, mapping, 256)
        except ValueError:
            rejected.append(label)
        else:
            raise ValueError('Unsafe mapping was accepted: ' + label)

    source_hashes = {}
    for item in (quantize_weight, PackedQuantizationCompressor, QuantizationArgs):
        path = Path(inspect.getsourcefile(item))
        relative = str(path).split('/site-packages/', 1)[-1]
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes['checks/' + Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    report = dict(status='passed_cpu_format_only', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Synthetic CPU Linear GPTQ -> compressed-tensors -> AWQ GEMM integer layout',
        model_loaded=False, calibration_dataset_loaded=False, cuda_initialized=False,
        real_model_quantized=False, lmdeploy_kernel_executed=False, formal_experiment_started=False,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'llmcompressor', 'compressed-tensors', 'transformers')},
        quantization=args.model_dump(mode='json'), seed=7, warnings=warnings,
        checks=checks, unsafe_mappings_rejected=rejected, source_sha256=source_hashes,
        next_gate='Real checkpoint conversion, native CUDA operators and Qwen3-VL requests remain unvalidated')
    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], checks=len(checks),
        negative_checks=len(rejected), cuda_initialized=False, report=REPORT.relative_to(ROOT).as_posix())))


if __name__ == '__main__':
    main()
