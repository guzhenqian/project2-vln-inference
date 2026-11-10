"""Numerical/packing/PTX gates, including real AWQ/GPTQ projection tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from navigation_precision_kernels import precision_linear, quantize_activations, _linear

ROOT = Path(__file__).resolve().parents[1]


def unpack(q, bits):
    shifts = [0, 16, 4, 20, 8, 24, 12, 28] if bits == 4 else [0, 8, 16, 24]
    return torch.stack([(q >> i) & ((1 << bits) - 1) for i in shifts], -1).reshape(q.shape[0], -1)


def pack(q, bits):
    order = [0, 2, 4, 6, 1, 3, 5, 7] if bits == 4 else [0, 1, 2, 3]
    grouped = q.reshape(q.shape[0], -1, len(order))[:, :, order]
    shift = torch.arange(len(order), device=q.device, dtype=torch.int32) * bits
    return (grouped << shift).sum(-1, dtype=torch.int32)


def activation_reference(x, bits):
    if bits == 16:
        return x.float()
    limit = 2**(bits - 1) - 1
    scale = (x.float().abs().amax(-1) / limit).clamp_min(1.e-8)
    q = (x.float() / scale[:, None]).round().clamp(-limit, limit)
    actual, actual_scale = quantize_activations(x, bits)
    if bits == 4:
        actual = torch.stack([(actual >> i) & 15 for i in range(0, 32, 4)], -1).reshape(x.shape)
        actual = torch.where(actual >= 8, actual - 16, actual)
    assert torch.equal(actual.float(), q), 'Activation integer packing mismatch'
    assert torch.equal(actual_scale, scale), 'Activation scale mismatch'
    return q * scale[:, None]


def validate(x, qw, s, qz, wb, ab, label):
    # Independently decode the packed storage, rather than repeat Triton indexing.
    dense = ((unpack(qw, wb) - unpack(qz, wb).repeat_interleave(128, 0)).half()
             * s.repeat_interleave(128, 0)).contiguous()
    a = activation_reference(x, ab)
    if ab == 16:
        expected = a @ dense.float()
    else:
        # Integer MMA scales after each group; do not introduce FP16 weight-rounding here.
        integers = (unpack(qw, wb) - unpack(qz, wb).repeat_interleave(128, 0)).float()
        expected = a @ (integers * s.float().repeat_interleave(128, 0))
    actual = precision_linear(x, qw, s, qz, wb, ab)
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    error = float((actual.float() - expected).norm() / expected.norm().clamp_min(1.e-8))
    assert error < .005, (label, wb, ab, x.shape, error)
    result = dict(label=label, weight_bits=wb, activation_bits=ab, rows=len(x),
                  relative_l2=error, activation_packing_verified=ab != 16)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--w8-model')
    args = parser.parse_args()
    report = ROOT / args.report
    assert not report.exists()
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    checks = []
    with torch.inference_mode():
        for wb, ab in [(8, 16), (4, 8), (4, 4)]:
            for k in (128, 256, 1024):
                n = 192
                w = torch.randint(0, 2**wb, (k, n), dtype=torch.int32, device='cuda')
                z = torch.randint(0, 2**wb, (k//128, n), dtype=torch.int32, device='cuda')
                s = torch.rand((k//128, n), dtype=torch.float16, device='cuda') * .01 + .001
                qw, qz = pack(w, wb), pack(z, wb)
                assert torch.equal(unpack(qw, wb), w) and torch.equal(unpack(qz, wb), z)
                for rows in (1, 17, 65):
                    x = torch.randn((rows, k), dtype=torch.float16, device='cuda') * .1
                    x[0] = 0  # Zero scale/zero output must be handled safely.
                    checks.append(validate(x, qw, s, qz, wb, ab, 'synthetic'))
        models = [('awq', 'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104'),
                  ('gptq', 'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_stage2_20261109')]
        if args.w8_model:
            models = [('w8', args.w8_model)]
        for name, model in models:
            with safe_open(str(ROOT/model/'model.safetensors'), framework='pt', device='cpu') as reader:
                for layer in (0, 18, 35):
                    for projection in ('self_attn.q_proj', 'mlp.down_proj'):
                        prefix = f'model.language_model.layers.{layer}.{projection}'
                        qw, s, qz = [reader.get_tensor(prefix+'.'+suffix).cuda() for suffix in ('qweight','scales','qzeros')]
                        for rows in (1, 701, 2815):
                            x = torch.randn((rows, qw.shape[0]), device='cuda', dtype=torch.float16)*.1
                            for wb, ab in ([(8, 16)] if args.w8_model else [(4, 8), (4, 4)]):
                                checks.append(validate(x, qw, s, qz, wb, ab, name+':'+prefix))
        # Read emitted PTX from Triton's actual launch cache; require integer MMA for A4/A8.
        ptx = []
        for cache in _linear.cache.values():
            for compiled in cache.values():
                if hasattr(compiled, 'asm'):
                    ptx.append(compiled.asm.get('ptx', ''))
        integer = [p for p in ptx if '.s8.s8.' in p]
        floating = [p for p in ptx if '.f16.f16.' in p]
        assert integer and floating, 'Compiled integer and FP16 MMA evidence missing'
        data = dict(status='passed', gpu=torch.cuda.get_device_name(), checks=checks,
                    kernel_source_sha256=hashlib.sha256((ROOT/'checks/navigation_precision_kernels.py').read_bytes()).hexdigest(),
                    ptx_integer_variants=len(integer), ptx_float_variants=len(floating),
                    ptx_sha256=[hashlib.sha256(p.encode()).hexdigest() for p in ptx],
                    w4a4_implementation='packed INT4 activation storage, widening to INT8 integer MMA; not native INT4 MMA',
                    activation_recipe='dynamic per-token symmetric maxabs; signed narrow range [-7,7]/[-127,127]',
                    tolerance_relative_l2=.005, max_relative_l2=max(c['relative_l2'] for c in checks))
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
        print('PRECISION_KERNEL_GATES_PASSED', len(checks), flush=True)


if __name__ == '__main__':
    main()
