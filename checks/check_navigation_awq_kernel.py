"""Check real LMDeploy CUDA AWQ GEMM against independently unpacked weights."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_AWQ'


def unpack(packed):
    # GEMM storage places natural columns 0,2,4,6,1,3,5,7 in successive nibbles.
    return torch.stack([(packed >> shift) & 15 for shift in [0,16,4,20,8,24,12,28]], -1).reshape(packed.shape[0], -1)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(4)
    checks = []
    with safe_open(str(MODEL / 'model.safetensors'), framework='pt', device='cpu') as reader:
        for layer in [0, 18, 35]:
            for projection in ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.down_proj']:
                prefix = 'model.language_model.layers.{}.{}'.format(layer, projection)
                qw, qz, scales = [reader.get_tensor(prefix + '.' + suffix).cuda() for suffix in ['qweight', 'qzeros', 'scales']]
                integers = unpack(qw)
                zeros = unpack(qz).repeat_interleave(128, 0)
                dense = ((integers - zeros).half() * scales.repeat_interleave(128, 0)).contiguous()
                for batch in [1, 16]:
                    x = torch.randn(batch, dense.shape[0], dtype=torch.float16, device='cuda') * 0.1
                    actual = awq_linear(x, qw, scales, qz)
                    expected = x @ dense
                    torch.cuda.synchronize()
                    assert torch.isfinite(actual).all(), prefix
                    relative_l2 = float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8))
                    assert relative_l2 < 0.01, (prefix, batch, relative_l2)
                    checks.append(dict(layer=prefix, batch=batch, relative_l2=relative_l2,
                                       max_abs=float((actual.float() - expected.float()).abs().max())))
                    print('KERNEL_OK', prefix, batch, relative_l2, flush=True)
                del qw, qz, scales, integers, zeros, dense, x, actual, expected
    report = dict(status='passed', scope='9 actual quantized projections, decode/prefill; layout/kernel correctness, not quantization quality',
                  tolerance_relative_l2=0.01, checks=checks)
    (ROOT / 'reports/navigation_awq_kernel.json').write_text(json.dumps(report, indent=2))
    print('AWQ_CUDA_KERNEL_VERIFIED', len(checks), flush=True)


if __name__ == '__main__':
    main()
