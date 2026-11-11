"""Validate selected AWQ CUDA operators against independently unpacked weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Expected a path inside project: ' + value)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    model, report_path = project_path(args.model), project_path(args.report)
    if report_path.exists():
        raise FileExistsError('Preserving existing report: ' + str(report_path))
    if model == report_path or model in report_path.parents:
        raise ValueError('Keep validation reports outside the model directory')
    config = json.loads((model / 'config.json').read_text(encoding='utf-8'))
    audit = json.loads((model / 'conversion_audit.json').read_text(encoding='utf-8'))
    quant = config['quantization_config']
    if (quant['quant_method'], quant['version'], quant['bits'], quant['group_size'], quant['zero_point']) != (
            'awq', 'gemm', 4, 128, True):
        raise ValueError('Expected AWQ GEMM W4 group128 asymmetric')
    weights = model / 'model.safetensors'
    if weights.stat().st_size != audit['bytes'] or audit['quantized_layer_count'] != 252:
        raise ValueError('Model size/layer count differs from conversion audit')
    if args.check_only:
        print('AWQ_KERNEL_PREFLIGHT_OK: paths/configuration checked; no GPU computation or report writes')
        return

    import torch
    from safetensors import safe_open
    from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the deployment environment')
    print('Verifying converted model SHA256', flush=True)
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != audit['sha256']:
        raise ValueError('Model SHA256 differs from conversion audit')

    def unpack(packed):
        # Inverse of the AWQ nibble order [0,2,4,6,1,3,5,7].
        return torch.stack([(packed >> shift) & 15 for shift in [0, 16, 4, 20, 8, 24, 12, 28]], -1).reshape(packed.shape[0], -1)

    torch.manual_seed(7)
    torch.set_num_threads(4)
    checks = []
    with torch.inference_mode(), safe_open(str(weights), framework='pt', device='cpu') as reader:
        for layer in [0, 18, 35]:
            for projection in ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.down_proj']:
                prefix = 'model.language_model.layers.%d.%s' % (layer, projection)
                qw, qz, scales = [reader.get_tensor(prefix + '.' + suffix).cuda()
                                  for suffix in ['qweight', 'qzeros', 'scales']]
                integers = unpack(qw)
                zeros = unpack(qz).repeat_interleave(128, 0)
                dense = ((integers - zeros).half() * scales.repeat_interleave(128, 0)).contiguous()
                for batch in [1, 16]:
                    x = torch.randn(batch, dense.shape[0], dtype=torch.float16, device='cuda') * 0.1
                    actual = awq_linear(x, qw, scales, qz)
                    expected = x @ dense
                    torch.cuda.synchronize()
                    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                        raise ValueError('Non-finite CUDA result: ' + prefix)
                    relative_l2 = float((actual.float() - expected.float()).norm() /
                                        expected.float().norm().clamp_min(1e-8))
                    if relative_l2 >= 0.01:
                        raise ValueError('CUDA result mismatch: %s batch=%d relative_l2=%g' %
                                         (prefix, batch, relative_l2))
                    checks.append(dict(layer=prefix, batch=batch, relative_l2=relative_l2,
                                       max_abs=float((actual.float() - expected.float()).abs().max())))
                    print('KERNEL_OK', prefix, batch, relative_l2, flush=True)
                del qw, qz, scales, integers, zeros, dense, x, actual, expected
    report = dict(status='passed', model=model.relative_to(ROOT).as_posix(), model_sha256=digest.hexdigest(),
                  scope='9 sampled projections x 2 input row counts; compares AWQ kernel with unpacked quantized weights, not source-model quality',
                  tolerance_relative_l2=0.01, checks=checks,
                  max_relative_l2=max(check['relative_l2'] for check in checks))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print('AWQ_CUDA_KERNEL_VERIFIED', len(checks), 'max_relative_l2=', report['max_relative_l2'], flush=True)


if __name__ == '__main__':
    main()
