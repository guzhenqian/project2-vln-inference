"""Isolated wider launch search around the unchanged LMDeploy W4A16 JIT.

The original packed asymmetric group128 weights and arithmetic are retained.
No package/old module is modified. A saved table supports exact measured M
only; no unmeasured dynamic-length policy is implied by this experiment.
"""
import json
from pathlib import Path

import torch
import triton

from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear_kernel
from navigation_w4a16_candidate_kernels import _validate


_jit = triton.jit(awq_linear_kernel.fn.fn, do_not_specialize=['M'])
_table = {}
_identity = None
_frozen = False


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def candidates(m, k, *, limit=32):
    """Deterministic coverage-first sequence, including the native launch family."""
    native_split = max(1, k // 4096)
    native_m = max(16, min(128, triton.next_power_of_2(m)))
    configs = []

    def add(bm, bn, split, warps=4, pipeline=2, compiler_stages=3, group_m=8):
        c = dict(block_m=bm, block_n=bn, split_k=split, num_warps=warps,
                 pipeline_stages=pipeline, compiler_stages=compiler_stages,
                 group_m=group_m)
        if c not in configs:
            configs.append(c)

    # First reproduce both native candidates; then vary one important axis.
    for bn in (64, 128):
        add(native_m, bn, native_split)
    for split in (1, 2, 4):
        for bn in (64, 128, 256):
            add(native_m, bn, split)
    for bm in ((16, 32) if m == 1 else (64, 32, 128)):
        for bn in (64, 128, 256):
            add(bm, bn, native_split)
    for bn in (64, 128, 256):
        for pipeline in (2, 3):
            add(native_m, bn, native_split, 8, pipeline)
            add(native_m, bn, native_split, 4, pipeline, 2)
    for bm in ((16,) if m == 1 else (32, 64, 128)):
        for bn in (64, 128, 256):
            for split in (1, 2, 4):
                for group_m in (4, 8):
                    add(bm, bn, split, 4, 2, 3, group_m)
    return configs[:limit] if limit else configs


def launch(x, qweight, scales, qzeros, config):
    m, k, n = _validate(x, qweight, scales, qzeros)
    if not x.is_contiguous():
        raise ValueError('Contiguous activations required')
    split = int(config['split_k'])
    if split not in (1, 2, 4) or split > k // 128:
        raise ValueError('Unsupported split count')
    out = scales.new_zeros((m, n)) if split > 1 else scales.new_empty((m, n))
    bm, bn = config['block_m'], config['block_n']
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), split)
    _jit[grid](x, qweight, scales, qzeros, out, m, n, k,
        stride_am=x.stride(0), stride_ak=x.stride(1),
        stride_wk=qweight.stride(0), stride_wn=qweight.stride(1),
        stride_sk=scales.stride(0), stride_sn=scales.stride(1),
        stride_zk=qzeros.stride(0), stride_zn=qzeros.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=128,
        SPLIT_K=split, NUM_STAGES=config['pipeline_stages'],
        GROUP_SIZE_M=config['group_m'], num_warps=config['num_warps'],
        num_stages=config['compiler_stages'])
    return out


def load_table(path, algorithm, model_weights_sha256):
    global _table, _identity
    if _table or _frozen:
        raise RuntimeError('Load a table only once in a fresh worker')
    receipt = json.loads(Path(path).read_text(encoding='utf-8'))
    if receipt.get('algorithm') != algorithm or receipt.get('model_weights_sha256') != model_weights_sha256:
        raise ValueError('Candidate table checkpoint identity mismatch')
    if receipt.get('scope') != 'exact_m_only' or not receipt.get('numerical_gate_passed'):
        raise ValueError('Only a numerically accepted exact-M table may be loaded')
    for row in receipt['shapes']:
        key = (int(row['m']), int(row['k']), int(row['n']))
        if key in _table:
            raise ValueError('Duplicate candidate table shape')
        _table[key] = dict(row['config'])
    if not _table:
        raise ValueError('Empty candidate table')
    _identity = dict(algorithm=algorithm, model_weights_sha256=model_weights_sha256)
    return receipt


def prepare_projection(qweight, scales, qzeros):
    if _frozen or not _table:
        raise RuntimeError('Load table and prepare before freezing')
    k, n = qweight.shape[0], scales.shape[1]
    rows = sorted(m for m, kk, nn in _table if kk == k and nn == n)
    if not rows:
        raise ValueError('Unknown projection shape')
    with torch.inference_mode():
        for m in rows:
            x = torch.zeros((m, k), device=qweight.device, dtype=torch.float16)
            launch(x, qweight, scales, qzeros, _table[(m, k, n)])
        torch.cuda.synchronize()
    return rows


def freeze():
    global _frozen
    if not _table:
        raise RuntimeError('No table loaded')
    def reject(*args, **kwargs):
        raise RuntimeError('Cold candidate kernel compilation after freeze')
    _jit.compile = reject
    _frozen = True
    return dict(**_identity, scope='exact_m_only', frozen=True,
                shapes=[dict(m=m, k=k, n=n, config=c) for (m, k, n), c in sorted(_table.items())])


def precision_linear(x, qweight, scales, qzeros):
    if not _frozen:
        raise RuntimeError('Prepare and freeze before measurement')
    m, k, n = _validate(x, qweight, scales, qzeros)
    key = (m, k, n)
    if key not in _table:
        raise ValueError('Exact M was not benchmarked: ' + repr(key))
    return launch(x, qweight, scales, qzeros, _table[key])
