"""Independent M=1 asymmetric group128 W4A16 GEMV experiment.

No service integration or modification of the existing native/wide tables.
Weights are rounded to FP16 after dequantization. Products and reductions use
FP32; their order differs from Tensor Core GEMM and needs numerical validation.
"""
import torch
import triton
from triton import language as tl

from lmdeploy.pytorch.kernels.cuda.awq_kernels import _unpack_weight
from navigation_w4a16_candidate_kernels import _validate


def candidates(limit=48):
    """Coverage-first order; full Cartesian family has 72 configurations."""
    result = []
    # Prefer moderate register tiles and enough CTAs before large BK/BN.
    for warps in (4, 8, 2):
        for bk in (128, 256):
            for split in (4, 2, 8, 1):
                for bn in (32, 64, 128):
                    result.append(dict(block_n=bn, block_k=bk, split_k=split,
                                       num_warps=warps, compiler_stages=1))
    return result[:limit] if limit else result


def schedule_description(k, n, config):
    """CPU-only upper-bound sizing; not measured register count/occupancy."""
    bn, bk, split = (config[name] for name in ('block_n', 'block_k', 'split_k'))
    tiles = (k + bk - 1) // bk
    return dict(ctas=((n + bn - 1) // bn) * split,
                max_k_iterations=(tiles + split - 1) // split,
                tile_elements=bk * bn,
                tile_elements_per_thread=bk * bn / (32 * config['num_warps']),
                packed_weight_bytes=k * n // 2,
                scale_and_zero_bytes=(k // 128) * (2 * n + n // 2),
                fp32_partial_bytes=4 * split * n if split > 1 else 0,
                launches=2 if split > 1 else 1,
                limitation='Elements/thread is not registers/thread; compiler metadata is measured separately.')


@triton.jit
def _gemv(x_ptr, qw_ptr, s_ptr, qz_ptr, out_ptr, partial_ptr,
          K: tl.constexpr, N: tl.constexpr,
          STRIDE_X: tl.constexpr, STRIDE_WK: tl.constexpr, STRIDE_WN: tl.constexpr,
          STRIDE_SK: tl.constexpr, STRIDE_SN: tl.constexpr,
          STRIDE_ZK: tl.constexpr, STRIDE_ZN: tl.constexpr,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    ns = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    qns = pn * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)
    ks = tl.arange(0, BLOCK_K)
    gs = tl.arange(0, BLOCK_K // 128)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    # Runtime start pk prevents static expansion of the complete K dimension.
    for block in range(pk, tl.cdiv(K, BLOCK_K), SPLIT_K):
        k = block * BLOCK_K + ks
        group = block * (BLOCK_K // 128) + gs
        qw = tl.load(qw_ptr + k[:, None] * STRIDE_WK + qns[None, :] * STRIDE_WN,
                     (k[:, None] < K) & (qns[None, :] < N // 8), other=0)
        qz = tl.load(qz_ptr + group[:, None] * STRIDE_ZK + qns[None, :] * STRIDE_ZN,
                     (group[:, None] < K // 128) & (qns[None, :] < N // 8), other=0)
        scales = tl.load(s_ptr + group[:, None] * STRIDE_SK + ns[None, :] * STRIDE_SN,
                         (group[:, None] < K // 128) & (ns[None, :] < N), other=0)
        w = _unpack_weight(qw).reshape((BLOCK_K // 128, 128, BLOCK_N))
        z = _unpack_weight(qz)
        # Integer differences are exact; explicitly round scaled weights to
        # FP16 before FP32 multiplication by x. Do not use FP32 dequant weights.
        weight = ((w.to(tl.float32) - z[:, None, :].to(tl.float32)) *
                  scales[:, None, :].to(tl.float32)).to(tl.float16)
        weight = weight.reshape((BLOCK_K, BLOCK_N)).to(tl.float32)
        x = tl.load(x_ptr + k * STRIDE_X, k < K, other=0).to(tl.float32)
        acc += tl.sum(weight * x[:, None], axis=0)
    if SPLIT_K == 1:
        tl.store(out_ptr + ns, acc.to(tl.float16), ns < N)
    else:
        # Each CTA owns this entire slice; no atomics or zero initialization.
        tl.store(partial_ptr + pk * N + ns, acc, ns < N)


@triton.jit
def _sum_partials(partial_ptr, out_ptr, N: tl.constexpr,
                  SPLIT_K: tl.constexpr, BLOCK_N: tl.constexpr):
    ns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    splits = tl.arange(0, SPLIT_K)
    values = tl.load(partial_ptr + splits[:, None] * N + ns[None, :],
                     ns[None, :] < N, other=0)
    result = tl.sum(values, axis=0).to(tl.float16)
    tl.store(out_ptr + ns, result, ns < N)


def _compiled_metadata(kernel):
    metadata = getattr(kernel, 'metadata', None)
    def get(name):
        value = getattr(kernel, name, getattr(metadata, name, None))
        return int(value) if isinstance(value, (int, bool)) else value if isinstance(value, str) else None
    return {name: get(name) for name in ('n_regs', 'n_spills', 'shared', 'num_warps')}


def launch(x, qweight, scales, qzeros, config, *, return_metadata=False):
    m, k, n = _validate(x, qweight, scales, qzeros)
    if m != 1 or not x.is_contiguous() or k < 128 or n < 8:
        raise ValueError('Only contiguous M=1 group128 input is supported')
    bn, bk, split, warps, stages = (config[key] for key in
        ('block_n', 'block_k', 'split_k', 'num_warps', 'compiler_stages'))
    if bn not in (32, 64, 128) or bk not in (128, 256) or split not in (1, 2, 4, 8):
        raise ValueError('Unsupported GEMV tile/split configuration')
    if warps not in (2, 4, 8) or stages != 1:
        raise ValueError('Unsupported GEMV launch options')
    output = scales.new_empty((1, n))
    partial = torch.empty((split, n), device=x.device, dtype=torch.float32) if split > 1 else output
    compiled = _gemv[(triton.cdiv(n, bn), split)](x, qweight, scales, qzeros, output, partial,
        k, n, x.stride(1), qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1), qzeros.stride(0), qzeros.stride(1),
        BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    reduction = None
    if split > 1:
        reduction = _sum_partials[(triton.cdiv(n, 256),)](partial, output, n, split, 256,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    if return_metadata:
        return output, dict(gemv=_compiled_metadata(compiled),
                            reduction=_compiled_metadata(reduction) if reduction is not None else None,
                            schedule=schedule_description(k, n, config))
    return output
