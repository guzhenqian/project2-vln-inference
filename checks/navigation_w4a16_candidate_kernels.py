# Copyright (c) OpenMMLab. All rights reserved.
# Native-kernel launch wrappers are adapted from LMDeploy v0.11.1
# (Apache License 2.0). The installed package is imported, never modified.
"""Isolated W4A16 performance candidates; no model or persistent weight cache.

All variants consume the existing AWQ/GPTQ GEMM-layout asymmetric group128
INT4 weights and FP16 activations. They do not introduce activation quantization.

Call ``candidate_linear(x, qweight, scales, qzeros, variant)`` with a 2D CUDA
FP16 input. ``precision_linear`` is an alias for runtime-wrapper convenience.
Numerical/performance validation belongs to the caller. This module performs
no GPU work at import time and does not install global monkeypatches.

Autotuning must be warmed for each requested M/N/K/split-K combination before
CUDA graph capture or measurement. Dense-prefill variants allocate only a
temporary weight for the projection being executed; no dense weights are kept
on module objects or in a process-wide cache. Memory cost remains measurable.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from lmdeploy.pytorch.kernels.cuda.awq_kernels import (
    _unpack_weight,
    awq_linear,
    awq_linear_kernel,
    get_cuda_autotune_config,
)


VARIANTS = ('stock', 'phase_tuned', 'decode_nosplit', 'prefill_dequant', 'hybrid')
PREFILL_DEQUANT_MIN_ROWS = 128
DECODE_MAX_ROWS = 16

# New wrapper/cache around the unchanged native JIT function. Unlike the
# installed wrapper's N,K-only key, the chosen configuration is specific to M
# and split-K. Reusing the JIT object does not mutate the installed autotuner.
_phase_awq_kernel = triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K', 'SPLIT_K'],
    reset_to_zero=['c_ptr'],
)(awq_linear_kernel.fn)


@triton.jit
def _materialize_weight(
    QW, QZ, S, W,
    K: tl.constexpr, N: tl.constexpr,
    QW_K: tl.constexpr, QW_N: tl.constexpr,
    QZ_K: tl.constexpr, QZ_N: tl.constexpr,
    S_K: tl.constexpr, S_N: tl.constexpr,
    BK: tl.constexpr, BN: tl.constexpr,
):
    """Materialize [N,K] FP16 using the same native packed-INT4 unpacker."""
    block_k = tl.program_id(0)
    block_n = tl.program_id(1)
    k = block_k * BK + tl.arange(0, BK)
    packed_n = block_n * (BN // 8) + tl.arange(0, BN // 8)
    n = block_n * BN + tl.arange(0, BN)
    # BK divides group128, so this tile never crosses a quantization group.
    group = (block_k * BK) // 128
    qw = tl.load(QW + k[:, None] * QW_K + packed_n[None, :] * QW_N,
                 (k[:, None] < K) & (packed_n[None, :] < N // 8), 0)
    qz = tl.load(QZ + group * QZ_K + packed_n * QZ_N,
                 packed_n < N // 8, 0)[None, :]
    scale = tl.load(S + group * S_K + n * S_N, n < N, 0)[None, :]
    # Match the native kernel's FP16 subtract/multiply rounding, rather than
    # materializing FP32 scales or changing the original quantized values.
    weight = (_unpack_weight(qw) - _unpack_weight(qz)) * scale
    tl.store(W + n[None, :] * K + k[:, None], weight,
             (k[:, None] < K) & (n[None, :] < N))


def _validate(x, qweight, scales, qzeros):
    if x.ndim != 2 or x.dtype != torch.float16 or not x.is_cuda:
        raise ValueError('W4A16 candidate requires 2D CUDA FP16 activations')
    m, k = x.shape
    if m < 1 or k % 128 or scales.ndim != 2:
        raise ValueError('Require nonempty activations and asymmetric group128')
    n = scales.shape[1]
    if (n % 8 or qweight.shape != (k, n // 8)
            or qzeros.shape != (k // 128, n // 8)
            or scales.shape != (k // 128, n)
            or qweight.dtype != torch.int32 or qzeros.dtype != torch.int32
            or scales.dtype != torch.float16
            or qweight.device != x.device or qzeros.device != x.device
            or scales.device != x.device):
        raise ValueError('Packed native AWQ/GPTQ group128 layout mismatch')
    return m, k, n


def _native_phase_linear(x, qweight, scales, qzeros, *, no_split_decode):
    m, k = x.shape
    n = scales.shape[1]
    split_k = 1 if no_split_decode and m <= DECODE_MAX_ROWS else max(1, k // 4096)
    output = scales.new_zeros((m, n)) if split_k > 1 else scales.new_empty((m, n))

    def grid(meta):
        return (triton.cdiv(m, meta['BLOCK_SIZE_M']) *
                triton.cdiv(n, meta['BLOCK_SIZE_N']), split_k)

    # Preserve the original architecture-specific software-pipeline choice.
    props = torch.cuda.get_device_properties(x.device)
    stages = 2 if props.major == 9 or (props.major == 8 and props.minor in (6, 9)) else 3
    block_m = max(16, min(128, triton.next_power_of_2(m)))
    _phase_awq_kernel[grid](
        x, qweight, scales, qzeros, output, m, n, k,
        stride_am=x.stride(0), stride_ak=x.stride(1),
        stride_wk=qweight.stride(0), stride_wn=qweight.stride(1),
        stride_sk=scales.stride(0), stride_sn=scales.stride(1),
        stride_zk=qzeros.stride(0), stride_zn=qzeros.stride(1),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_K=128,
        SPLIT_K=split_k, NUM_STAGES=stages,
    )
    return output


def _dense_prefill_linear(x, qweight, scales, qzeros):
    k, n = x.shape[1], scales.shape[1]
    dense_weight = torch.empty((n, k), dtype=torch.float16, device=x.device)
    _materialize_weight[(triton.cdiv(k, 32), triton.cdiv(n, 128))](
        qweight, qzeros, scales, dense_weight, k, n,
        qweight.stride(0), qweight.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        scales.stride(0), scales.stride(1),
        32, 128, num_warps=4, num_stages=1,
    )
    # The temporary is released after this call. Current-stream ordering and
    # PyTorch's caching allocator preserve its lifetime through the GEMM.
    return F.linear(x, dense_weight)


def candidate_linear(x, qweight, scales, qzeros, variant='phase_tuned'):
    """Return [M,N] FP16, preserving the existing packed W4 numerical recipe.

    stock: unchanged installed awq_linear.
    phase_tuned: native JIT, independently tuned for each M/N/K/split-K.
    decode_nosplit: phase_tuned with split-K disabled when M <= 16.
    prefill_dequant: temporary per-projection FP16 GEMM for M >= 128;
        phase_tuned otherwise.
    hybrid: same temporary high-M GEMM; decode_nosplit otherwise.

    Bias/all-reduce and original outer shape are handled by the caller, just
    like the low-level native awq_linear API. No model state is retained.
    """
    if variant not in VARIANTS:
        raise ValueError(f'Unknown W4A16 candidate: {variant}')
    if variant == 'stock':
        return awq_linear(x, qweight, scales, qzeros)
    m, _, _ = _validate(x, qweight, scales, qzeros)
    if variant in ('prefill_dequant', 'hybrid') and m >= PREFILL_DEQUANT_MIN_ROWS:
        return _dense_prefill_linear(x, qweight, scales, qzeros)
    return _native_phase_linear(x, qweight, scales, qzeros,
                                no_split_decode=variant in ('decode_nosplit', 'hybrid'))


precision_linear = candidate_linear
