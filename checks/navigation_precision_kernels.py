"""Experimental packed integer kernels for the isolated precision study.

W4A4 uses packed INT4 inputs, widened to INT8 in registers for integer MMA.
It does NOT claim native INT4 Tensor Core acceleration. W4A8 uses INT8 MMA.
FP16 is retained between projections, in attention/normalization, and in vision.
The existing W4A16 kernel and frozen experiments are untouched.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, BITS: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    limit: tl.constexpr = (1 << (BITS - 1)) - 1
    scale = tl.maximum(tl.max(tl.abs(x), 0) * (1.0 / limit), 1.e-8)
    # CUDA round-to-nearest-even, matching torch.round used by the reference.
    rounded = tl.inline_asm_elementwise('cvt.rni.s32.f32 $0, $1;', '=r,f', [tl.div_rn(x, scale)],
                                       dtype=tl.int32, is_pure=True, pack=1)
    q = tl.minimum(tl.maximum(rounded, -limit), limit)
    tl.store(S + row, scale)
    if BITS == 8:
        tl.store(Q + row * K + col, q.to(tl.int8), col < K)
    else:
        q4 = (q & 15).reshape((BK // 8, 8))
        shift = tl.arange(0, 8) * 4
        packed = tl.sum(q4 << shift[None, :], axis=1).to(tl.int32)
        idx = tl.arange(0, BK // 8)
        tl.store(Q + row * (K // 8) + idx, packed, idx < K // 8)


@triton.jit
def _linear(A, AS, W, Z, S, Y, M, K: tl.constexpr, N: tl.constexpr,
            WB: tl.constexpr, AB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, 128)
    wp: tl.constexpr = 32 // WB
    if WB == 4:
        # AWQ interleaved nibble layout [0,2,4,6,1,3,5,7].
        shift = ((cols % 8) // 2 + (cols % 2) * 4) * 4
    else:
        shift = (cols % 4) * 8
    acc = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 128):
        k = group * 128 + ks
        w = tl.load(W + k[:, None] * (N // wp) + cols[None, :] // wp,
                    cols[None, :] < N, 0)
        z = tl.load(Z + group * (N // wp) + cols // wp, cols < N, 0)
        wi = (w >> shift[None, :]) & ((1 << WB) - 1)
        zi = (z >> shift) & ((1 << WB) - 1)
        scale = tl.load(S + group * N + cols, cols < N, 0).to(tl.float32)
        if AB == 16:
            a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, 0)
            b = ((wi - zi[None, :]).to(tl.float16) * scale[None, :].to(tl.float16)).to(tl.float16)
            acc = tl.dot(a, b, acc)
        else:
            if AB == 8:
                a = tl.load(A + rows[:, None] * K + k[None, :], rows[:, None] < M, 0)
            else:
                packed = tl.load(A + rows[:, None] * (K // 8) + k[None, :] // 8,
                                 rows[:, None] < M, 0)
                unsigned = (packed >> ((k[None, :] % 8) * 4)) & 15
                a = tl.where(unsigned >= 8, unsigned - 16, unsigned).to(tl.int8)
            b = (wi - zi[None, :]).to(tl.int8)
            partial = tl.dot(a, b, out_dtype=tl.int32)
            acc += partial.to(tl.float32) * scale[None, :]
    if AB != 16:
        act_scale = tl.load(AS + rows, rows < M, 0)
        acc *= act_scale[:, None]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
             (rows[:, None] < M) & (cols[None, :] < N))


def quantize_activations(x, bits):
    if x.dtype != torch.float16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('Require contiguous CUDA FP16 projection input')
    m, k = x.shape
    if bits not in (4, 8) or k % 128:
        raise ValueError('Require INT4/INT8 and input width multiple of128')
    q = torch.empty((m, k if bits == 8 else k // 8), device=x.device,
                    dtype=torch.int8 if bits == 8 else torch.int32)
    scales = torch.empty(m, device=x.device, dtype=torch.float32)
    _quantize_rows[(m,)](x, q, scales, k, bits, triton.next_power_of_2(k))
    return q, scales


def precision_linear(x, qweight, scales, qzeros, weight_bits, activation_bits):
    if (weight_bits, activation_bits) not in ((8, 16), (4, 8), (4, 4)):
        raise ValueError('Only the three explicitly requested new precision configurations')
    x = x.contiguous()
    m, k = x.shape
    n = scales.shape[1]
    if (x.dtype != torch.float16 or qweight.dtype != torch.int32 or qzeros.dtype != torch.int32
            or scales.dtype != torch.float16 or qweight.shape != (k, n // (32 // weight_bits))
            or qzeros.shape != (k // 128, n // (32 // weight_bits))
            or scales.shape != (k // 128, n) or k % 128 or n % 64):
        raise ValueError('Packed tensor shape/dtype contract violated')
    a, ascales = (x, x) if activation_bits == 16 else quantize_activations(x, activation_bits)
    out = torch.empty((m, n), device=x.device, dtype=torch.float16)
    bm = 16 if m <= 16 else 32
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, 64))](
        a, ascales, qweight, qzeros, scales, out, m, k, n, weight_bits, activation_bits, bm, 64,
        num_warps=4, num_stages=2)
    return out
