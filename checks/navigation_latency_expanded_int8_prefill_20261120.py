"""Independent expanded W8A8 prefill arithmetic experiment; no service hooks.

The original group128 W4 checkpoint is dequantized with an FP16 round, then
requantized to symmetric per-output-channel INT8. This is a NEW approximation,
not an equivalent implementation of the original W4A16 arithmetic. Decode is
outside this module. All GPU work is explicit; nothing runs at import time.
"""
import torch
import triton
import triton.language as tl


LAYOUTS = ('kn_contiguous', 'nk_transposed')


@triton.jit(do_not_specialize=['M'])
def _quantize_rows(X, Q, S, M, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BK)
    x = tl.load(X + row * K + col, (row < M) & (col < K), 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    # A zero row has scale one and quantized zeros, including padded rows.
    scale = tl.where(maximum > 0, tl.div_rn(maximum, 127.), 1.)
    rounded = tl.inline_asm_elementwise(
        'cvt.rni.s32.f32 $0, $1;', '=r,f', [tl.div_rn(x, scale)],
        dtype=tl.int32, is_pure=True, pack=1)
    q = tl.minimum(tl.maximum(rounded, -127), 127)
    tl.store(Q + row * K + col, q.to(tl.int8), col < K)
    tl.store(S + row, scale)


@triton.jit(do_not_specialize=['M'])
def _scale_output(I, RS, CS, Y, M, N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < M * N
    row, col = offset // N, offset % N
    integer = tl.load(I + offset, valid, 0).to(tl.float32)
    row_scale = tl.load(RS + row, valid, 1.)
    channel_scale = tl.load(CS + col, valid, 1.)
    # This precise multiplication order is part of the independent protocol.
    result = (integer * row_scale) * channel_scale
    tl.store(Y + offset, result.to(tl.float16), valid)


@triton.jit
def _weight_scales(MAXIMUM, SCALE, N: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    maximum = tl.load(MAXIMUM + col, col < N, 0)
    scale = tl.where(maximum > 0, tl.div_rn(maximum, 127.), 1.)
    tl.store(SCALE + col, scale, col < N)


@triton.jit
def _quantize_weight(W, S, Q, K: tl.constexpr, N: tl.constexpr,
                     NK_LAYOUT: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < K * N
    row, col = offset // N, offset % N
    value = tl.load(W + offset, valid, 0).to(tl.float32)
    scale = tl.load(S + col, valid, 1.)
    rounded = tl.inline_asm_elementwise(
        'cvt.rni.s32.f32 $0, $1;', '=r,f', [tl.div_rn(value, scale)],
        dtype=tl.int32, is_pure=True, pack=1)
    quantized = tl.minimum(tl.maximum(rounded, -127), 127).to(tl.int8)
    destination = col * K + row if NK_LAYOUT else offset
    tl.store(Q + destination, quantized, valid)


def prepare_weight(dense_kn, layout):
    """Startup conversion only. Caller times and accounts for it separately."""
    if layout not in LAYOUTS:
        raise ValueError('Unknown INT8 weight layout')
    if dense_kn.ndim != 2 or dense_kn.dtype != torch.float16 or not dense_kn.is_cuda or not dense_kn.is_contiguous():
        raise ValueError('Expected an already FP16-rounded CUDA [K,N] weight')
    k, n = dense_kn.shape
    if k % 128 or n % 8:
        raise ValueError('Expected group128 K and packed-eight N')
    # FP16 -> FP32 conversion, abs and max are exact; division and rounding use
    # explicit RN operations rather than scalar torch division optimizations.
    maximum = dense_kn.float().abs().amax(dim=0)
    scale = torch.empty((n,),device=dense_kn.device,dtype=torch.float32)
    quantized = (torch.empty((k,n),device=dense_kn.device,dtype=torch.int8) if layout=='kn_contiguous'
                 else torch.empty((n,k),device=dense_kn.device,dtype=torch.int8).T)
    _weight_scales[(triton.cdiv(n,256),)](maximum,scale,n,256,num_warps=4,enable_fp_fusion=False)
    _quantize_weight[(triton.cdiv(k*n,1024),)](dense_kn,scale,quantized,k,n,layout=='nk_transposed',1024,
                                            num_warps=4,enable_fp_fusion=False)
    expected_stride = (n, 1) if layout == 'kn_contiguous' else (1, k)
    if tuple(quantized.stride()) != expected_stride:
        raise RuntimeError('Requested physical INT8 layout was not materialized')
    return quantized, scale


class Pipeline:
    """Preallocated quantization + whole-K cuBLASLt INT8 + FP32 epilogue.

    Padding is an explicit separate choice. The default uses the actual M;
    torch2.5.1 accepts M>16 and K,N divisible by8. No matrix clones or weight
    conversions are part of this hot path. The probe verifies actual support.
    """
    def __init__(self, weight_kn, channel_scale, m, pad_m_multiple=0):
        k, n = weight_kn.shape
        if m <= 16 or k % 8 or n % 8 or pad_m_multiple not in (0, 8, 16):
            raise ValueError('Unsupported matrix shape/padding policy')
        if weight_kn.dtype != torch.int8 or channel_scale.dtype != torch.float32:
            raise ValueError('Expected INT8 weights and FP32 per-channel scales')
        if channel_scale.shape != (n,) or not channel_scale.is_contiguous():
            raise ValueError('Invalid channel scale layout')
        if weight_kn.device != channel_scale.device or not weight_kn.is_cuda:
            raise ValueError('Weight and scales must share a CUDA device')
        if tuple(weight_kn.stride()) not in ((n, 1), (1, k)):
            raise ValueError('Only exact dense KN and NK.T layouts are allowed')
        self.m, self.k, self.n = m, k, n
        self.padded_m = ((m + pad_m_multiple - 1) // pad_m_multiple) * pad_m_multiple if pad_m_multiple else m
        self.pad_m_multiple = pad_m_multiple
        self.weight, self.channel_scale = weight_kn, channel_scale
        self.qx = torch.empty((self.padded_m, k), device=weight_kn.device, dtype=torch.int8)
        self.row_scale = torch.empty((self.padded_m,), device=weight_kn.device, dtype=torch.float32)
        self.integer = torch.empty((self.padded_m, n), device=weight_kn.device, dtype=torch.int32)
        self.output = torch.empty((m, n), device=weight_kn.device, dtype=torch.float16)

    def quantize(self, x):
        if x.shape != (self.m, self.k) or x.dtype != torch.float16 or not x.is_contiguous():
            raise ValueError('Expected contiguous FP16 [actual M,K] input')
        if x.device != self.weight.device:
            raise ValueError('Input and weight devices differ')
        _quantize_rows[(self.padded_m,)](x, self.qx, self.row_scale, self.m, self.k,
            triton.next_power_of_2(self.k), num_warps=8, enable_fp_fusion=False)
        return self.qx

    def gemm(self):
        torch._int_mm(self.qx, self.weight, out=self.integer)
        return self.integer

    def epilogue(self):
        _scale_output[(triton.cdiv(self.m * self.n, 1024),)](
            self.integer, self.row_scale, self.channel_scale, self.output,
            self.m, self.n, 1024, num_warps=4, enable_fp_fusion=False)
        return self.output

    def run(self, x):
        self.quantize(x)
        self.gemm()
        return self.epilogue()

    def storage_receipt(self):
        values = dict(weight=self.weight, channel_scale=self.channel_scale, quantized_activation=self.qx,
                      row_scale=self.row_scale, integer_output=self.integer, fp16_output=self.output)
        return {name: dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                           logical_bytes=t.numel() * t.element_size(),
                           storage_bytes=t.untyped_storage().nbytes()) for name, t in values.items()}
