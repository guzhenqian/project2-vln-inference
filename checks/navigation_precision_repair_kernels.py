"""Isolated precision repair candidates; no import-time GPU work or autotuning.

W4A4 uses dynamic symmetric per-token/per-K128 activation scales and packed
INT4 storage widened to INT8 MMA. W4A8 retains the old per-token maxabs recipe
and uses INT8 MMA. W8A16 retains byte-packed asymmetric weights and FP16 MMA.
Vision, inter-projection tensors and the caller's original checkpoints are not
changed. These are project kernels, not claims of stock backend support.

All launch choices come from a fixed small table; no exact-M autotune can run
inside a timed request. A caller must warm the selected variant before graph
capture. Numerical, performance and navigation acceptance remain separate.
"""
import hashlib

import torch
import triton
import triton.language as tl


VARIANTS = ('balanced', 'wide_m', 'wide_n', 'split8')
ACTIVATION_RECIPES = {
    4: 'dynamic_per_token_per_group128_symmetric_maxabs_narrow_range',
    8: 'dynamic_per_token_symmetric_maxabs_narrow_range',
    16: 'none',
}
PREPARE_ROWS = (1, 16, 17, 701, 2815, 2816)
MAX_ROWS = 3072
_PREPARED = {}
_PREPARED_QUANT = set()
_FROZEN = False
_FROZEN_CACHE = None
_PREVIOUS_CACHE_HOOK = None
_CACHE_HOOK_INSTALLED = False


@triton.jit
def _quantize_a4_groups(X, Q, S, K: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    offset = group * 128 + tl.arange(0, 128)
    x = tl.load(X + row * K + offset).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) * (1.0 / 7), 1.e-8)
    rounded = tl.inline_asm_elementwise(
        'cvt.rni.s32.f32 $0, $1;', '=r,f', [tl.div_rn(x, scale)],
        dtype=tl.int32, is_pure=True, pack=1)
    q = tl.minimum(tl.maximum(rounded, -7), 7)
    packed = tl.sum((q & 15).reshape((16, 8)) << (tl.arange(0, 8)[None, :] * 4), 1)
    tl.store(Q + row * (K // 8) + group * 16 + tl.arange(0, 16), packed.to(tl.int32))
    tl.store(S + row * (K // 128) + group, scale)


@triton.jit
def _quantize_a8_rows(X, Q, S, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) * (1.0 / 127), 1.e-8)
    rounded = tl.inline_asm_elementwise(
        'cvt.rni.s32.f32 $0, $1;', '=r,f', [tl.div_rn(x, scale)],
        dtype=tl.int32, is_pure=True, pack=1)
    q = tl.minimum(tl.maximum(rounded, -127), 127)
    tl.store(Q + row * K + col, q.to(tl.int8), col < K)
    tl.store(S + row, scale)


@triton.jit
def _unpack_weight_integer(packed, WB: tl.constexpr):
    width: tl.constexpr = 32 // WB
    column = tl.arange(0, width)
    if WB == 4:
        shift = (column // 2 + (column % 2) * 4) * 4
    else:
        shift = column * 8
    values = (packed[:, :, None] >> shift[None, None, :]) & ((1 << WB) - 1)
    return values.reshape((packed.shape[0], packed.shape[1] * width))


@triton.jit(do_not_specialize=['M'])
def _linear_repair(A, AS, W, Z, S, Y, M, K: tl.constexpr, N: tl.constexpr,
                   WB: tl.constexpr, AB: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, SPLIT_K: tl.constexpr):
    # Bounded groups of M tiles improve packed-weight reuse across rows.
    pid, split = tl.program_id(0), tl.program_id(1)
    m_tiles, n_tiles = tl.cdiv(M, BM), tl.cdiv(N, BN)
    tile_group = pid // (8 * n_tiles)
    first_m = tile_group * 8
    group_m = tl.minimum(m_tiles - first_m, 8)
    within_group = pid % (8 * n_tiles)
    row_tile = first_m + within_group % group_m
    column_tile = within_group // group_m
    rows = row_tile * BM + tl.arange(0, BM)
    cols = column_tile * BN + tl.arange(0, BN)
    ks = tl.arange(0, 128)
    wp: tl.constexpr = 32 // WB
    groups: tl.constexpr = K // 128
    groups_per_split: tl.constexpr = (groups + SPLIT_K - 1) // SPLIT_K
    packed_cols = column_tile * (BN // wp) + tl.arange(0, BN // wp)
    acc = tl.zeros((BM, BN), tl.float32)
    for index in range(groups_per_split):
        group = split * groups_per_split + index
        valid_group = group < groups
        k = group * 128 + ks
        w = tl.load(W + k[:, None] * (N // wp) + packed_cols[None, :],
                    valid_group & (packed_cols[None, :] < N // wp), 0)
        z = tl.load(Z + group * (N // wp) + packed_cols,
                    valid_group & (packed_cols < N // wp), 0)[None, :]
        wi = _unpack_weight_integer(w, WB)
        zi = _unpack_weight_integer(z, WB)
        ws = tl.load(S + group * N + cols, valid_group & (cols < N), 0)
        if AB == 16:
            a = tl.load(A + rows[:, None] * K + k[None, :],
                        (rows[:, None] < M) & valid_group, 0)
            # Preserve the original W8 FP16 dequantization rounding.
            b = ((wi - zi).to(tl.float16) * ws[None, :]).to(tl.float16)
            acc = tl.dot(a, b, acc)
        else:
            if AB == 8:
                a = tl.load(A + rows[:, None] * K + k[None, :],
                            (rows[:, None] < M) & valid_group, 0)
            else:
                packed_k = group * 16 + tl.arange(0, 16)
                packed = tl.load(A + rows[:, None] * (K // 8) + packed_k[None, :],
                                 (rows[:, None] < M) & valid_group, 0)
                unsigned = ((packed[:, :, None] >> (tl.arange(0, 8)[None, None, :] * 4)) & 15).reshape((BM, 128))
                a = tl.where(unsigned >= 8, unsigned - 16, unsigned).to(tl.int8)
            b = (wi - zi).to(tl.int8)
            partial = tl.dot(a, b, out_dtype=tl.int32).to(tl.float32)
            if AB == 4:
                act_scale = tl.load(AS + rows * groups + group,
                                    (rows < M) & valid_group, 0)
                acc += partial * ws[None, :].to(tl.float32) * act_scale[:, None]
            else:
                acc += partial * ws[None, :].to(tl.float32)
    if AB == 8:
        act_scale = tl.load(AS + rows, rows < M, 0)
        acc *= act_scale[:, None]
    if SPLIT_K == 1:
        tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
                 (rows[:, None] < M) & (cols[None, :] < N))
    else:
        tl.store(Y + split * M * N + rows[:, None] * N + cols[None, :], acc,
                 (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit(do_not_specialize=['SIZE'])
def _reduce_splits(P, Y, SIZE, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in range(SPLIT_K):
        acc += tl.load(P + split * SIZE + offset, offset < SIZE, 0)
    tl.store(Y + offset, acc.to(tl.float16), offset < SIZE)


def launch_config(m, k, variant='balanced'):
    """Finite shape buckets, never an online search or exact-M tuning cache."""
    if variant not in VARIANTS:
        raise ValueError('Unknown precision repair variant: ' + str(variant))
    if m <= 16:
        split = min(8 if variant == 'split8' else 4, k // 128)
        return dict(bm=16, bn=64, split_k=max(1, split), num_warps=4, num_stages=2)
    if m <= 64:
        return dict(bm=32, bn=64, split_k=1, num_warps=4, num_stages=2)
    if variant == 'wide_m':
        return dict(bm=128, bn=128, split_k=1, num_warps=8, num_stages=1)
    if variant == 'wide_n':
        return dict(bm=64, bn=256, split_k=1, num_warps=8, num_stages=1)
    return dict(bm=64, bn=128, split_k=1, num_warps=4, num_stages=2)


def _require_alignment(*tensors):
    # Triton specializes pointer alignment. All model/internal buffers must use
    # the same16-byte-aligned category that prepare_projection warms. Contiguous
    # offset views can still be misaligned, so is_contiguous alone is insufficient.
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        raise ValueError('Precision repair requires16-byte-aligned tensor pointers')


def _projection_key(qweight, scales, qzeros, weight_bits, activation_bits, variant):
    return (str(qweight.device), qweight.shape[0], scales.shape[1], weight_bits,
            activation_bits, variant, tuple(qweight.stride()), tuple(scales.stride()),
            tuple(qzeros.stride()), str(qweight.dtype), str(scales.dtype), str(qzeros.dtype))


def _require_prepared(qweight, scales, qzeros, weight_bits, activation_bits, variant):
    if _FROZEN and _projection_key(qweight, scales, qzeros, weight_bits, activation_bits, variant) not in _PREPARED:
        raise RuntimeError('Unprepared projection geometry/precision/variant after freeze')


def _private_jits():
    return (_quantize_a4_groups, _quantize_a8_rows, _linear_repair, _reduce_splits)


def _cache_inventory():
    result = {}
    for kernel in _private_jits():
        keys = []
        for device, entries in kernel.cache.items():
            for key in entries:
                keys.append(hashlib.sha256((str(device) + ':' + str(key)).encode()).hexdigest())
        result[kernel.__name__] = dict(count=len(keys), key_sha256=sorted(keys))
    return result


def _reject_private_cold_compile(**kwargs):
    info = kwargs.get('fn')
    jit = getattr(info, 'jit_function', None)
    if _FROZEN and any(jit is candidate for candidate in _private_jits()):
        raise RuntimeError('Frozen precision repair rejected a cold JIT launch: ' + str(getattr(info, 'name', '?')))
    if _PREVIOUS_CACHE_HOOK is not None:
        return _PREVIOUS_CACHE_HOOK(**kwargs)
    return False


def snapshot():
    """Read-only CPU evidence; does not synchronize, initialize or execute CUDA."""
    cache = _cache_inventory()
    return dict(frozen=_FROZEN, prepared_projection_count=len(_PREPARED),
                prepared_projections=list(_PREPARED.values()), cache=cache,
                cache_unchanged_since_freeze=(cache == _FROZEN_CACHE) if _FROZEN else None,
                private_cold_compile_guard_installed=_CACHE_HOOK_INSTALLED,
                allowed_dynamic_rows=[1, MAX_ROWS], pointer_alignment_bytes=16,
                disabled_specialization=dict(_linear_repair=['M'], _reduce_splits=['SIZE']))


def prepare_projection(qweight, scales, qzeros, weight_bits, activation_bits, variant='balanced'):
    """Explicitly warm all fixed launch buckets before any fullmodel forward.

    Invoke once for each module's real packed tensors. Identical geometries are
    deduplicated, so the model's144 modules usually require four actual warms.
    This function only prepares kernels; it does not validate model quality.
    """
    key = _projection_key(qweight, scales, qzeros, weight_bits, activation_bits, variant)
    if key in _PREPARED:
        return dict(_PREPARED[key], already_prepared=True)
    if _FROZEN:
        raise RuntimeError('Cannot prepare a new projection after freeze')
    _require_alignment(qweight, scales, qzeros)
    before = _cache_inventory()
    with torch.inference_mode():
        for rows in PREPARE_ROWS:
            x = torch.zeros((rows, qweight.shape[0]), dtype=torch.float16, device=qweight.device)
            result = precision_linear(x, qweight, scales, qzeros, weight_bits, activation_bits, variant)
            del result, x
        torch.cuda.synchronize(qweight.device)
    evidence = dict(device=str(qweight.device), k=qweight.shape[0], n=scales.shape[1],
                    weight_bits=weight_bits, activation_bits=activation_bits, variant=variant,
                    prepared_rows=list(PREPARE_ROWS), continuous_strides=True,
                    qweight_stride=list(qweight.stride()), scales_stride=list(scales.stride()),
                    qzeros_stride=list(qzeros.stride()), pointer_alignment_bytes=16,
                    cache_counts_before={name: value['count'] for name, value in before.items()},
                    cache_counts_after={name: value['count'] for name, value in _cache_inventory().items()})
    _PREPARED[key] = evidence
    if activation_bits < 16:
        _PREPARED_QUANT.add((str(qweight.device), qweight.shape[0], activation_bits))
    return dict(evidence, already_prepared=False)


def freeze():
    """Seal prepared geometries and reject private cache misses BEFORE compilation.

    Triton's process hook is scoped to this module's four launched JIT kernels;
    other models/kernels retain the previously installed hook and behavior.
    """
    global _FROZEN, _FROZEN_CACHE, _PREVIOUS_CACHE_HOOK, _CACHE_HOOK_INSTALLED
    if _FROZEN:
        if _cache_inventory() != _FROZEN_CACHE:
            raise RuntimeError('Private kernel cache changed after freeze')
        return snapshot()
    if not _PREPARED:
        raise RuntimeError('prepare_projection must precede freeze')
    from triton.runtime.jit import JITFunction
    _PREVIOUS_CACHE_HOOK = JITFunction.cache_hook
    JITFunction.cache_hook = _reject_private_cold_compile
    _CACHE_HOOK_INSTALLED = True
    _FROZEN_CACHE = _cache_inventory()
    _FROZEN = True
    return snapshot()


def quantize_activations(x, bits):
    if (x.ndim != 2 or x.dtype != torch.float16 or not x.is_cuda
            or not x.is_contiguous() or x.shape[0] < 1 or x.shape[1] % 128):
        raise ValueError('Require nonempty contiguous CUDA FP16 [M,K], K multiple of128')
    m, k = x.shape
    if m > MAX_ROWS:
        raise ValueError('Precision repair is limited to3072 input rows')
    _require_alignment(x)
    if _FROZEN and (str(x.device), k, bits) not in _PREPARED_QUANT:
        raise RuntimeError('Unprepared activation geometry/precision after freeze')
    if bits == 4:
        q = torch.empty((m, k // 8), device=x.device, dtype=torch.int32)
        scales = torch.empty((m, k // 128), device=x.device, dtype=torch.float32)
        _quantize_a4_groups[(m, k // 128)](x, q, scales, k, num_warps=4)
    elif bits == 8:
        q = torch.empty((m, k), device=x.device, dtype=torch.int8)
        scales = torch.empty(m, device=x.device, dtype=torch.float32)
        _quantize_a8_rows[(m,)](x, q, scales, k, triton.next_power_of_2(k))
    else:
        raise ValueError('Activation quantization requires4 or8 bits')
    return q, scales


def _validate(x, qweight, scales, qzeros, weight_bits, activation_bits):
    if (weight_bits, activation_bits) not in ((4, 4), (4, 8), (8, 16)):
        raise ValueError('Only W4A4/W4A8/W8A16 are supported')
    if x.ndim != 2 or x.dtype != torch.float16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError('Require contiguous2D CUDA FP16 input')
    m, k = x.shape
    n = scales.shape[1] if scales.ndim == 2 else 0
    if (m < 1 or m > MAX_ROWS or k % 128 or n < 1 or n % 64
            or qweight.shape != (k, n // (32 // weight_bits))
            or qzeros.shape != (k // 128, n // (32 // weight_bits))
            or scales.shape != (k // 128, n)
            or qweight.dtype != torch.int32 or qzeros.dtype != torch.int32
            or scales.dtype != torch.float16
            or any(t.device != x.device or not t.is_contiguous() for t in (qweight, qzeros, scales))):
        raise ValueError('Packed group128 tensor contract violated')
    _require_alignment(x, qweight, scales, qzeros)
    return m, k, n


def linear_from_quantized(a, activation_scales, qweight, scales, qzeros,
                          m, k, weight_bits, activation_bits, variant='balanced'):
    """Internal split timing entry; callers must validate storage with precision_linear first."""
    n = scales.shape[1]
    if m < 1 or m > MAX_ROWS:
        raise ValueError('Precision repair is limited to1..3072 input rows')
    _require_alignment(a, activation_scales, qweight, scales, qzeros)
    _require_prepared(qweight, scales, qzeros, weight_bits, activation_bits, variant)
    cfg = launch_config(m, k, variant)
    out = torch.empty((m, n), device=a.device, dtype=torch.float16)
    split = cfg['split_k']
    target = (torch.empty((split, m, n), device=a.device, dtype=torch.float32)
              if split > 1 else out)
    _linear_repair[(triton.cdiv(m, cfg['bm']) * triton.cdiv(n, cfg['bn']), split)](
        a, activation_scales, qweight, qzeros, scales, target, m, k, n,
        weight_bits, activation_bits, cfg['bm'], cfg['bn'], split,
        num_warps=cfg['num_warps'], num_stages=cfg['num_stages'])
    if split > 1:
        _reduce_splits[(triton.cdiv(m * n, 256),)](target, out, m * n, split, 256)
    return out


def precision_linear(x, qweight, scales, qzeros, weight_bits, activation_bits,
                     variant='balanced'):
    x = x.contiguous()
    m, k, _ = _validate(x, qweight, scales, qzeros, weight_bits, activation_bits)
    a, activation_scales = ((x, x) if activation_bits == 16
                            else quantize_activations(x, activation_bits))
    return linear_from_quantized(a, activation_scales, qweight, scales, qzeros,
                                 m, k, weight_bits, activation_bits, variant)
