# Copyright (c) OpenMMLab. All rights reserved.
# Launch wrapper around the LMDeploy 0.11.1 native JIT (Apache License 2.0).
"""Prepare a finite W4A16 launch table before navigation, then forbid tuning.

The underlying INT4/FP16 arithmetic is the installed native kernel. Actual M
always reaches that kernel; buckets choose launch configurations, never padding
or truncating the input. Preparation owns the only autotuner calls. The generic
JIT disables M specialization so an unseen navigation length cannot compile a
new divisibility/equal-to-one specialization. M=1 retains the original native
JIT and its efficient decode specialization.

Lifecycle in a fresh worker: prepare_projection(qweight, scales, qzeros) for
each loaded fused projection, freeze(), then precision_linear(...). Repeated
shapes are prepared once. This module neither loads models nor patches packages.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import triton

from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear_kernel, get_cuda_autotune_config
from navigation_w4a16_candidate_kernels import _validate


REPRESENTATIVES = (1, 16, 17, 33, 65, 701, 2815)
MAX_ROWS = 3072
# Own both wrapper objects, so freezing their compile methods cannot change the
# installed backend or the separate historical phase_tuned comparison path.
_native_jit = triton.jit(awq_linear_kernel.fn.fn)
_generic_jit = triton.jit(awq_linear_kernel.fn.fn, do_not_specialize=['M'])
_tuners = {
    True: triton.autotune(configs=get_cuda_autotune_config(),
                         key=['M', 'N', 'K', 'SPLIT_K'], reset_to_zero=['c_ptr'])(_native_jit),
    False: triton.autotune(configs=get_cuda_autotune_config(),
                          key=['M', 'N', 'K', 'SPLIT_K'], reset_to_zero=['c_ptr'])(_generic_jit),
}
_table = {}
_frozen = False
_prepare_seconds = 0.0
_identity = {}
_replay_configs = None
_replay_source = None


def validate_table(value, algorithm, model_weights_sha256):
    """Pure schema/identity check; accepts only the current native launch family."""
    if algorithm not in ('awq', 'gptq') or len(model_weights_sha256) != 64:
        raise ValueError('Require an explicit AWQ/GPTQ checkpoint SHA identity')
    if value.get('algorithm') != algorithm or value.get('model_weights_sha256') != model_weights_sha256:
        raise ValueError('Frozen table belongs to another quantization algorithm/checkpoint')
    if (not value.get('frozen') or value.get('representatives') != list(REPRESENTATIVES)
            or value.get('max_rows') != MAX_ROWS or not value.get('actual_m_preserved')
            or not value.get('generic_m_specialization_disabled')
            or value.get('runtime_autotuning_allowed') is not False
            or value.get('runtime_cold_compilation_allowed') is not False):
        raise ValueError('Frozen table execution domain/guard schema differs')
    allowed = [c.all_kwargs() for c in get_cuda_autotune_config()]
    result = {}
    for row in value['shapes']:
        k, n = row['k'], row['n']
        if (type(k) is not int or type(n) is not int or k <= 0 or k % 128 or n <= 0 or n % 8
                or row['split_k'] != max(1, k // 4096)):
            raise ValueError('Malformed projection geometry in frozen table')
        key = (row['device'], k, n, tuple(row['weight_stride']), tuple(row['scales_stride']), tuple(row['zeros_stride']))
        if key in result or set(row['configs']) != {str(m) for m in REPRESENTATIVES}:
            raise ValueError('Duplicate/incomplete frozen projection table')
        if any(config not in allowed for config in row['configs'].values()):
            raise ValueError('Frozen configuration is outside the native candidate family')
        result[key] = copy.deepcopy(row['configs'])
    if len(result) != 4:
        raise ValueError('Frozen table must contain exactly four fused projection layouts')
    return result


def load_launch_table(path, algorithm, model_weights_sha256):
    """Load an explicit prior20 receipt/table before any projection preparation.

    A wrapped repair_execution_audit may obtain missing legacy identity fields
    from its adjacent launch.json, which is hashed as provenance. A standalone
    table must carry identity itself. Neither path runs an autotuner.
    """
    global _replay_configs, _replay_source, _identity
    if _frozen or _table or _replay_configs is not None:
        raise RuntimeError('Load one frozen table before preparing any projection')
    path = Path(path).resolve()
    raw = path.read_bytes()
    value = json.loads(raw)
    table = copy.deepcopy(value.get('frozen_launch_table', value))
    identity_source = None
    if 'algorithm' not in table or 'model_weights_sha256' not in table:
        if 'frozen_launch_table' not in value:
            raise ValueError('Standalone frozen table is missing algorithm/checkpoint identity')
        launch_path = path.parent / 'launch.json'
        launch_raw = launch_path.read_bytes()
        launch = json.loads(launch_raw)
        table.setdefault('algorithm', launch['recipe']['method'])
        table.setdefault('model_weights_sha256', launch['model_weights_sha256'])
        identity_source = dict(path=str(launch_path), sha256=hashlib.sha256(launch_raw).hexdigest())
    _replay_configs = validate_table(table, algorithm, model_weights_sha256)
    _identity = dict(algorithm=algorithm, model_weights_sha256=model_weights_sha256)
    _replay_source = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), identity_source=identity_source)
    return copy.deepcopy(_replay_source)


def _config_from_kwargs(value):
    value = dict(value)
    options = {key:value.pop(key) for key in ('num_warps', 'num_stages', 'num_ctas', 'maxnreg') if key in value}
    return triton.Config(value, **options)


def representative(m):
    if not 1 <= m <= MAX_ROWS:
        raise ValueError('Frozen navigation launch domain is 1..3072 rows')
    if m == 1:
        return 1
    if m <= 16:
        return 16
    if m <= 32:
        return 17
    if m <= 64:
        return 33
    if m <= 128:
        return 65
    return 701 if m <= 1758 else 2815


def _shape_key(qweight, scales, qzeros):
    return (qweight.device.index, qweight.shape[0], scales.shape[1],
            tuple(qweight.stride()), tuple(scales.stride()), tuple(qzeros.stride()))


def _launch(x, qweight, scales, qzeros, kernel, config=None):
    m, k = x.shape
    n = scales.shape[1]
    split = max(1, k // 4096)
    out = scales.new_zeros((m, n)) if split > 1 else scales.new_empty((m, n))
    props = torch.cuda.get_device_properties(x.device)
    stages = 2 if props.major == 9 or (props.major == 8 and props.minor in (6, 9)) else 3
    kwargs = dict(stride_am=x.stride(0), stride_ak=x.stride(1),
        stride_wk=qweight.stride(0), stride_wn=qweight.stride(1),
        stride_sk=scales.stride(0), stride_sn=scales.stride(1),
        stride_zk=qzeros.stride(0), stride_zn=qzeros.stride(1),
        stride_cm=out.stride(0), stride_cn=out.stride(1),
        BLOCK_SIZE_M=max(16, min(128, triton.next_power_of_2(m))),
        BLOCK_SIZE_K=128, SPLIT_K=split, NUM_STAGES=stages)
    if config is not None:
        if config.pre_hook is not None:
            raise RuntimeError('Unexpected native configuration pre_hook')
        kwargs.update(config.all_kwargs())

    def grid(meta):
        return (triton.cdiv(m, meta['BLOCK_SIZE_M']) * triton.cdiv(n, meta['BLOCK_SIZE_N']), split)

    kernel[grid](x, qweight, scales, qzeros, out, m, n, k, **kwargs)
    return out


def prepare_projection(qweight, scales, qzeros):
    """Tune all fixed representatives for one real loaded projection shape.

    Call only after weights are loaded, outside capture and request timers.
    Returns False for an already prepared layout and True for new preparation.
    """
    global _prepare_seconds
    if _frozen:
        raise RuntimeError('Frozen launch table cannot be extended')
    key = _shape_key(qweight, scales, qzeros)
    if key in _table:
        return False
    if _replay_configs is not None and key not in _replay_configs:
        raise ValueError('Loaded projection does not match the explicit frozen table')
    started = time.perf_counter()
    configs = {}
    with torch.inference_mode():
        for m in REPRESENTATIVES:
            x = torch.zeros((m, qweight.shape[0]), device=qweight.device, dtype=torch.float16)
            _validate(x, qweight, scales, qzeros)
            if _replay_configs is None:
                tuner = _tuners[m == 1]
                _launch(x, qweight, scales, qzeros, tuner)
                config = copy.deepcopy(tuner.best_config)
            else:
                config = _config_from_kwargs(_replay_configs[key][str(m)])
            configs[m] = config
            # Compile/warm the exact direct-call path before freezing.
            _launch(x, qweight, scales, qzeros, _native_jit if m == 1 else _generic_jit, config)
        torch.cuda.synchronize(qweight.device)
    _table[key] = configs
    _prepare_seconds += time.perf_counter() - started
    return True


def freeze(algorithm=None, model_weights_sha256=None):
    """Seal preparation and return a JSON-serializable launch-table receipt."""
    global _frozen, _identity
    if not _table:
        raise RuntimeError('Cannot freeze an empty launch table')
    if _replay_configs is not None and set(_table) != set(_replay_configs):
        raise ValueError('Prepared layouts do not exactly match the frozen source table')
    if algorithm is not None or model_weights_sha256 is not None:
        identity = dict(algorithm=algorithm, model_weights_sha256=model_weights_sha256)
        if _identity and identity != _identity:
            raise ValueError('Identity differs from frozen table at freeze')
        _identity = identity
    def reject_cold_compile(*args, **kwargs):
        raise RuntimeError('Unprepared W4A16 specialization after freeze; no timed cold compile allowed')
    _native_jit.compile = reject_cold_compile
    _generic_jit.compile = reject_cold_compile
    _frozen = True
    return snapshot()


def snapshot():
    rows = []
    for key, configs in sorted(_table.items()):
        device, k, n, ws, ss, zs = key
        rows.append(dict(device=device, k=k, n=n, weight_stride=list(ws),
            scales_stride=list(ss), zeros_stride=list(zs), split_k=max(1, k // 4096),
            configs={str(m): c.all_kwargs() for m, c in configs.items()}))
    return dict(frozen=_frozen, representatives=list(REPRESENTATIVES), max_rows=MAX_ROWS,
        **_identity, selection_mode='replayed_frozen_table' if _replay_configs is not None else 'startup_autotune',
        frozen_table_source=copy.deepcopy(_replay_source),
        selection='M1 native decode; M2..16->16;17..32->17;33..64->33;65..128->65;129..1758->701;1759..3072->2815',
        actual_m_preserved=True, generic_m_specialization_disabled=True,
        runtime_autotuning_allowed=False, prepare_seconds=_prepare_seconds, shapes=rows,
        runtime_cold_compilation_allowed=False,
        compiled_entries={name:sum(len(cache) for cache in kernel.cache.values())
                          for name,kernel in [('decode', _native_jit), ('generic', _generic_jit)]},
        autotuner_cache_entries={str(k): len(v.cache) for k, v in _tuners.items()},
        dtype='packed INT4 weights, FP16 activations and output; native FP32 dot accumulator')


def precision_linear(x, qweight, scales, qzeros):
    if not _frozen:
        raise RuntimeError('Prepare and freeze W4A16 launch table before inference')
    if not x.is_contiguous():
        raise ValueError('Require contiguous FP16 projection activations')
    m, _, _ = _validate(x, qweight, scales, qzeros)
    key = _shape_key(qweight, scales, qzeros)
    if key not in _table:
        raise RuntimeError('Unknown projection layout after freeze: ' + repr(key))
    config = _table[key][representative(m)]
    return _launch(x, qweight, scales, qzeros, _native_jit if m == 1 else _generic_jit, config)


def install_for_model(model):
    """Prepare all loaded W4 modules and install the native backend wrapper.

    Invoke once in the worker before its first synthetic forward, after loading
    weights and before graph capture. The returned receipt is safe for JSON.
    """
    import lmdeploy.pytorch.backends.cuda.awq_modules as backend
    objects = model._awq_objects
    if len(objects) != 144 or model._awq_loaded_weights != model._awq_expected_weights:
        raise ValueError('Require all 144 fused modules and their loaded W4 tensors')
    launch = json.loads((Path(os.environ['PROJECT2_STAGE2_RUN_DIR']) / 'launch.json').read_text())
    if launch['recipe']['method'] != model._method or launch['recipe']['weight_bits'] != 4 or launch['recipe']['activation_bits'] != 16:
        raise ValueError('Service launch identity does not match W4A16 model')
    source = os.environ.get('PROJECT2_W4A16_REPAIR_TABLE')
    if source:
        load_launch_table(source, model._method, launch['model_weights_sha256'])
    for module in objects.values():
        prepare_projection(module.qweight, module.scales, module.qzeros)
    receipt = freeze(model._method, launch['model_weights_sha256'])
    if len(receipt['shapes']) != 4:
        raise ValueError('Expected four unique fused projection layouts')

    def frozen_wq_gemm(x, qweight, qzeros, scales, w_bit=4, group_size=128,
                       bias=None, out_features=0):
        if w_bit != 4 or group_size != 128 or out_features != scales.size(1):
            raise ValueError('Only native group128 W4A16 projections are supported')
        shape = x.shape[:-1] + (out_features,)
        output = precision_linear(x.flatten(0, -2).contiguous(), qweight, scales, qzeros)
        if bias is not None:
            output = output + bias
        output = output.reshape(shape)
        return output.unsqueeze(0) if output.ndim == 2 else output

    backend.wq_gemm_forward = frozen_wq_gemm
    return dict(receipt, loaded_module_count=len(objects), wrapper='frozen_wq_gemm',
                native_shape_and_bias_protocol_preserved=True)
