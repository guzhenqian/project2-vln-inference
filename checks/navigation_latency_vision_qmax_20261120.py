"""Opt-in native vision FlashAttention launch-grid bound, with no tensor math changes.

Set PROJECT2_LATENCY_VISION_QMAX=1 and call install() before constructing the
model. The existing CPU grid supplies one Python max per image prefill. Only
vision FlashAttention instances are wrapped. No GPU scalar reads, device
synchronization, feature caching, token removal, or language changes are added.
Receipts describe host execution; they do not claim CUDA completion or speed.
PROJECT2_LATENCY_VISION_QMAX_VALIDATE=1 additionally enables a diagnostic-only
vision double-forward for the two saved input warmups (indices 0/1, 2/9 images).
That optional numerical gate synchronizes for scalar comparisons; its timing
and peak memory must be excluded from performance evidence. Later shapes never
trigger an additional forward. Language forward still executes exactly once.
"""
from contextvars import ContextVar
import functools
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType


SOURCE_SHA256 = {
    'pytorch/models/qwen3_vl.py': '377b26921dfe42b7d91c2306a908713741fa461a792863b18da0182435127e25',
    'pytorch/models/qwen2_5_vl.py': 'b5dff117a94bef52686a883b4581befd2c0dfc429c4e4784a2ddd26ef93fef99',
    'pytorch/nn/attention.py': '9773ac13b792f6f67f87476b259046e16b5e96f9dd1923ffdd59570c2514a474',
    'pytorch/backends/cuda/flash_attention.py': 'a9ecd1ce93d93c31b4e569f53149e39f7f84a2301bb775935b9cc35f2f25ed18',
    'pytorch/kernels/cuda/flashattention.py': 'fbf78dfe596f6a0d2138378ba5a22a9f0a365d796549ffafe853dd68af0d3fec',
}
_CONTEXT = ContextVar('project2_vision_qmax', default=None)
_INSTALLED = False
_INSTALL_WRITTEN = False
_SHAPES_WRITTEN = set()
_PATCH_SHA256 = None


def _require(value, message):
    if not value:
        raise RuntimeError(message)


def _function(fn):
    return dict(module=fn.__module__, name=fn.__name__, qualname=fn.__qualname__,
                code_filename=fn.__code__.co_filename)


def _receipt(name, fields):
    directory = os.environ.get('PROJECT2_STAGE2_RUN_DIR')
    if not directory:
        return False
    path = Path(directory)/name
    with path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), time_ns=time.time_ns(),
                       patch_module=__name__, patch_module_sha256=_PATCH_SHA256,
                       original_source_sha256=SOURCE_SHA256,
                       enabled_env=os.environ.get('PROJECT2_LATENCY_VISION_QMAX'),
                       validation_enabled_env=os.environ.get('PROJECT2_LATENCY_VISION_QMAX_VALIDATE'),
                       **fields), stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return True


def _grid_context(grid, pixel_values, model):
    _require(grid is not None and grid.device.type == 'cpu', 'Vision qmax requires the native CPU grid_thw')
    _require(len(grid.shape) == 2 and grid.shape[1] == 3, 'Unexpected grid_thw shape')
    rows = grid.tolist()  # The pinned native prepare function already put this tiny tensor on CPU.
    _require(rows and all(len(r) == 3 and all(isinstance(v, int) and v > 0 for v in r) for r in rows),
             'Invalid vision grid values')
    _require(all(t == 1 for t, h, w in rows), 'This candidate is restricted to static images with t=1')
    total = sum(t*h*w for t, h, w in rows)
    _require(pixel_values.shape[0] == total, 'Pixel patch count differs from CPU image grid')
    signature = tuple(tuple(r) for r in rows)
    return dict(model_id=id(model), rows=rows, signature=signature,
                total_patches=total, segments=sum(t for t, h, w in rows),
                max_q_seqlen=max(h*w for t, h, w in rows),
                calls=[] if signature not in _SHAPES_WRITTEN else None)


def _wrap_attention(module, index, model_id, attention_type, impl_type, native_kernel):
    _require(type(module) is attention_type and type(module.impl) is impl_type,
             'Unexpected vision attention module/backend')
    _require(module.impl.flash_attention_fwd is native_kernel, 'Vision attention kernel has already been replaced')
    _require(module.impl.causal is False, 'Vision attention must remain noncausal')
    original = module.forward
    _require(getattr(original, '__func__', None) is attention_type.forward,
             'Vision attention forward has already been wrapped')

    @functools.wraps(original)
    def forward(self, query, key, value, q_start_loc, q_seqlens,
                kv_start_loc=None, kv_seqlens=None, max_q_seqlen=None):
        context = _CONTEXT.get()
        _require(context is not None and context['model_id'] == model_id,
                 'Vision attention called outside its CPU-grid context')
        _require(max_q_seqlen is None, 'Native caller unexpectedly supplied max_q_seqlen')
        _require(len(query.shape) == 3 and query.shape[0] == context['total_patches'],
                 'Vision query shape differs from CPU grid')
        _require(tuple(key.shape) == tuple(query.shape) and tuple(value.shape) == tuple(query.shape),
                 'Unexpected non-self vision attention shapes')
        _require(q_start_loc.numel() == context['segments'] and q_seqlens.numel() == context['segments'],
                 'Vision segment metadata shape differs from CPU grid')
        effective = None if context.get('mode') == 'native_validation' else context['max_q_seqlen']
        result = original(query, key, value, q_start_loc, q_seqlens,
                          kv_start_loc=kv_start_loc, kv_seqlens=kv_seqlens,
                          max_q_seqlen=effective)
        if context['calls'] is not None:
            context['calls'].append(dict(layer=index, query_shape=list(query.shape),
                query_dtype=str(query.dtype), query_device=str(query.device),
                original_default_max_q_seqlen=context['total_patches'],
                effective_max_q_seqlen=context['total_patches'] if effective is None else effective,
                attention_class=type(self).__module__+'.'+type(self).__qualname__,
                backend_class=type(self.impl).__module__+'.'+type(self.impl).__qualname__,
                installed_forward=_function(self.forward),
                original_forward=_function(original), native_kernel=_function(native_kernel)))
        return result

    module.forward = MethodType(forward, module)


def _compare_features(reference, candidate):
    # Called only by the explicitly enabled, excluded numerical warmup gate.
    import torch
    _require(isinstance(reference, tuple) and isinstance(candidate, tuple)
             and len(reference) == len(candidate) == 2, 'Unexpected visual output structure')
    _require(len(reference[1]) == len(candidate[1]) == 3, 'All three DeepStack outputs are required')
    results = []
    for label, old, new in zip(('main', 'deepstack0', 'deepstack1', 'deepstack2'),
                               [reference[0], *reference[1]], [candidate[0], *candidate[1]]):
        _require(torch.is_tensor(old) and torch.is_tensor(new), 'Visual features must be tensors')
        metadata_equal = old.shape == new.shape and old.dtype == new.dtype and old.device == new.device
        finite = bool(torch.isfinite(old).all().item()) and bool(torch.isfinite(new).all().item())
        distinct_storage = old.untyped_storage().data_ptr() != new.untyped_storage().data_ptr()
        equal = bool(torch.equal(old, new)) if metadata_equal else False
        bitwise_equal = bool(torch.equal(old.contiguous().view(torch.uint8),
                                        new.contiguous().view(torch.uint8))) if metadata_equal else False
        max_error = relative_l2 = None
        if metadata_equal and finite:
            delta = new.float()-old.float()
            max_error = float(delta.abs().max().item())
            numerator = float(torch.linalg.vector_norm(delta).item())
            denominator = float(torch.linalg.vector_norm(old.float()).item())
            relative_l2 = numerator/max(denominator, 1e-30)
        results.append(dict(feature=label, reference_shape=list(old.shape), candidate_shape=list(new.shape),
            reference_dtype=str(old.dtype), candidate_dtype=str(new.dtype), reference_device=str(old.device),
            candidate_device=str(new.device), metadata_equal=metadata_equal, finite=finite,
            distinct_storage=distinct_storage, torch_equal=equal, bitwise_equal=bitwise_equal,
            max_abs_error=max_error, relative_l2=relative_l2,
            passed=metadata_equal and finite and distinct_storage and equal and bitwise_equal))
    return dict(passed=all(r['passed'] for r in results), features=results)


def _validation_eligibility(model, context, input_ids):
    index = getattr(model, '_fix_index', None)
    plan = getattr(model, '_fix_plan', None)
    if index not in (0, 1) or not isinstance(plan, list) or len(plan) <= index:
        return False, 'outside saved input warmup indices 0/1'
    spec = plan[index]
    if spec.get('phase') != 'input_warmup' or spec.get('tokens') != 0 or spec.get('sample') != index:
        return False, 'saved request is not the expected natural input warmup'
    if len(context['rows']) != (2 if index == 0 else 9):
        return False, 'grid is outside the explicit 2-image/9-image gate scope'
    directory = getattr(model, '_fix_dir', None)
    if directory is None:
        return False, 'native input SHA warmup directory missing'
    path = Path(directory)/f'input_{input_ids.numel()}.json'
    if not path.is_file():
        return False, 'native input SHA warmup receipt missing'
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('request_index') != index or not all(k in receipt for k in ('input_ids', 'pixel_values')):
        return False, 'native input SHA receipt does not match this first warmup'
    return True, dict(request_index=index, request_spec=spec, input_sha_receipt=str(path))


def _wrap_validation(model, layer_count):
    original = model.visual.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        context = _CONTEXT.get()
        _require(context is not None and context['model_id'] == id(model), 'Missing visual numerical-gate context')
        if not context.get('validate_now'):
            return original(*args, **kwargs)
        reference_context = dict(context, mode='native_validation', calls=[])
        token = _CONTEXT.set(reference_context)
        try:
            reference = original(*args, **kwargs)
        finally:
            _CONTEXT.reset(token)
        candidate = original(*args, **kwargs)
        _require([r['layer'] for r in reference_context['calls']] == list(range(layer_count)),
                 'Native numerical-gate execution did not cover all vision layers once')
        _require([r['layer'] for r in context['calls']] == list(range(layer_count)),
                 'Candidate numerical-gate execution did not cover all vision layers once')
        comparison = _compare_features(reference, candidate)
        filename = f'vision_qmax_numeric_{os.getpid()}_{len(context["rows"]):02d}images.json'
        context['validation'] = dict(status='passed' if comparison['passed'] else 'failed',
            receipt=filename, scope='Excluded first input warmup only; visual double-forward, language once',
            additional_visual_forwards=1, scalar_comparisons_synchronize_cuda=True,
            warmup=context['validation_eligibility'], **comparison)
        _receipt(filename, dict(kind='vision_numerical_gate', grid_thw=context['rows'],
            native_layer_calls=reference_context['calls'], candidate_layer_calls=context['calls'],
            **context['validation']))
        _require(comparison['passed'], 'Vision qmax numerical gate failed; see its receipt')
        return candidate

    model.visual.forward = MethodType(forward, model.visual)


def _prepare_model(model, attention_type, impl_type, native_kernel, validate=False):
    if getattr(model, '_latency_vision_qmax_layers', None) is not None:
        return model._latency_vision_qmax_layers
    blocks = model.visual.blocks
    _require(len(blocks) == 24, 'This candidate requires the pinned 24-layer vision encoder')
    for index, block in enumerate(blocks):
        _wrap_attention(block.attn.attention, index, id(model), attention_type, impl_type, native_kernel)
    if validate:
        _wrap_validation(model, len(blocks))
    model._latency_vision_qmax_layers = len(blocks)
    return len(blocks)


def install():
    global _INSTALLED, _INSTALL_WRITTEN, _PATCH_SHA256
    if _INSTALLED:
        return True
    if os.environ.get('PROJECT2_LATENCY_VISION_QMAX') != '1':
        return False
    from lmdeploy.pytorch.models import qwen3_vl
    from lmdeploy.pytorch.nn.attention import FlashAttention
    from lmdeploy.pytorch.backends.cuda.flash_attention import TritonFlashAttentionImpl
    package = Path(qwen3_vl.__file__).parents[2]
    for relative, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((package/relative).read_bytes()).hexdigest()
        _require(actual == expected, f'Vision qmax source guard failed: {relative} {actual}')
    model_type = qwen3_vl.Qwen3VLForConditionalGeneration
    original = model_type.forward
    _require(Path(original.__code__.co_filename).resolve() == Path(qwen3_vl.__file__).resolve(),
             'Native Qwen3 forward has already been wrapped')
    _PATCH_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    kernel_path = (package/'pytorch/kernels/cuda/flashattention.py').resolve()
    validate = os.environ.get('PROJECT2_LATENCY_VISION_QMAX_VALIDATE') == '1'

    @functools.wraps(original)
    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None,
                inputs_embeds=None, mrope_position_ids=None, pixel_values=None,
                vis_cu_seqlens=None, vis_pos_emb=None, image_mask=None,
                pos_embeds=None, grid_thw=None, **kwargs):
        global _INSTALL_WRITTEN
        args = dict(attn_metadata=attn_metadata, inputs_embeds=inputs_embeds,
                    mrope_position_ids=mrope_position_ids, pixel_values=pixel_values,
                    vis_cu_seqlens=vis_cu_seqlens, vis_pos_emb=vis_pos_emb,
                    image_mask=image_mask, pos_embeds=pos_embeds, grid_thw=grid_thw, **kwargs)
        if pixel_values is None or inputs_embeds is not None:
            return original(self, input_ids, position_ids, past_key_values, **args)
        # Do not import kernels.cuda during install: its package initializer
        # probes the CUDA driver. The native model has already built this exact
        # backend before the first image forward, so inspect its existing fn.
        native_kernel = self.visual.blocks[0].attn.attention.impl.flash_attention_fwd
        _require(native_kernel.__module__ == 'lmdeploy.pytorch.kernels.cuda.flashattention'
                 and native_kernel.__name__ == 'flash_attention_fwd'
                 and Path(native_kernel.__code__.co_filename).resolve() == kernel_path,
                 'Unexpected actual vision FlashAttention kernel function')
        layer_count = _prepare_model(self, FlashAttention, TritonFlashAttentionImpl, native_kernel, validate)
        context = _grid_context(grid_thw, pixel_values, self)
        context['validation'] = dict(status='disabled' if not validate else 'not_repeated_after_first_shape')
        if validate and context['calls'] is not None:
            eligible, reason = _validation_eligibility(self, context, input_ids)
            context['validate_now'] = eligible
            context['validation_eligibility'] = reason
            if not eligible:
                context['validation'] = dict(status='skipped_outside_explicit_warmup_scope', reason=reason)
        token = _CONTEXT.set(context)
        try:
            output = original(self, input_ids, position_ids, past_key_values, **args)
        finally:
            _CONTEXT.reset(token)
        if context['calls'] is not None:
            _require([c['layer'] for c in context['calls']] == list(range(layer_count)),
                     'Actual vision attention execution did not cover each original layer once')
            if not _INSTALL_WRITTEN:
                _INSTALL_WRITTEN = _receipt(f'vision_qmax_install_{os.getpid()}.json', dict(
                    kind='install_receipt', actual_installed_model_forward=_function(model_type.forward),
                    original_model_forward=_function(original)))
            written = _receipt(f'vision_qmax_execute_{os.getpid()}_{len(_SHAPES_WRITTEN):03d}.json', dict(
                kind='first_shape_execution_receipt', scope='Host native forward returned; not a GPU completion timestamp',
                model_class=type(self).__module__+'.'+type(self).__qualname__, grid_thw=context['rows'],
                request_index=getattr(self, '_fix_index', None), input_tokens=input_ids.numel(),
                request_spec=(self._fix_plan[self._fix_index] if getattr(self, '_fix_plan', None)
                              and 0 <= self._fix_index < len(self._fix_plan) else None),
                cpu_grid_reused=True, regular_path_device_scalar_reads_added=0,
                regular_path_added_cuda_synchronizations=0, numerical_gate=context['validation'],
                layer_count=layer_count, total_patches=context['total_patches'], segments=context['segments'],
                effective_max_q_seqlen=context['max_q_seqlen'], actual_layer_calls=context['calls'],
                launch_grid_bound=dict(old_max_q_seqlen=context['total_patches'],
                    new_max_q_seqlen=context['max_q_seqlen'],
                    grid_formula='(ceil(max_q_seqlen/native_BLOCK_M), query_heads, segment_count)',
                    kernel_meta_selection_unchanged=True, per_image_grid_thw=context['rows']),
                change='Only native FlashAttention max_q_seqlen argument changes from total patches to max per-image patches',
                kernel_meta_contract='Pinned CUDA kernel source uses max_seqlen only for grid.x=ceil(max_seqlen/BLOCK_M); no kernel argument or block/warp/stage selection depends on it',
                preserved=['Q/K/V tensors', 'per-image start/length tensors', 'all valid program_id coordinates',
                           'native kernel and BLOCK_M/BLOCK_N/num_warps/num_stages rules',
                           'DeepStack', 'visual tokens and image order', 'language prefill', 'mRoPE', 'FP16 KV']))
            if written:
                _SHAPES_WRITTEN.add(context['signature'])
        return output

    model_type.forward = forward
    _INSTALLED = True
    _INSTALL_WRITTEN = _receipt(f'vision_qmax_install_{os.getpid()}.json', dict(
        kind='install_receipt', actual_installed_model_forward=_function(model_type.forward),
        original_model_forward=_function(original)))
    return True
