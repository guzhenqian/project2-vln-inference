"""Isolated LMDeploy 0.11.1 AWQ plus layer-internal DART adapter.

The default short profile accepts two images and <=1024 prompt tokens. The
navigation profile accepts 2..9 images and <=3072 prompt tokens. Both require
one complete, fresh prefill in a single-sequence eager engine. Low layers keep
the original contiguous KV sequence; layers
2 onward use a compact contiguous sequence. The scheduler still reserves the
original capacity. This is not a cache-allocation or latency benchmark.

The 100% control calls the original outer and language forwards unchanged.
The 75% experiment copies only the native language control flow, keeping the
original modules, weights, DeepStack order, and semantic mRoPE positions.
"""

import json
import hashlib
import os
from importlib.metadata import version
from pathlib import Path
from types import MethodType
from uuid import uuid4

import torch

from lmdeploy.pytorch.engine.input_process import PreprocessInputResult
from lmdeploy.pytorch.models.qwen3_vl import Qwen3VLForConditionalGeneration

from navigation_dart_kv_layout import (build_single_batch_metadata,
                                      compact_suffix_kv_length)
from navigation_dart_selection import ALGORITHM, select_dart_tokens
from navigation_awq_dart_profiles import AWQ_SPEC, profile_config
from navigation_awq_dart_guard import (validate_config, expected_source_names,
    normalize_source_name, inspect_awq_modules, execution_receipt)


ROOT = Path(__file__).resolve().parents[1]
STATE_KEY = 'project2_awq_dart'
BASE_RUNTIME_SHA256 = 'fd7df464152b7e28bb1b52a76cdc1dae5139deca51df36b0286e8024a7396d72'


def _scalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('DART requires exactly one sequence per batch')
        return int(value.detach().cpu().item())
    return int(value)


def _spans(mask):
    positions = mask.nonzero(as_tuple=False).flatten().tolist()
    spans = []
    for position in positions:
        if not spans or spans[-1][1] != position:
            spans.append([position, position + 1])
        else:
            spans[-1][1] += 1
    return spans


def _grid_token_counts(grid, merge_size):
    """Validate native still-image grids and return their merged token counts."""
    grid = torch.as_tensor(grid).detach().cpu()
    if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] == 0:
        raise ValueError('Expected one [t, h, w] grid row per image')
    if grid.dtype not in (torch.int32, torch.int64) or bool((grid <= 0).any()):
        raise ValueError('Image grid dimensions must be positive integers')
    if bool((grid[:, 0] != 1).any()):
        raise ValueError('Each input must contain one still image; video is unsupported')
    if merge_size <= 0 or bool((grid[:, 1:] % merge_size != 0).any()):
        raise ValueError('Image grid dimensions must align with the native spatial merge size')
    return (grid.prod(-1) // (merge_size ** 2)).tolist()


class _ProfileInputProcessor:
    """Reject unsupported inputs before LMDeploy can split the prefill."""

    def __init__(self, delegate, config, profile):
        self.delegate = delegate
        self.config = config
        self.profile = profile
        self.limits = profile_config(profile)

    def validate_image_count(self, count):
        if not self.limits['min_images'] <= count <= self.limits['max_images']:
            expected = 'exactly two' if self.profile == 'short' else '2..9'
            raise ValueError(f'DART {self.profile} profile requires {expected} images')

    def preprocess_input(self, input_ids, input_multimodals=None, **kwargs):
        maximum = self.limits['max_prompt_tokens']
        if len(input_ids) > maximum:
            raise ValueError(f'DART {self.profile} profile requires <={maximum} input tokens in one prefill')
        video_token = getattr(self.config, 'video_token_id', None)
        if video_token is not None and video_token in input_ids:
            raise ValueError('Video is unsupported by the DART diagnostic')
        if input_multimodals:
            if not isinstance(input_multimodals, (list, tuple)):
                raise ValueError('DART requires a list of image inputs')
            self.validate_image_count(len(input_multimodals))
            previous_end = 0
            for item in input_multimodals:
                if ('video_grid_thw' in item or 'pixel_values_videos' in item
                        or 'image_grid_thw' not in item):
                    raise ValueError('Only still-image multimodal inputs are supported')
                counts = _grid_token_counts(item['image_grid_thw'],
                                            self.config.vision_config.spatial_merge_size)
                if len(counts) != 1 or int(item['image_tokens']) != counts[0]:
                    raise ValueError('Each image token count must match its native grid')
                start, end = int(item['offset']), int(item['offset']) + counts[0]
                if not previous_end <= start < end <= len(input_ids):
                    raise ValueError('Image spans must be ordered, disjoint and inside the input')
                if any(token != self.config.image_token_id for token in input_ids[start:end]):
                    raise ValueError('Image spans must cover their image placeholder tokens')
                previous_end = end
            last_end = max(int(item['offset']) + int(item['image_tokens'])
                           for item in input_multimodals)
            if len(input_ids) - last_end < 4:
                raise ValueError('DART requires at least four suffix text/special tokens')
        elif getattr(self.config, 'image_token_id', None) in input_ids:
            raise ValueError('Image placeholder tokens require native image inputs')
        result = self.delegate.preprocess_input(input_ids, input_multimodals, **kwargs)
        # LMDeploy 0.11.1's inherited processor returns a tuple for text-only input.
        if isinstance(result, tuple):
            result = PreprocessInputResult(input_ids=result[0], input_multimodals=result[1])
        return result


class AwqDartQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Original Qwen3-VL modules with a deliberately restricted runtime adapter."""

    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        if version('lmdeploy') != '0.11.1':
            raise ValueError('This diagnostic adapter is pinned to LMDeploy 0.11.1')
        keep = float(os.environ['PROJECT2_AWQ_DART_KEEP_RATIO'])
        if keep not in (1.0, 0.75):
            raise ValueError('PROJECT2_AWQ_DART_KEEP_RATIO must be 1.0 or 0.75')
        profile = os.environ.get('PROJECT2_AWQ_DART_PROFILE', 'short')
        limits = profile_config(profile)
        run_dir = Path(os.environ['PROJECT2_AWQ_DART_RUN_DIR']).resolve()
        if not run_dir.is_dir() or not run_dir.is_relative_to((ROOT / 'logs').resolve()):
            raise ValueError('PROJECT2_AWQ_DART_RUN_DIR must be an existing directory inside project logs')
        validate_config(config, dtype)
        # Fail closed if the retained C source changes; this independent copy
        # preserves its forward/KV/mRoPE/DeepStack algorithms exactly.
        if hashlib.sha256((ROOT / 'checks/navigation_dart_runtime.py').read_bytes()).hexdigest() != BASE_RUNTIME_SHA256:
            raise ValueError('Pinned DART source changed; review the independent combination adapter')
        if len(config.vision_config.deepstack_visual_indexes) != 3:
            raise ValueError('This adapter requires exactly three native DeepStack injections')
        super().__init__(config, ctx_mgr, dtype=dtype, device=device)
        self._awq_inventory, self._awq_objects = inspect_awq_modules(self)
        self._awq_expected_weights = expected_source_names(config.text_config.num_hidden_layers)
        self._awq_loaded_weights = set()
        self._awq_first_forward_checked = False
        self._awq_calls = None
        self._awq_hooks = []
        for module_name, module in self._awq_objects.items():
            self._awq_hooks.append(module.register_forward_pre_hook(self._awq_call_hook(module_name)))
        self._dart_keep = keep
        self._dart_profile = profile
        self._dart_limits = limits
        self._dart_run_dir = run_dir
        self._dart_active = None
        self._dart_post_key = None
        self._dart_processor = _ProfileInputProcessor(self.input_processor, config, profile)
        self._dart_native_language_forward = self.language_model.forward
        if keep == 0.75:
            if len(self.language_model.layers) < 3:
                raise ValueError('DART requires at least three decoder layers')
            self._dart_hook = self.language_model.layers[1].self_attn.apply_rotary_pos_emb.register_forward_hook(
                self._capture_post_rotary_key)
            owner = self

            def local_forward(language, *args, **kwargs):
                return owner._language_forward(language, *args, **kwargs)

            self.language_model.forward = MethodType(local_forward, self.language_model)
        self._write_json('runtime_init_' + uuid4().hex, {
            'event': 'runtime_initialized', 'runtime_class': type(self).__name__,
            'keep_ratio': keep, **self._precision_evidence(),
            'awq_modules': self._awq_inventory, 'weights_loaded': False,
            'base_runtime_sha256': BASE_RUNTIME_SHA256,
            'profile': profile, 'profile_limits': limits,
            'native_module_weights_preserved': True, 'lmdeploy_version': '0.11.1',
            'scope': 'batch1 eager fresh-request complete-prefill; no chunked prefill',
        })

    @staticmethod
    def _precision_evidence():
        return {'weight_dtype': 'int4_packed+float16', 'activation_dtype': 'float16',
                'awq': True, 'quantization_config': dict(AWQ_SPEC),
                'awq_evidence_scope': 'native packed modules, loader consumption and forward invocation; not numerical equivalence'}

    def _awq_call_hook(self, name):
        def hook(module, args):
            if self._awq_calls is not None:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ValueError('AWQ projection must receive a tensor positional input')
                x = args[0]
                if x.shape[-1] != module.in_features or x.dtype != torch.float16:
                    raise ValueError(f'{name}: AWQ input must be FP16 with native feature width')
                self._awq_calls.setdefault(name, []).append(x.numel() // x.shape[-1])
        return hook

    def load_weights(self, weights):
        # The loader may be invoked more than once (checkpoint shards). Record
        # each tensor only after the native consumer has processed the yield.
        def tracked():
            for name, tensor in weights:
                normalized = normalize_source_name(name)
                is_quant = normalized.endswith(('.qweight', '.qzeros', '.scales'))
                if is_quant and normalized not in self._awq_expected_weights:
                    raise ValueError('Unexpected packed AWQ tensor: ' + name)
                if normalized in self._awq_loaded_weights:
                    raise ValueError('Duplicate packed AWQ tensor: ' + name)
                yield name, tensor
                if is_quant:
                    self._awq_loaded_weights.add(normalized)
        result = super().load_weights(tracked())
        self._awq_first_forward_checked = False
        return result

    def _awq_execution(self, lower_rows, upper_rows):
        return execution_receipt(self._awq_calls, self._awq_inventory,
                                 self._awq_loaded_weights, self._awq_expected_weights,
                                 lower_rows, upper_rows)

    def _write_json(self, stem, data):
        path = self._dart_run_dir / (stem + '.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')

    def get_input_processor(self):
        return self._dart_processor

    def support_cuda_graph(self, *args, **kwargs):
        return False

    def _validate_context(self, context):
        if context is None:
            raise ValueError('DART execution requires an LMDeploy StepContext')
        q_len = _scalar(context.q_seqlens)
        kv_len = _scalar(context.kv_seqlens)
        if q_len <= 0:
            raise ValueError('Empty DART input')
        metadata = context.attn_metadata
        if getattr(metadata, 'quant_policy', 0) != 0:
            raise ValueError('DART does not support KV quantization')
        cache_config = getattr(context, 'cache_config', None)
        if cache_config is not None:
            if cache_config.enable_prefix_caching or cache_config.quant_policy != 0:
                raise ValueError('DART requires prefix caching disabled and KV quantization disabled')
            maximum = self._dart_limits['max_prompt_tokens']
            if cache_config.max_prefill_token_num < maximum:
                raise ValueError(f'DART requires max_prefill_token_num >={maximum} to prevent input splitting')
            if getattr(cache_config, 'window_size', -1) not in (-1, 0):
                raise ValueError('DART does not support sliding-window KV caches')
        if getattr(context, 'kv_quant_policy', 0) != 0:
            raise ValueError('DART does not support KV quantization')
        if metadata.block_offsets.ndim != 2 or metadata.block_offsets.shape[0] != 1:
            raise ValueError('DART supports batch size one only')
        if context.is_decoding:
            if q_len != 1:
                raise ValueError('DART decoding requires exactly one query token')
        elif q_len > self._dart_limits['max_prompt_tokens'] or kv_len != q_len:
            raise ValueError('DART requires a fresh nonchunked prefill within the profile token limit and zero history')
        elif _scalar(context.position_ids.reshape(-1)[0]) != 0:
            raise ValueError('DART prefill must start at original position zero; history is unsupported')
        history = getattr(context, 'history_lengths', None)
        if not context.is_decoding and history is not None and _scalar(history) != 0:
            raise ValueError('DART does not support history continuation or prefix-cache hits')
        return q_len, kv_len

    def update_model_metas(self, past_key_values, inputs_embeds=None, context=None):
        self._validate_context(context)
        if not context.is_decoding and context.model_metas is not None:
            if len(context.model_metas) != 1:
                raise ValueError('DART supports exactly one request metadata entry')
            previous = context.model_metas[0]
            if previous is not None and (STATE_KEY in previous
                                         or _scalar(previous.get('mrope_delta', 0)) != 0):
                raise ValueError('Fresh DART prefill cannot reuse prior request state or mRoPE delta')
        result = super().update_model_metas(past_key_values, inputs_embeds, context)
        # Prefill rebuilds dictionaries in the native implementation. Explicitly
        # replace the context list, which ModelAgent gives precedence over result.
        context.model_metas = result
        if context.is_decoding:
            meta = result[0]
            if STATE_KEY not in meta and _scalar(meta.get('mrope_delta', 0)) != 0:
                raise ValueError('Image decode lost DART request state; refusing unpruned fallback')
        return result

    def _capture_post_rotary_key(self, module, args, output):
        active = self._dart_active
        if active is not None and active['prefill']:
            self._dart_post_key = output[1].detach().clone()

    @staticmethod
    def _block_size(past_key_values):
        sizes = []
        for index in (0, 2):
            cache = past_key_values[index]
            if len(cache) != 2 or any(t.dtype != torch.float16 or t.ndim != 4 for t in cache):
                raise ValueError('DART requires standard, unquantized FP16 K/V caches')
            if cache[0].shape != cache[1].shape:
                raise ValueError('K/V cache shapes differ')
            sizes.append(int(cache[0].shape[1]))
        if sizes[0] != sizes[1]:
            raise ValueError('Decoder layers use different cache block sizes')
        return sizes[0]

    def _navigation_cache_pool(self, context, past_key_values, block_size):
        """Check actual global cache tensors, not the current request's block table.

        This checks allocated token capacity only. It does not guarantee enough
        free GPU memory for visual encoding, activations or full-prefill kernels.
        """
        if self._dart_profile != 'navigation':
            return None
        allocated_counts = []
        expected_tail = past_key_values[0][0].shape[1:]
        for cache in past_key_values:
            if (len(cache) != 2 or any(t.dtype != torch.float16 or t.ndim != 4
                                       or t.shape[1:] != expected_tail for t in cache)
                    or cache[0].shape != cache[1].shape):
                raise ValueError('Navigation DART requires the same standard K/V block layout in every layer')
            allocated_counts.extend(int(t.shape[0]) for t in cache)
        allocated = min(allocated_counts)
        reserved = int(getattr(getattr(context, 'cache_config', None), 'num_reserved_gpu_blocks', 1))
        required = self._dart_limits['max_prompt_tokens'] + self._dart_limits['max_new_tokens']
        usable = (allocated - reserved) * block_size
        if reserved < 0 or reserved >= allocated or usable < required:
            raise ValueError(f'Navigation DART global KV pool has {usable} usable token slots; '
                             f'{required} required for the profile prompt and generation budget '
                             f'({allocated} allocated blocks, {reserved} reserved, block size {block_size})')
        return {'allocated_blocks_per_layer_min': allocated,
                'reserved_gpu_blocks': reserved, 'block_size': block_size,
                'usable_token_capacity': usable, 'required_token_capacity': required,
                'source': 'actual K/V tensor block dimensions across every layer',
                'activation_memory_guaranteed': False}

    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None, **kwargs):
        if not self._awq_first_forward_checked:
            inspect_awq_modules(self)
            if self._awq_loaded_weights != self._awq_expected_weights:
                raise ValueError('Missing packed AWQ weights from native loader; refusing forward')
            self._awq_first_forward_checked = True
        context = self.ctx_mgr.current_context()
        q_len, kv_len = self._validate_context(context)
        if self._dart_active is not None:
            raise ValueError('Concurrent model forwards are unsupported by this diagnostic')
        if context.model_metas is None or len(context.model_metas) != 1:
            raise ValueError('DART requires one initialized model metadata dictionary')
        meta = context.model_metas[0]
        state = meta.get(STATE_KEY)
        image_prefill = kwargs.get('pixel_values') is not None
        if image_prefill:
            if context.is_decoding or kwargs.get('inputs_embeds') is not None:
                raise ValueError('Images require a fresh prefill with native visual encoding')
            grid = kwargs.get('grid_thw')
            if grid is None:
                raise ValueError('DART requires native image grid metadata')
            image_token_counts = _grid_token_counts(grid, self.config.vision_config.spatial_merge_size)
            self._dart_processor.validate_image_count(len(image_token_counts))
            mask = kwargs['image_mask'].reshape(-1).detach().cpu().bool()
            spans = _spans(mask)
            if (len(spans) != len(image_token_counts) or mask.numel() != q_len
                    or q_len - spans[-1][1] < 4):
                raise ValueError('Image spans must match the image count and leave four suffix text/special tokens')
            if [end - start for start, end in spans] != image_token_counts:
                raise ValueError('Actual image span lengths must match native grid token counts')
            state = {'trace_id': uuid4().hex, 'keep_ratio': self._dart_keep, 'profile': self._dart_profile,
                     'original_len': q_len, 'retained_len': q_len, 'removed': 0,
                     'decode_calls': 0, 'first_decode_logged': False}
            meta[STATE_KEY] = state
        elif context.is_decoding and state is None and _scalar(meta.get('mrope_delta', 0)) != 0:
            raise ValueError('Image decode lost DART state')
        elif not context.is_decoding and state is not None:
            raise ValueError('Re-prefilling an existing DART request is unsupported')
        if state is not None and state['keep_ratio'] != self._dart_keep:
            raise ValueError('DART request metadata belongs to another keep-ratio mode')
        if state is not None and state.get('profile') != self._dart_profile:
            raise ValueError('DART request metadata belongs to another input profile')
        block_size = self._block_size(past_key_values) if state is not None else None
        cache_pool = (self._navigation_cache_pool(context, past_key_values, block_size)
                      if state is not None else None)
        active = None
        if state is not None:
            active = {'prefill': image_prefill, 'context': context, 'state': state,
                      'q_length': q_len, 'kv_length': kv_len, 'block_size': block_size}
            if image_prefill:
                active.update(mask=mask, spans=spans, kept=list(range(q_len)))
            else:
                expected_kv = state['original_len'] + state['decode_calls'] + 1
                if kv_len != expected_kv:
                    raise ValueError(f'DART decode KV length {kv_len} != expected {expected_kv}')
            self._dart_active = active
        self._awq_calls = {}
        try:
            output = super().forward(input_ids, position_ids, past_key_values,
                                     attn_metadata=attn_metadata, **kwargs)
            if active is not None:
                if image_prefill:
                    self._write_json('prefill_' + state['trace_id'], {
                        'event': 'image_prefill', 'trace_id': state['trace_id'],
                        'actual_mode': 'native_control' if self._dart_keep == 1.0 else ALGORITHM,
                        'keep_ratio': self._dart_keep, 'prune_before_layer_index': 2,
                        'profile': self._dart_profile, 'profile_limits': self._dart_limits,
                        'complete_prefill': True, 'image_count': len(spans),
                        'image_tokens_per_image': image_token_counts,
                        'retained_image_tokens_per_image': [
                            sum(start <= pos < end for pos in active['kept']) for start, end in spans],
                        'original_length': q_len, 'retained_length': state['retained_len'],
                        'removed': state['removed'], 'image_spans': spans,
                        'original_image_tokens': int(mask.sum()),
                        'retained_image_tokens': int(mask.sum()) - state['removed'],
                        'kept_original_positions': active['kept'],
                        'block_size': block_size, 'scheduler_length': q_len,
                        'lower_layer_kv_length': q_len, 'upper_layer_kv_length': state['retained_len'],
                        'mrope_delta': _scalar(meta['mrope_delta']),
                        'semantic_decode_start': q_len + _scalar(meta['mrope_delta']),
                        **self._precision_evidence(), 'kv_quant_policy': 0,
                        'awq_execution': self._awq_execution(q_len, state['retained_len']),
                        'cache_allocation_reduced': False,
                        'cache_pool': cache_pool,
                        'selection': active.get('selection'),
                    })
                else:
                    if not state['first_decode_logged']:
                        semantic = context.mrope_position_ids.detach().cpu().reshape(3, -1)
                        expected = position_ids.detach().cpu().reshape(1, -1) + _scalar(meta['mrope_delta'])
                        if not torch.equal(semantic, expected.expand_as(semantic)):
                            raise ValueError('DART changed semantic decode mRoPE coordinates')
                        self._write_json('first_decode_' + state['trace_id'], {
                            'event': 'first_decode', 'trace_id': state['trace_id'],
                            **self._precision_evidence(),
                            'awq_execution': self._awq_execution(1, 1),
                            'lower_layer_kv_length': kv_len,
                            'upper_layer_kv_length': kv_len - state['removed'],
                            'query_length': q_len, 'scheduler_positions': position_ids.detach().cpu().tolist(),
                            'mrope_positions': semantic.tolist(),
                            'mrope_delta': _scalar(meta['mrope_delta']),
                            'original_positions_preserved': True,
                        })
                        state['first_decode_logged'] = True
                    state['decode_calls'] += 1
            return output
        finally:
            self._awq_calls = None
            self._dart_active = None
            self._dart_post_key = None

    def _language_forward(self, language, input_ids=None, position_ids=None,
                          past_key_values=None, attn_metadata=None, inputs_embeds=None,
                          mrope_position_ids=None, visual_pos_masks=None,
                          deepstack_visual_embeds=None):
        active = self._dart_active
        if active is None:
            return self._dart_native_language_forward(
                input_ids=input_ids, position_ids=position_ids, past_key_values=past_key_values,
                attn_metadata=attn_metadata, inputs_embeds=inputs_embeds,
                mrope_position_ids=mrope_position_ids, visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds)
        if inputs_embeds is None:
            inputs_embeds = language.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if hidden_states.shape[0] != 1:
            raise ValueError('DART supports one packed sequence only')
        rotary_ids = position_ids if mrope_position_ids is None else mrope_position_ids.unsqueeze(1)
        cos, sin = language.rotary_emb(hidden_states, rotary_ids)
        cos, sin = cos[0], sin[0]
        original_shape = hidden_states.shape
        state = active['state']
        upper_metadata = None
        sequence_keep = None
        if active['prefill']:
            if deepstack_visual_embeds is None or len(deepstack_visual_embeds) != 3:
                raise ValueError('DART requires all three native DeepStack feature arrays')
        else:
            upper_metadata = build_single_batch_metadata(
                1, compact_suffix_kv_length(active['kv_length'], state['removed']),
                attn_metadata.block_offsets, is_decoding=True, block_size=active['block_size'])
        residual = None
        for idx, decoder_layer in enumerate(language.layers):
            if idx == 2 and active['prefill']:
                if self._dart_post_key is None:
                    raise ValueError('Layer 1 post-key-norm/post-mRoPE keys were not captured')
                # LMDeploy RMSNorm returns a tensor without a residual, and a
                # (tensor, residual) pair with one. DeepStack 1 clears residual.
                norm_result = language.norm(hidden_states.clone(),
                                            None if residual is None else residual.clone())
                normed = norm_result[0] if isinstance(norm_result, tuple) else norm_result
                selected = select_dart_tokens(
                    self._dart_post_key.reshape(active['q_length'], -1).cpu().float(),
                    normed.reshape(active['q_length'], -1).detach().cpu().float(),
                    active['mask'], active['spans'], keep_ratio=self._dart_keep)
                keep_cpu = selected['sequence_keep_indices']
                if int(keep_cpu[-1]) != active['q_length'] - 1:
                    raise ValueError('DART must retain the final prompt token for next-token logits')
                sequence_keep = keep_cpu.to(hidden_states.device)
                visual_keep = selected['visual_keep_indices'].to(hidden_states.device)
                hidden_states = hidden_states.index_select(1, sequence_keep)
                if residual is not None:
                    residual = residual.index_select(1, sequence_keep)
                cos, sin = cos.index_select(0, sequence_keep), sin.index_select(0, sequence_keep)
                visual_pos_masks = visual_pos_masks.index_select(1, sequence_keep)
                deepstack_visual_embeds = [features.index_select(0, visual_keep)
                                          for features in deepstack_visual_embeds]
                state['retained_len'] = int(keep_cpu.numel())
                state['removed'] = active['q_length'] - state['retained_len']
                active['kept'] = keep_cpu.tolist()
                active['selection'] = selected['diagnostics']
                upper_metadata = build_single_batch_metadata(
                    state['retained_len'], state['retained_len'], attn_metadata.block_offsets,
                    is_decoding=False, block_size=active['block_size'])
                self._dart_post_key = None
            hidden_states, residual = decoder_layer(
                hidden_states, rotary_pos_emb=(cos, sin), past_key_value=past_key_values[idx],
                residual=residual, attn_metadata=attn_metadata if idx < 2 else upper_metadata)
            if deepstack_visual_embeds is not None and idx < len(deepstack_visual_embeds):
                hidden_states = hidden_states + residual
                hidden_states = language._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[idx])
                residual = None
        norm_result = language.norm(hidden_states, residual)
        hidden_states = norm_result[0] if isinstance(norm_result, tuple) else norm_result
        if sequence_keep is not None:
            # ModelAgent/scheduler retain their original sequence lengths. Only
            # the retained final prompt token is used for next-token generation.
            restored = hidden_states.new_zeros(original_shape)
            restored.index_copy_(1, sequence_keep, hidden_states)
            hidden_states = restored
        return hidden_states
