"""Independent experimental CPU visual-feature LRU; never imported by primary service.

Only the dedicated entry may install this module. Actual decoded RGB identities
are created inside the trusted native preprocessing path. The cache holds four
CPU feature tensors per image; language inputs, positions and tokens stay native.
All hash/transfer/assembly costs remain online. Numerical comparison runs only
in the two excluded input warmups and fails closed on any byte difference.
"""
from collections import OrderedDict
from contextvars import ContextVar
import functools
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType


GUARDS = {
    'lmdeploy/pytorch/models/qwen3_vl.py': '377b26921dfe42b7d91c2306a908713741fa461a792863b18da0182435127e25',
    'lmdeploy/pytorch/models/qwen2_5_vl.py': 'b5dff117a94bef52686a883b4581befd2c0dfc429c4e4784a2ddd26ef93fef99',
    'lmdeploy/pytorch/multimodal/data_type.py': '5d59a944fed0d9998e519fb3d12a06b9d245149d60228c148a43fd16a185e1ad',
    'lmdeploy/vl/model/qwen3.py': '5b0369a27ae133b661309eee58cec1315b3a562d35cd6e4994b21b8d5ed91173',
    'lmdeploy/vl/model/base.py': '3fdc7a552b83449fa000247ab02ee8b1048f7e700799fdb1da96306e6801d9d1',
    'transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py': '09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c',
    'transformers/models/qwen2_vl/image_processing_qwen2_vl.py': '7820a0fcca107e75605e08d9b774285ca2b0316f857bc0225c779794705ecf4f',
    'transformers/image_processing_utils_fast.py': '038a4aa9e57716be46a18822f5f2f977c8a5b9d0e9cc3db63f032cddafd7d1ac',
}
NUMERIC_HELPER_SHA = 'b5d1f66b9ef9d54e7393866e4d35c444d35b1d35cd4fb5f3c07db14bf3775a75'
IDENTITY_KEY = '_navigation_cpu_vision_identity_v1'
PRIVATE_KWARG = '_navigation_cpu_vision_identities_v1'
_ACTIVE = ContextVar('navigation_cpu_vision_cache', default=None)
_INSTALLED = False
_MODE = None
_MODULE_SHA = None
_PROCESSOR_RECEIPTS = set()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_receipt(directory, filename, value):
    # v2: processor configuration legitimately repeats the guarded source map.
    # Preserve the authoritative envelope, allow identical repeated metadata,
    # and reject conflicting claims before creating any output file.
    envelope = dict(pid=os.getpid(), time_ns=time.time_ns(), patch_module_sha256=_MODULE_SHA,
                    guarded_source_sha256=GUARDS, mode=_MODE)
    for key in envelope.keys() & value.keys():
        require(value[key] == envelope[key], 'Receipt payload conflicts with authoritative metadata: '+key)
    payload = dict(value)
    payload.update(envelope)
    with (Path(directory)/filename).open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def feature_bytes(features):
    return sum(t.numel()*t.element_size() for t in features)


class CpuLRU:
    """Batch lookup protects current hits; final recency follows original slots."""
    def __init__(self, capacity):
        require(capacity in (96, 200), 'CPU vision cache capacity must be96 or200')
        self.capacity = capacity
        self.entries = OrderedDict()
        self.bytes = 0
        self.evictions = 0

    def clear(self):
        before = dict(entries=len(self.entries), bytes=self.bytes)
        self.entries.clear()
        self.bytes = 0
        return before

    def lookup(self, keys):
        hits = {}
        for key in keys:
            if key in self.entries:
                hits[key] = self.entries[key][0]
                self.entries.move_to_end(key)
        return hits

    def put(self, key, features):
        size = feature_bytes(features)
        if key in self.entries:
            self.bytes -= self.entries.pop(key)[1]
        self.entries[key] = (features, size)
        self.bytes += size
        while len(self.entries) > self.capacity:
            _, (_, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1

    def touch_slots(self, keys):
        for key in keys:
            require(key in self.entries, 'A current image was evicted within a small request')
            self.entries.move_to_end(key)

    def discard_for_numeric_gate(self, keys):
        """Only excluded warmup code calls this; not a formal eviction policy."""
        removed = 0
        for key in keys:
            require(key in self.entries, 'Partial-hit gate requires a fully populated cache')
            _, size = self.entries.pop(key)
            self.bytes -= size
            removed += size
        return removed


def flatten_features(output):
    require(isinstance(output, tuple) and len(output) == 2 and len(output[1]) == 3,
            'Native visual must return primary plus exactly three DeepStack tensors')
    return (output[0], *output[1])


def describe_features(output):
    return [dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device)) for t in flatten_features(output)]


class ModelCacheState:
    def __init__(self, model):
        self.model = model
        self.directory = Path(os.environ['PROJECT2_STAGE2_RUN_DIR'])
        self.launch = read(self.directory/'launch.json')
        self.contract = self.launch['vision_cache_contract']
        require(self.contract['mode'] == _MODE and self.contract['storage'] == 'cpu_pageable', 'Cache launch mode/storage mismatch')
        self.lru = CpuLRU(self.contract['capacity'])
        self.plan_document = read(self.directory/'cache_request_plan.json')
        require(self.plan_document['schema'] == 'navigation_vision_cpu_cache_plan/v1', 'Unexpected cache plan schema')
        self.plan_sha = sha((self.directory/'cache_request_plan.json').read_bytes())
        require(self.plan_sha == self.contract['request_plan_sha256'], 'Cache reset plan changed')
        require(self.plan_document['native_plan_sha256'] == sha((self.directory/'request_plan.json').read_bytes()),
                'Cache and native timing plans differ')
        self.plan = self.plan_document['requests']
        require([p['index'] for p in self.plan] == list(range(len(self.plan))), 'Cache plan is not contiguous')
        self.last_index = -1
        self.generation = 0
        self.pending = None
        self.numeric_passed = set()
        self.identity = dict(checkpoint_sha256=self.launch['model_weights_sha256'],
            model_config_sha256=self.launch['model_config_sha256'],
            processor_config_files_sha256=self.contract['processor_config_files_sha256'],
            engine_dtype=self.launch['engine']['dtype'], guarded_source_sha256=GUARDS)
        require(not model.visual.training, 'Visual cache requires evaluation mode')
        require(model.visual.spatial_merge_size == 2 and len(model.visual.deepstack_visual_indexes) == 3,
                'Unexpected native visual merge/DeepStack layout')
        write_receipt(self.directory, 'vision_cache_model_install.json', dict(kind='model_install',
            actual_model_class=type(model).__module__+'.'+type(model).__qualname__, contract=self.contract,
            plan_sha256=self.plan_sha, identity=self.identity, native_language_and_mrope_unchanged=True))

    def begin(self, grids, input_tokens):
        if self.pending is not None:
            write_receipt(self.directory, f'vision_cache_request_{self.pending["request_index"]:04d}.json', self.pending)
            self.pending = None
        index = self.model._fix_index
        require(index == self.last_index+1 and 0 <= index < len(self.plan), 'Unregistered/reordered cache request')
        spec = self.plan[index]
        require(spec['spec'] == self.model._fix_plan[index], 'Cache request differs from native timing plan')
        require(spec['expected_images'] == len(grids) and spec['expected_input_tokens'] == input_tokens,
                'Actual cache request image/token shape differs from reset plan')
        if spec['spec']['phase'] == 'measured' and _MODE == 'cpu':
            require(self.numeric_passed == {0, 1}, 'Both excluded numerical warmups must pass before cache timing')
        if spec['reset_before']:
            before = self.lru.clear()
            self.generation += 1
            write_receipt(self.directory, f'vision_cache_reset_{index:04d}.json', dict(kind='actual_cache_clear',
                request_index=index, episode_id=spec.get('episode_id'), replay_index=spec.get('replay_index'),
                plan_sha256=self.plan_sha, generation=self.generation, before=before,
                after=dict(entries=len(self.lru.entries), bytes=self.lru.bytes)))
        require(spec['reset_before'] == bool(spec.get('episode_start')), 'Every episode start must perform an actual clear')
        self.last_index = index
        return dict(state=self, request_index=index, spec=spec, grids=grids, stats=None,
                    numerical_gate=dict(status='disabled_for_off_control' if _MODE == 'off' else 'not_performed_for_formal_request'))

    def finish(self, context):
        require(context['stats'] is not None, 'Native visual cache wrapper did not execute')
        self.pending = dict(kind='request_execution', request_index=context['request_index'],
            plan_sha256=self.plan_sha, spec=context['spec'], generation=self.generation,
            numerical_gate=context['numerical_gate'], **context['stats'],
            cache_entries=len(self.lru.entries), cache_bytes=self.lru.bytes, cache_capacity=self.lru.capacity,
            completed_host_model_forward=True, scope='Host execution and copy counts; not GPU kernel timing')


def _cached_visual(original, state, grids, identities, pixels, cu, rotary, positions):
    import torch
    require(len(identities) == len(grids), 'Missing ordered trusted RGB identities')
    raw_sizes = [t*h*w for t, h, w in grids]
    merged_sizes = [n//4 for n in raw_sizes]
    require(all(t == 1 and n % 4 == 0 for (t, h, w), n in zip(grids, raw_sizes)), 'Only complete static-image grids are supported')
    require(pixels.shape[0] == sum(raw_sizes) and positions.shape[0] == sum(raw_sizes), 'Native raw patch layout mismatch')
    require(len(rotary) == 2 and all(t.shape[0] == sum(raw_sizes) for t in rotary), 'Native rotary layout mismatch')
    keys = [sha(canonical(dict(input=identity, grid=grid, target_dtype=str(pixels.dtype), model=state.identity)))
            for identity, grid in zip(identities, grids)]
    first = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    require(len(first) <= state.lru.capacity, 'Current request cannot fit in the CPU LRU')
    hits = state.lru.lookup(keys)
    missing = [key for key in first if key not in hits]
    before_evictions = state.lru.evictions
    gpu_values = {}
    h2d = d2h = 0
    for key, features in hits.items():
        require(all(t.device.type == 'cpu' and t.dtype == pixels.dtype for t in features), 'Cached feature dtype/device changed')
        gpu_values[key] = tuple(t.to(device=pixels.device, non_blocking=False) for t in features)
        h2d += feature_bytes(features)
    original_full_output = None
    if missing:
        indices = [first[key] for key in missing]
        if len(missing) == len(keys):
            # No hits or duplicates: preserve exactly the original full batch.
            encoded = original(pixels, cu_seqlens=cu, rotary_pos_emb=rotary, pos_embeds=positions)
            original_full_output = encoded
        else:
            raw_parts = torch.split(pixels, raw_sizes)
            pos_parts = torch.split(positions, raw_sizes)
            rot_parts = [torch.split(value, raw_sizes) for value in rotary]
            lengths = [raw_sizes[i] for i in indices]
            cumulative, total = [], 0
            for length in lengths:
                total += length
                cumulative.append(total)
            # Native visual adds the initial zero itself. Keep its dtype/device.
            selected_cu = cu.new_tensor(cumulative)
            encoded = original(torch.cat([raw_parts[i] for i in indices]), cu_seqlens=selected_cu,
                rotary_pos_emb=tuple(torch.cat([parts[i] for i in indices]) for parts in rot_parts),
                pos_embeds=torch.cat([pos_parts[i] for i in indices]))
        splits = [torch.split(t, [merged_sizes[i] for i in indices]) for t in flatten_features(encoded)]
        for j, key in enumerate(missing):
            features = tuple(parts[j] for parts in splits)
            gpu_values[key] = features
            cpu_features = tuple(t.detach().to(device='cpu', non_blocking=False).contiguous() for t in features)
            require(all(t.device.type == 'cpu' for t in cpu_features), 'Cache insertion did not reach CPU storage')
            d2h += feature_bytes(cpu_features)
            state.lru.put(key, cpu_features)
    state.lru.touch_slots(keys)
    if original_full_output is not None:
        output = original_full_output
    else:
        assembled = [torch.cat([gpu_values[key][j] for key in keys], dim=0) for j in range(4)]
        output = (assembled[0], assembled[1:])
    stats = dict(image_slots=len(keys), distinct_images=len(first), ordered_key_sha256=keys, grids=grids,
        historical_hit_slots=sum(key in hits for key in keys), historical_hit_distinct=len(hits),
        within_request_duplicate_slots=len(keys)-len(first),
        within_request_deduplicated_miss_slots=sum(key not in hits for key in keys)-len(missing),
        encoded_distinct_misses=len(missing), native_vision_forward_calls=int(bool(missing)),
        h2d_feature_bytes=h2d, d2h_feature_bytes=d2h, evictions=state.lru.evictions-before_evictions,
        rgb_identity_bytes=sum(i['rgb_bytes'] for i in identities), output_features=describe_features(output),
        assembly='original_native_full_batch' if original_full_output is not None else 'original_slot_order_full_four_features')
    return output, stats


def _install_visual_wrapper(model, state):
    original = model.visual.forward

    @functools.wraps(original)
    def visual(self, pixels, cu_seqlens, rotary_pos_emb, pos_embeds):
        context = _ACTIVE.get()
        require(context is not None and context['state'] is state, 'Visual cache called outside registered model context')

        def operation():
            if _MODE == 'off' or context['spec']['spec']['phase'] == 'flush':
                output = original(pixels, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, pos_embeds=pos_embeds)
                context['stats'] = dict(image_slots=len(context['grids']), grids=context['grids'],
                    historical_hit_slots=0, historical_hit_distinct=0, within_request_duplicate_slots=None,
                    encoded_distinct_misses=None, native_vision_forward_calls=1, h2d_feature_bytes=0,
                    d2h_feature_bytes=0, rgb_identity_bytes=0, output_features=describe_features(output),
                    assembly='off_or_excluded_flush_native', no_rgb_hash_or_feature_cache=True)
                return output
            index = context['request_index']
            numerical = index in (0, 1) and context['spec']['spec']['phase'] == 'input_warmup'
            if numerical:
                require(context['spec']['spec'] == dict(phase='input_warmup', sample=index, tokens=0, repeat=0),
                        'Numerical comparison requires the exact excluded natural-warmup spec')
                require(context['spec']['expected_images'] == (2 if index == 0 else 9), 'Unexpected numerical warmup image count')
                receipt = read(state.directory/f'input_{context["spec"]["expected_input_tokens"]}.json')
                require(receipt['request_index'] == index, 'Numerical gate lacks corresponding excluded input SHA')
                reference = original(pixels, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, pos_embeds=pos_embeds)
            candidate, stats = _cached_visual(original, state, context['grids'], context['identities'],
                                              pixels, cu_seqlens, rotary_pos_emb, pos_embeds)
            context['stats'] = stats
            if numerical:
                from navigation_latency_vision_qmax_20261120 import _compare_features
                comparison = _compare_features(reference, candidate)
                # Explicitly exercise CPU storage and all-hit H2D assembly too.
                roundtrip, roundtrip_stats = _cached_visual(original, state, context['grids'], context['identities'],
                                                            pixels, cu_seqlens, rotary_pos_emb, pos_embeds)
                roundtrip_comparison = _compare_features(reference, roundtrip)
                passed = comparison['passed'] and roundtrip_comparison['passed']
                partial_gates = []
                unique_keys = list(dict.fromkeys(stats['ordered_key_sha256']))
                eligible_masks = [n for n in (1, 3, 4, 8) if n < len(unique_keys)] if index == 1 else []
                if passed:
                    for missing_count in eligible_masks:
                        # The preceding candidate repopulates all current keys.
                        # Each mask starts fully warm and removes a fixed prefix
                        # of distinct keys, leaving at least one historical hit.
                        evicted_keys = unique_keys[:missing_count]
                        removed = state.lru.discard_for_numeric_gate(evicted_keys)
                        partial, partial_stats = _cached_visual(original, state, context['grids'], context['identities'],
                                                               pixels, cu_seqlens, rotary_pos_emb, pos_embeds)
                        checked = _compare_features(reference, partial)
                        require(partial_stats['encoded_distinct_misses'] == missing_count and
                                partial_stats['historical_hit_distinct'] == len(unique_keys)-missing_count and
                                partial_stats['native_vision_forward_calls'] == 1,
                                'Partial-hit numerical gate did not execute the requested miss batch')
                        partial_gates.append(dict(distinct_miss_count=missing_count, removed_cpu_bytes=removed,
                            evicted_key_sha256=evicted_keys, execution=partial_stats,
                            reference_vs_partial_hit=checked, passed=checked['passed']))
                        del partial
                        if not checked['passed']:
                            passed = False
                            break
                context['numerical_gate'] = dict(status='passed' if passed else 'failed',
                    reference_vs_candidate=comparison, reference_vs_all_hit_cpu_roundtrip=roundtrip_comparison,
                    initial_execution=stats, all_hit_execution=roundtrip_stats,
                    partial_hit_gates=partial_gates, eligible_partial_distinct_miss_counts=eligible_masks,
                    partial_gate_selection='Fixed first distinct RGB keys; masks1/3/4/8 when at least one hit remains; index1 only',
                    extra_full_reference_visual_forwards=1, extra_language_forwards=0,
                    scope='Excluded numerical input warmup only; comparison synchronizes and increases peak memory')
                write_receipt(state.directory, f'vision_cache_numeric_{index:04d}.json', dict(kind='warmup_numeric_gate',
                    request_index=index, plan_sha256=state.plan_sha, **context['numerical_gate']))
                require(passed, 'CPU visual cache bytewise gate failed; stop candidate without relaxing tolerance')
                state.numeric_passed.add(index)
            return candidate

        # Native `vision` timer measures the single missing-image batch, if any.
        # This separate timer includes transfers/assembly and exists on all hits.
        return model._fix_time('vision_cache_total', operation)

    model.visual.forward = MethodType(visual, model.visual)


def install():
    global _INSTALLED, _MODE, _MODULE_SHA
    if _INSTALLED:
        return True
    _MODE = os.environ.get('PROJECT2_LATENCY_VISION_CACHE', 'off')
    require(_MODE in ('off', 'cpu'), 'Expected explicit CPU cache or off control')
    require(os.environ.get('PROJECT2_LATENCY_VISION_CACHE_VALIDATE', '1') == '1', 'This candidate requires strict numerical warmups')
    for flag in ('PROJECT2_LATENCY_VISION_QMAX', 'PROJECT2_LATENCY_W4_TABLE', 'PROJECT2_LATENCY_HYBRID_TABLE', 'PROJECT2_LATENCY_LINEAR_IPC',
                 'PROJECT2_LATENCY_PROFILE', 'PROJECT2_LATENCY_SCHEDULE_TRACE'):
        require(os.environ.get(flag) in (None, '', '0'), 'Do not combine independent cache with other candidates: '+flag)
    from lmdeploy.pytorch.models import qwen3_vl, qwen2_5_vl
    from lmdeploy.vl.model.qwen3 import Qwen3VLModel
    package_root = Path(qwen3_vl.__file__).parents[3]
    for relative, expected in GUARDS.items():
        require(sha((package_root/relative).read_bytes()) == expected, 'CPU cache source guard failed: '+relative)
    require(sha(Path(__file__).with_name('navigation_latency_vision_qmax_20261120.py').read_bytes()) == NUMERIC_HELPER_SHA,
            'Numerical comparison helper changed')
    _MODULE_SHA = sha(Path(__file__).read_bytes())
    model_type = qwen3_vl.Qwen3VLForConditionalGeneration
    originals = ((model_type.forward, 'lmdeploy/pytorch/models/qwen3_vl.py'),
                 (model_type.prepare_inputs_for_generation, 'lmdeploy/pytorch/models/qwen3_vl.py'),
                 (Qwen3VLModel.preprocess, 'lmdeploy/vl/model/qwen3.py'),
                 (qwen2_5_vl.Qwen2_5_VLInputProcessor.preprocess_input, 'lmdeploy/pytorch/models/qwen2_5_vl.py'))
    for function, relative in originals:
        require(Path(function.__code__.co_filename).resolve() == (package_root/relative).resolve(),
                'Unexpected earlier wrapper around native cache hook: '+function.__qualname__)

    if _MODE == 'cpu':
        original_preprocess = Qwen3VLModel.preprocess

        @functools.wraps(original_preprocess)
        def preprocess(self, messages, mm_processor_kwargs=None):
            processor = self.processor.image_processor
            require(type(processor).__name__ == 'Qwen2VLImageProcessorFast', 'Unexpected image processor implementation')
            configuration = dict(configuration=processor.to_dict(), effective_size=self.get_processor_args(mm_processor_kwargs),
                mm_processor_kwargs=mm_processor_kwargs, processor_class=type(processor).__module__+'.'+type(processor).__qualname__,
                guarded_source_sha256=GUARDS)
            processor_sha = sha(canonical(configuration))
            identities = []
            for image, parameters in self.collect_images(messages):
                rgb = image if image.mode == 'RGB' else image.convert('RGB')
                try:
                    raw = rgb.tobytes()
                    identities.append(dict(rgb_sha256=sha(raw), rgb_size=list(rgb.size), rgb_mode='RGB',
                        rgb_bytes=len(raw), processor_sha256=processor_sha))
                    del raw
                finally:
                    if rgb is not image:
                        rgb.close()
            result = original_preprocess(self, messages, mm_processor_kwargs)
            groups = [m['content'] for m in result if m['role'] == 'preprocess']
            require(len(groups) == 1 and len(groups[0]) == len(identities), 'RGB/preprocessor output alignment failed')
            for item, identity in zip(groups[0], identities):
                grid = item['image_grid_thw']
                require(grid.device.type == 'cpu' and tuple(grid.shape) == (1, 3), 'Expected one native CPU grid per image')
                values = grid.tolist()[0]
                require(values[0] == 1 and item['pixel_values'].shape[0] == values[0]*values[1]*values[2],
                        'Actual processed pixel/grid structure differs')
                item[IDENTITY_KEY] = dict(identity, native_grid=values)
            if processor_sha not in _PROCESSOR_RECEIPTS:
                write_receipt(Path(os.environ['PROJECT2_STAGE2_RUN_DIR']), f'vision_cache_processor_{os.getpid()}_{processor_sha}.json',
                    dict(kind='trusted_actual_RGB_processor', processor_sha256=processor_sha, **configuration))
                _PROCESSOR_RECEIPTS.add(processor_sha)
            return result

        Qwen3VLModel.preprocess = preprocess
        original_processor = qwen2_5_vl.Qwen2_5_VLInputProcessor.preprocess_input

        @functools.wraps(original_processor)
        def processor(self, input_ids, input_multimodals=None, **kwargs):
            result = original_processor(self, input_ids, input_multimodals, **kwargs)
            if not input_multimodals:
                return result
            images = result.input_multimodals['image']
            require(len(images) == len(input_multimodals), 'Multimodal identity count changed')
            for source, target in zip(input_multimodals, images):
                require(IDENTITY_KEY in source and target.data.device.type == 'cpu', 'Trusted CPU RGB identity missing')
                target.meta[IDENTITY_KEY] = source[IDENTITY_KEY]
            return result

        qwen2_5_vl.Qwen2_5_VLInputProcessor.preprocess_input = processor
        original_prepare = model_type.prepare_inputs_for_generation

        @functools.wraps(original_prepare)
        def prepare(self, past_key_values, inputs_embeds=None, context=None):
            result = original_prepare(self, past_key_values, inputs_embeds, context)
            if result.get('pixel_values') is not None:
                images = [image for group in context.input_multimodals for image in group.get('image', [])]
                result[PRIVATE_KWARG] = [image.meta[IDENTITY_KEY] for image in images]
            return result

        model_type.prepare_inputs_for_generation = prepare

    original_forward = model_type.forward

    @functools.wraps(original_forward)
    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None, inputs_embeds=None,
                mrope_position_ids=None, pixel_values=None, vis_cu_seqlens=None, vis_pos_emb=None,
                image_mask=None, pos_embeds=None, grid_thw=None, **kwargs):
        identities = kwargs.pop(PRIVATE_KWARG, None)
        native = dict(attn_metadata=attn_metadata, inputs_embeds=inputs_embeds, mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values, vis_cu_seqlens=vis_cu_seqlens, vis_pos_emb=vis_pos_emb, image_mask=image_mask,
            pos_embeds=pos_embeds, grid_thw=grid_thw, **kwargs)
        if pixel_values is None:
            return original_forward(self, input_ids, position_ids, past_key_values, **native)
        require(inputs_embeds is None, 'External image embeddings would bypass native DeepStack; rejected')
        require(grid_thw.device.type == 'cpu' and grid_thw.shape[1] == 3, 'Expected native CPU grid')
        grids = grid_thw.tolist()
        require(2 <= len(grids) <= 9 and all(t == 1 and h > 0 and w > 0 for t, h, w in grids), 'Unsupported static-image request')
        if not hasattr(self, '_navigation_vision_cpu_state'):
            state = ModelCacheState(self)
            _install_visual_wrapper(self, state)
            self._navigation_vision_cpu_state = state
        state = self._navigation_vision_cpu_state
        context = state.begin(grids, input_ids.numel())
        if _MODE == 'cpu':
            require(identities is not None and [i['native_grid'] for i in identities] == grids,
                    'Trusted CPU identities differ from actual native forward grids')
            context['identities'] = identities
        token = _ACTIVE.set(context)
        try:
            output = original_forward(self, input_ids, position_ids, past_key_values, **native)
        finally:
            _ACTIVE.reset(token)
        state.finish(context)
        return output

    model_type.forward = forward
    _INSTALLED = True
    return True
