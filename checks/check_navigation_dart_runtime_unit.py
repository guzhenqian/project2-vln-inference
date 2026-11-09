"""Synthetic CPU integration checks for the restricted DART runtime adapter.

The real runtime and selector run against fake LMDeploy modules and deterministic
small decoder components. No real weights, CUDA kernels, services, or runtime
log files are used. These checks validate control flow, not production numerics.
"""

import copy
import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import math

from navigation_dart_profiles import profile_config

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from torch import nn


WIDTH = 4
LAYERS = 4  # The last layer has no DeepStack injection, like the real 36 layers.
ROOT = Path(__file__).resolve().parents[1]


class FakeNorm(nn.Module):
    """Match native RMSNorm's different return types, including residual=None."""

    def forward(self, hidden, residual=None):
        combined = hidden if residual is None else hidden + residual
        normalized = combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + 1e-5)
        self.last_normalized = normalized.detach().clone()
        return normalized if residual is None else (normalized, combined)


class FakeEmbedding(nn.Module):
    def forward(self, ids):
        scale = torch.tensor([0.013, 0.021, 0.037, 0.051])
        return ids.float().unsqueeze(-1) * scale + torch.arange(WIDTH).float() * 0.17


class FakeRotary(nn.Module):
    def forward(self, hidden, positions):
        if positions.ndim == 3:
            coordinate = positions[:, 0].float()
            phase = coordinate[0] * 0.003 + coordinate[1] * 0.005 + coordinate[2] * 0.007
        else:
            phase = positions.reshape(-1).float() * 0.015
        phase = phase[:, None] + torch.arange(WIDTH).float()[None] * 0.03
        return phase.cos()[None], phase.sin()[None]


class FakeApplyRotary(nn.Module):
    def forward(self, query, key, cos, sin):
        result = (query + cos[:, None] * 0.03, key + sin[:, None] * 0.07)
        self.last_key = result[1].detach().clone()
        return result


class FakeDecoder(nn.Module):
    """Native residual order with a cheap causal prefix-mixing operation."""

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.input_layernorm = FakeNorm()
        self.post_attention_layernorm = FakeNorm()
        self.self_attn = nn.Module()
        self.self_attn.apply_rotary_pos_emb = FakeApplyRotary()
        self.calls = []

    def forward(self, hidden, rotary_pos_emb, past_key_value, residual, attn_metadata):
        q_length = int(attn_metadata.q_seqlens.item())
        kv_length = int(attn_metadata.kv_seqlens.item())
        if hidden.shape[1] != q_length:
            raise AssertionError('Fake decoder received mismatched query metadata')
        cos, sin = rotary_pos_emb
        self.calls.append(dict(q=q_length, kv=kv_length, cos=cos.clone(), sin=sin.clone(),
                               input_hidden=hidden.clone(),
                               input_residual=None if residual is None else residual.clone()))
        if residual is None:
            residual = hidden
            normalized = self.input_layernorm(hidden)
        else:
            normalized, residual = self.input_layernorm(hidden, residual)
        query = normalized[0, :, None] * 0.31
        key = normalized[0, :, None] * (0.23 + self.index * 0.01)
        query, key = self.self_attn.apply_rotary_pos_emb(query, key, cos, sin)
        value = normalized[0] * (0.19 + self.index * 0.02)
        cache_key, cache_value = [tensor.reshape(-1, WIDTH) for tensor in past_key_value]
        start = kv_length - q_length
        cache_key[start:kv_length].copy_(key[:, 0].half())
        cache_value[start:kv_length].copy_(value.half())
        denominator = torch.arange(1, kv_length + 1).float()[:, None]
        prefix_mean = cache_value[:kv_length].float().cumsum(0) / denominator
        attention = query[:, 0] * 0.11 + prefix_mean[start:] * 0.17
        normalized, residual = self.post_attention_layernorm(attention[None], residual)
        return normalized * (0.27 + self.index * 0.03), residual


class FakeNativeText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = FakeEmbedding()
        self.rotary_emb = FakeRotary()
        self.layers = nn.ModuleList(FakeDecoder(i) for i in range(LAYERS))
        self.norm = FakeNorm()
        self.native_calls = 0
        self.deepstack_calls = []

    def _deepstack_process(self, hidden, mask, features):
        addition = torch.zeros_like(hidden).masked_scatter(mask, features)
        output = hidden + addition
        self.deepstack_calls.append(dict(mask=mask.clone(), features=features.clone(), output=output.clone()))
        return output

    def forward(self, input_ids=None, position_ids=None, past_key_values=None,
                attn_metadata=None, inputs_embeds=None, mrope_position_ids=None,
                visual_pos_masks=None, deepstack_visual_embeds=None):
        self.native_calls += 1
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        ids = position_ids if mrope_position_ids is None else mrope_position_ids.unsqueeze(1)
        cos, sin = self.rotary_emb(hidden, ids)
        residual = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, (cos[0], sin[0]), past_key_values[i],
                                     residual, attn_metadata)
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                hidden = self._deepstack_process(hidden + residual, visual_pos_masks,
                                                  deepstack_visual_embeds[i])
                residual = None
        return self.norm(hidden, residual)[0]


class FakeDelegate:
    def preprocess_input(self, input_ids, input_multimodals=None, **kwargs):
        return input_ids, input_multimodals


class FakeNativeOuter(nn.Module):
    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.input_processor = FakeDelegate()
        self.language_model = FakeNativeText()
        self.native_outer_calls = 0

    def update_model_metas(self, past_key_values, inputs_embeds=None, context=None):
        if context.is_decoding:
            result = context.model_metas
            context.mrope_position_ids = (
                context.position_ids + result[0]['mrope_delta']).expand(3, -1).clone()
            return result
        # Match the native behavior that rebuilds the dict on fresh prefill.
        context.mrope_position_ids = context.prefill_mrope.clone()
        return [dict(mrope_delta=context.semantic_delta)]

    @staticmethod
    def image_features(pixel_values):
        return [pixel_values * (0.04 + i * 0.03) + (i + 1) * 0.015 for i in range(3)]

    def forward(self, input_ids, position_ids, past_key_values, attn_metadata=None,
                inputs_embeds=None, mrope_position_ids=None, pixel_values=None,
                image_mask=None, **kwargs):
        self.native_outer_calls += 1
        hidden = self.language_model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        mask, deepstack = None, None
        if pixel_values is not None:
            mask = image_mask.unsqueeze(-1).expand_as(hidden)
            hidden = hidden.masked_scatter(mask, pixel_values)
            deepstack = self.image_features(pixel_values)
        return self.language_model(input_ids=input_ids, position_ids=position_ids,
                                   past_key_values=past_key_values, attn_metadata=attn_metadata,
                                   inputs_embeds=hidden, mrope_position_ids=mrope_position_ids,
                                   visual_pos_masks=mask, deepstack_visual_embeds=deepstack)


def fake_lmdeploy_modules():
    names = ('lmdeploy', 'lmdeploy.pytorch', 'lmdeploy.pytorch.engine',
             'lmdeploy.pytorch.engine.input_process', 'lmdeploy.pytorch.models',
             'lmdeploy.pytorch.models.qwen3_vl', 'lmdeploy.pytorch.backends',
             'lmdeploy.pytorch.backends.cuda', 'lmdeploy.pytorch.backends.cuda.attention')
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules['lmdeploy.pytorch.models.qwen3_vl'].Qwen3VLForConditionalGeneration = FakeNativeOuter
    modules['lmdeploy.pytorch.engine.input_process'].PreprocessInputResult = types.SimpleNamespace
    modules['lmdeploy.pytorch.backends.cuda.attention'].TritonAttentionMetadata = types.SimpleNamespace
    return modules


def metadata(q, kv, *, decode=False):
    return types.SimpleNamespace(q_seqlens=torch.tensor([q]), kv_seqlens=torch.tensor([kv]),
                                 block_offsets=torch.arange(64).reshape(1, 64),
                                 is_decoding=decode, quant_policy=0)


def caches(num_blocks=64):
    return [(torch.zeros(num_blocks, 64, 1, WIDTH, dtype=torch.float16),
             torch.zeros(num_blocks, 64, 1, WIDTH, dtype=torch.float16)) for _ in range(LAYERS)]


def make_request(image_length=8, *, image_lengths=None, profile='short', prefix=2,
                 gaps=None, suffix=6):
    image_lengths = [image_length] * 2 if image_lengths is None else image_lengths
    gaps = [2] * (len(image_lengths) - 1) if gaps is None else gaps
    if len(gaps) != len(image_lengths) - 1:
        raise ValueError('Expected one separator length between adjacent images')
    spans, cursor = [], prefix
    for index, count in enumerate(image_lengths):
        spans.append((cursor, cursor + count))
        cursor += count + (gaps[index] if index < len(gaps) else 0)
    length = spans[-1][1] + suffix
    ids = torch.arange(length).reshape(1, length)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for start, end in spans:
        mask[:, start:end] = True
    pixel = torch.arange(sum(image_lengths) * WIDTH).reshape(-1, WIDTH).float() * 0.011 + 0.1
    positions = torch.arange(length).reshape(1, length)
    context = types.SimpleNamespace(
        q_seqlens=torch.tensor([length]), kv_seqlens=torch.tensor([length]),
        position_ids=positions, history_lengths=torch.tensor([0]), is_decoding=False,
        attn_metadata=metadata(length, length), model_metas=[{'stale_request': True}],
        cache_config=types.SimpleNamespace(enable_prefix_caching=False, quant_policy=0,
                                          max_prefill_token_num=profile_config(profile)['max_prompt_tokens'],
                                          window_size=-1),
        prefill_mrope=torch.stack((positions[0], positions[0] * 2, positions[0] * 3)),
        semantic_delta=-8,
    )
    return dict(context=context, ids=ids, positions=positions, cache=caches(),
                pixel=pixel, mask=mask, spans=spans, length=length,
                grid=torch.tensor([[1, 2, 2 * count] for count in image_lengths]))


class DartRuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module_patch = mock.patch.dict(sys.modules, fake_lmdeploy_modules())
        cls.module_patch.start()
        cls.runtime = importlib.import_module('navigation_dart_runtime')

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()
        sys.modules.pop('navigation_dart_runtime', None)

    def setUp(self):
        self.context = None
        self.manager = types.SimpleNamespace(current_context=lambda: self.context)
        self.config = types.SimpleNamespace(
            text_config=types.SimpleNamespace(),
            vision_config=types.SimpleNamespace(deepstack_visual_indexes=[5, 11, 17], spatial_merge_size=2),
            image_token_id=999998, video_token_id=999999)
        self.patches = [
            mock.patch.object(self.runtime, 'version', return_value='0.11.1'),
            mock.patch.object(self.runtime.DartQwen3VLForConditionalGeneration,
                              '_write_json', new=self.record_event),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()

    @staticmethod
    def record_event(model, stem, data):
        if not hasattr(model, 'test_events'):
            model.test_events = []
        model.test_events.append((stem, copy.deepcopy(data)))

    def make_model(self, keep, profile='short'):
        with mock.patch.dict(os.environ, {
            'PROJECT2_DART_KEEP_RATIO': str(keep),
            'PROJECT2_DART_RUN_DIR': str(ROOT / 'logs'),
            'PROJECT2_DART_PROFILE': profile,
        }):
            return self.runtime.DartQwen3VLForConditionalGeneration(
                self.config, self.manager, dtype=torch.float16, device='cpu')

    def prefill(self, model, request):
        context = request['context']
        self.context = context
        model.update_model_metas(request['cache'], context=context)
        return model(request['ids'], request['positions'], request['cache'],
                     attn_metadata=context.attn_metadata, pixel_values=request['pixel'],
                     image_mask=request['mask'], grid_thw=request['grid'],
                     mrope_position_ids=context.mrope_position_ids)

    def decode(self, model, request, appended):
        original = request['context']
        position = request['length'] + appended - 1
        context = types.SimpleNamespace(
            q_seqlens=torch.tensor([1]), kv_seqlens=torch.tensor([position + 1]),
            position_ids=torch.tensor([[position]]), is_decoding=True,
            attn_metadata=metadata(1, position + 1, decode=True),
            model_metas=original.model_metas, cache_config=original.cache_config)
        self.context = context
        model.update_model_metas(request['cache'], context=context)
        out = model(torch.tensor([[31 + appended]]), context.position_ids, request['cache'],
                    attn_metadata=context.attn_metadata, mrope_position_ids=context.mrope_position_ids)
        return out, context

    def test_keep100_matches_native_math_and_does_not_select(self):
        model = self.make_model(1.0)
        request = make_request()
        with mock.patch.object(self.runtime, 'select_dart_tokens', side_effect=AssertionError('identity selected')):
            actual = self.prefill(model, request)
        native = FakeNativeOuter(self.config, self.manager)
        reference = make_request()
        native.update_model_metas(reference['cache'], context=reference['context'])
        expected = native(reference['ids'], reference['positions'], reference['cache'],
                          attn_metadata=reference['context'].attn_metadata,
                          pixel_values=reference['pixel'], image_mask=reference['mask'],
                          mrope_position_ids=reference['context'].mrope_position_ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(model.language_model.native_calls, 1)
        self.assertEqual(request['context'].model_metas[0][self.runtime.STATE_KEY]['removed'], 0)
        self.assertEqual([layer.calls[0]['q'] for layer in model.language_model.layers], [request['length']] * LAYERS)

    def test_keep75_selects_once_preserves_positions_deepstack_and_scatter(self):
        model = self.make_model(0.75)
        request = make_request()
        original_meta = request['context'].attn_metadata
        with mock.patch.object(self.runtime, 'select_dart_tokens', wraps=self.runtime.select_dart_tokens) as select:
            output = self.prefill(model, request)
        self.assertEqual(select.call_count, 1)
        # The selector must receive the hook's transformed key, not a projection input.
        captured_keys = model.language_model.layers[1].self_attn.apply_rotary_pos_emb.last_key
        torch.testing.assert_close(select.call_args.args[0], captured_keys.reshape(request['length'], -1))
        event = next(data for _, data in model.test_events if data['event'] == 'image_prefill')
        keep = torch.tensor(event['kept_original_positions'])
        removed = torch.ones(request['length'], dtype=torch.bool)
        removed[keep] = False
        self.assertEqual(event['removed'], 4)
        self.assertEqual(event['retained_length'], request['length'] - 4)
        self.assertEqual(int(keep[-1]), request['length'] - 1)
        self.assertEqual(output.shape, (1, request['length'], WIDTH))
        self.assertTrue(torch.equal(output[:, removed], torch.zeros_like(output[:, removed])))
        self.assertGreater(float(output[:, -1].abs().sum()), 0)
        # Every retained row, especially the sampler's final row, must recover
        # the final compact normalized hidden rather than a shifted row.
        torch.testing.assert_close(output.index_select(1, keep),
                                   model.language_model.norm.last_normalized, atol=0, rtol=0)
        low, high = model.language_model.layers[1].calls[0], model.language_model.layers[2].calls[0]
        self.assertEqual([layer.calls[0]['q'] for layer in model.language_model.layers],
                         [request['length']] * 2 + [len(keep)] * 2)
        self.assertEqual([layer.calls[0]['kv'] for layer in model.language_model.layers],
                         [request['length']] * 2 + [len(keep)] * 2)
        torch.testing.assert_close(high['cos'], low['cos'][keep], atol=0, rtol=0)
        torch.testing.assert_close(high['sin'], low['sin'][keep], atol=0, rtol=0)
        self.assertEqual(int(original_meta.q_seqlens.item()), request['length'])
        self.assertEqual(int(original_meta.kv_seqlens.item()), request['length'])
        visual_positions = request['mask'][0].nonzero().flatten()
        visual_keep = torch.isin(visual_positions, keep).nonzero().flatten()
        deepstack = model.language_model.deepstack_calls
        self.assertEqual([len(call['features']) for call in deepstack], [16, 16, 12])
        torch.testing.assert_close(high['input_hidden'], deepstack[1]['output'].index_select(1, keep),
                                   atol=0, rtol=0)
        self.assertIsNone(high['input_residual'])
        expected_scoring_hidden = FakeNorm()(deepstack[1]['output'])[0]
        torch.testing.assert_close(select.call_args.args[1], expected_scoring_hidden, atol=0, rtol=0)
        torch.testing.assert_close(deepstack[2]['features'],
                                   FakeNativeOuter.image_features(request['pixel'])[2][visual_keep], atol=0, rtol=0)
        self.assertEqual(model.language_model.native_calls, 0)
        self.assertIsNone(model._dart_active)
        self.assertIsNone(model._dart_post_key)

    def test_decode_uses_two_cache_lengths_and_original_semantic_positions(self):
        model = self.make_model(0.75)
        request = make_request()
        self.prefill(model, request)
        prefix_snapshots = [[tensor.clone() for tensor in layer] for layer in request['cache']]
        with mock.patch.object(self.runtime, 'select_dart_tokens', side_effect=AssertionError('decode reselected')):
            for appended in (1, 2):
                output, context = self.decode(model, request, appended)
                self.assertEqual(output.shape, (1, 1, WIDTH))
                full = request['length'] + appended
                self.assertEqual([layer.calls[-1]['kv'] for layer in model.language_model.layers],
                                 [full, full, full - 4, full - 4])
                self.assertEqual(int(context.attn_metadata.kv_seqlens.item()), full)
                self.assertTrue(torch.equal(context.mrope_position_ids,
                                            torch.full((3, 1), full - 1 - 8)))
                for layer in model.language_model.layers[1:]:
                    torch.testing.assert_close(layer.calls[-1]['cos'],
                                               model.language_model.layers[0].calls[-1]['cos'], atol=0, rtol=0)
        for index, layer in enumerate(request['cache']):
            valid_prefix = request['length'] if index < 2 else request['length'] - 4
            for before, after in zip(prefix_snapshots[index], layer):
                torch.testing.assert_close(before.reshape(-1, WIDTH)[:valid_prefix],
                                           after.reshape(-1, WIDTH)[:valid_prefix], atol=0, rtol=0)
        self.assertEqual(request['context'].model_metas[0][self.runtime.STATE_KEY]['decode_calls'], 2)

    def test_fresh_request_resets_removed_count_and_trace(self):
        model = self.make_model(0.75)
        first, second = make_request(8), make_request(12)
        self.prefill(model, first)
        state_first = first['context'].model_metas[0][self.runtime.STATE_KEY]
        self.prefill(model, second)
        state_second = second['context'].model_metas[0][self.runtime.STATE_KEY]
        self.assertIsNot(state_first, state_second)
        self.assertNotEqual(state_first['trace_id'], state_second['trace_id'])
        self.assertEqual((state_first['removed'], state_second['removed']), (4, 6))
        self.decode(model, second, 1)
        self.assertEqual(model.language_model.layers[2].calls[-1]['kv'], second['length'] + 1 - 6)
        self.assertEqual(state_first['decode_calls'], 0)
        self.assertNotIn('stale_request', second['context'].model_metas[0])

    def test_reprefill_and_orphan_image_placeholders_are_rejected(self):
        model = self.make_model(0.75)
        request = make_request()
        self.prefill(model, request)
        count = model.native_outer_calls
        with self.assertRaisesRegex(ValueError, 'reuse prior request'):
            model.update_model_metas(request['cache'], context=request['context'])
        self.assertEqual(model.native_outer_calls, count)
        fresh = make_request()
        fresh['context'].model_metas = [{'mrope_delta': -1}]
        with self.assertRaisesRegex(ValueError, 'reuse prior request'):
            model.update_model_metas(fresh['cache'], context=fresh['context'])
        processor = model.get_input_processor()
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            processor.preprocess_input([1, self.config.image_token_id, 2], None)
        # Text-only warmup remains valid and uses the native mathematical path.
        result = processor.preprocess_input([1, 2, 3], None)
        self.assertEqual(result.input_ids, [1, 2, 3])

    def test_rejects_long_history_chunked_and_missing_image_state(self):
        model = self.make_model(0.75)
        processor = model.get_input_processor()
        with self.assertRaisesRegex(ValueError, '1024'):
            processor.preprocess_input([1] * 1025, None)
        with self.assertRaisesRegex(ValueError, 'exactly two'):
            processor.preprocess_input([1] * 50, [{}] * 9)
        for mutation in ('long', 'kv_history', 'history_field', 'position_history', 'split_size'):
            request = make_request()
            context = request['context']
            if mutation == 'long':
                context.q_seqlens = context.kv_seqlens = torch.tensor([1025])
            elif mutation == 'kv_history':
                context.kv_seqlens += 2
            elif mutation == 'history_field':
                context.history_lengths += 1
            elif mutation == 'position_history':
                context.position_ids += 1
            else:
                context.cache_config.max_prefill_token_num = 512
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                model.update_model_metas(request['cache'], context=context)
        request = make_request()
        context = request['context']
        context.is_decoding = True
        context.q_seqlens = torch.tensor([1])
        context.kv_seqlens += 1
        context.position_ids = torch.tensor([[request['length']]])
        context.model_metas = [{'mrope_delta': -8}]
        with self.assertRaisesRegex(ValueError, 'lost DART'):
            model.update_model_metas(request['cache'], context=context)
        self.assertEqual(model.native_outer_calls, 0)

    def test_failed_selection_clears_transient_state(self):
        model = self.make_model(0.75)
        with mock.patch.object(self.runtime, 'select_dart_tokens', side_effect=ValueError('injected selection failure')):
            with self.assertRaisesRegex(ValueError, 'injected selection'):
                self.prefill(model, make_request())
        self.assertIsNone(model._dart_active)
        self.assertIsNone(model._dart_post_key)
        request = make_request()
        output = self.prefill(model, request)
        self.assertEqual(output.shape[1], request['length'])

    def assert_navigation_selection(self, model, request):
        """Check actual selector, complete-span gathers and one decode together."""
        with mock.patch.object(self.runtime, 'select_dart_tokens', wraps=self.runtime.select_dart_tokens) as select:
            output = self.prefill(model, request)
        self.assertEqual(select.call_count, 1)
        self.assertEqual(select.call_args.args[3], [list(span) for span in request['spans']])
        self.assertEqual(select.call_args.args[0].shape[0], request['length'])
        event = [data for _, data in model.test_events if data['event'] == 'image_prefill'][-1]
        keep = torch.tensor(event['kept_original_positions'])
        counts = [end - start for start, end in request['spans']]
        budgets = [math.ceil(count * 0.75) for count in counts]
        removed_count = sum(counts) - sum(budgets)
        retained = request['length'] - removed_count
        self.assertEqual(event['profile'], 'navigation')
        self.assertEqual(event['profile_limits'], profile_config('navigation'))
        self.assertEqual(event['cache_pool']['allocated_blocks_per_layer_min'], 64)
        self.assertEqual(event['cache_pool']['reserved_gpu_blocks'], 1)
        self.assertEqual(event['cache_pool']['usable_token_capacity'], 63 * 64)
        self.assertEqual(event['cache_pool']['required_token_capacity'],
                         sum(profile_config('navigation')[key] for key in ('max_prompt_tokens', 'max_new_tokens')))
        self.assertEqual(event['image_count'], len(counts))
        self.assertEqual(event['image_tokens_per_image'], counts)
        self.assertEqual(event['retained_image_tokens_per_image'], budgets)
        self.assertEqual((event['retained_length'], event['removed']), (retained, removed_count))
        self.assertTrue(event['complete_prefill'])
        self.assertEqual(keep.tolist(), sorted(set(keep.tolist())))
        text_positions = (~request['mask'][0]).nonzero().flatten()
        self.assertTrue(bool(torch.isin(text_positions, keep).all()))
        self.assertEqual(event['selection']['text_pool']['start'], request['spans'][-1][1])
        shared_pivots = event['selection']['text_pivot_indices']
        self.assertEqual(len(shared_pivots), 4)
        for image in event['selection']['images']:
            self.assertEqual(image['pivot_order'][-4:], shared_pivots)
        self.assertTrue(all(request['spans'][-1][1] <= index < request['length']
                            for index in shared_pivots))
        low, high = model.language_model.layers[1].calls[-1], model.language_model.layers[2].calls[-1]
        self.assertEqual([layer.calls[-1]['q'] for layer in model.language_model.layers],
                         [request['length']] * 2 + [retained] * 2)
        self.assertEqual([layer.calls[-1]['kv'] for layer in model.language_model.layers],
                         [request['length']] * 2 + [retained] * 2)
        torch.testing.assert_close(high['cos'], low['cos'][keep], atol=0, rtol=0)
        torch.testing.assert_close(high['sin'], low['sin'][keep], atol=0, rtol=0)
        deepstack = model.language_model.deepstack_calls[-3:]
        expected_features = FakeNativeOuter.image_features(request['pixel'])
        for branch in (0, 1):
            torch.testing.assert_close(deepstack[branch]['features'], expected_features[branch], atol=0, rtol=0)
        visual_positions = request['mask'][0].nonzero().flatten()
        visual_keep = torch.isin(visual_positions, keep).nonzero().flatten()
        torch.testing.assert_close(deepstack[2]['features'], expected_features[2][visual_keep], atol=0, rtol=0)
        torch.testing.assert_close(deepstack[2]['mask'], deepstack[1]['mask'].index_select(1, keep))
        torch.testing.assert_close(high['input_hidden'], deepstack[1]['output'].index_select(1, keep), atol=0, rtol=0)
        torch.testing.assert_close(output.index_select(1, keep), model.language_model.norm.last_normalized,
                                   atol=0, rtol=0)
        removed = torch.ones(request['length'], dtype=torch.bool)
        removed[keep] = False
        self.assertTrue(torch.equal(output[:, removed], torch.zeros_like(output[:, removed])))
        prefix_caches = [[tensor.clone() for tensor in layer] for layer in request['cache']]
        with mock.patch.object(self.runtime, 'select_dart_tokens', side_effect=AssertionError('decode reselected')):
            _, context = self.decode(model, request, 1)
        self.assertEqual([layer.calls[-1]['kv'] for layer in model.language_model.layers],
                         [request['length'] + 1] * 2 + [retained + 1] * 2)
        self.assertTrue(torch.equal(context.mrope_position_ids,
                                    torch.full((3, 1), request['length'] + request['context'].semantic_delta)))
        for index, layer in enumerate(model.language_model.layers):
            valid = request['length'] if index < 2 else retained
            torch.testing.assert_close(layer.calls[-1]['cos'], model.language_model.layers[0].calls[-1]['cos'],
                                       atol=0, rtol=0)
            for old, new in zip(prefix_caches[index], request['cache'][index]):
                torch.testing.assert_close(old.reshape(-1, WIDTH)[:valid], new.reshape(-1, WIDTH)[:valid],
                                           atol=0, rtol=0)
                self.assertTrue(torch.equal(new.reshape(-1, WIDTH)[valid + 1:],
                                            torch.zeros_like(new.reshape(-1, WIDTH)[valid + 1:])))
        return event

    def test_navigation_two_through_nine_images_variable_spans_and_request_isolation(self):
        model = self.make_model(0.75, profile='navigation')
        trace_ids, states = [], []
        for count in range(2, 10):
            lengths = [8 + 3 * index for index in range(count)]
            request = make_request(image_lengths=lengths, profile='navigation', prefix=count,
                                   gaps=list(range(1, count)), suffix=count + 4)
            with self.subTest(image_count=count):
                event = self.assert_navigation_selection(model, request)
                trace_ids.append(event['trace_id'])
                state = request['context'].model_metas[0][self.runtime.STATE_KEY]
                states.append(state)
                self.assertEqual(state['original_len'], request['length'])
                self.assertEqual(state['removed'], sum(lengths) - sum(math.ceil(x * 0.75) for x in lengths))
        self.assertEqual(len(set(trace_ids)), 8)
        self.assertEqual(len({id(state) for state in states}), 8)
        self.assertTrue(all(state['decode_calls'] == 1 for state in states))

    def test_navigation_nine_300_token_images_2815_to_2140_and_native_control(self):
        request_args = dict(image_lengths=[300] * 9, profile='navigation',
                            prefix=36, gaps=[9] * 8, suffix=7)
        model = self.make_model(0.75, profile='navigation')
        request = make_request(**request_args)
        self.assertEqual(request['length'], 2815)
        event = self.assert_navigation_selection(model, request)
        self.assertEqual(event['retained_length'], 2140)
        self.assertEqual(event['retained_image_tokens_per_image'], [225] * 9)
        control = self.make_model(1.0, profile='navigation')
        control_request = make_request(**request_args)
        with mock.patch.object(self.runtime, 'select_dart_tokens', side_effect=AssertionError('native control selected')):
            actual = self.prefill(control, control_request)
        native = FakeNativeOuter(self.config, self.manager)
        reference = make_request(**request_args)
        native.update_model_metas(reference['cache'], context=reference['context'])
        expected = native(reference['ids'], reference['positions'], reference['cache'],
                          attn_metadata=reference['context'].attn_metadata,
                          pixel_values=reference['pixel'], image_mask=reference['mask'],
                          mrope_position_ids=reference['context'].mrope_position_ids)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(control.language_model.native_calls, 1)
        self.assertEqual([layer.calls[-1]['kv'] for layer in control.language_model.layers], [2815] * LAYERS)
        control_event = [data for _, data in control.test_events if data['event'] == 'image_prefill'][-1]
        self.assertEqual(control_event['retained_image_tokens_per_image'], [300] * 9)

    def processor_payload(self, request):
        ids = request['ids'].reshape(-1).tolist()
        images = []
        for index, (start, end) in enumerate(request['spans']):
            ids[start:end] = [self.config.image_token_id] * (end - start)
            images.append({'offset': start, 'image_tokens': end - start,
                           'image_grid_thw': request['grid'][index:index + 1].clone()})
        return ids, images

    def test_navigation_processor_accepts_valid_profiles_and_rejects_grid_span_mismatch(self):
        model = self.make_model(0.75, profile='navigation')
        processor = model.get_input_processor()
        for count in range(2, 10):
            request = make_request(image_lengths=[8 + i for i in range(count)], profile='navigation')
            ids, images = self.processor_payload(request)
            with self.subTest(count=count):
                self.assertEqual(processor.preprocess_input(ids, images).input_ids, ids)
        maximum = profile_config('navigation')['max_prompt_tokens']
        self.assertEqual(maximum, 3072)
        with self.assertRaisesRegex(ValueError, str(maximum)):
            processor.preprocess_input([1] * (maximum + 1), None)
        self.assertEqual(len(processor.preprocess_input([1] * maximum, None).input_ids), maximum)
        for count in (1, 10):
            request = make_request(image_lengths=[8] * count, profile='navigation')
            ids, images = self.processor_payload(request)
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, '2..9'):
                processor.preprocess_input(ids, images)
            with self.subTest(runtime_count=count), self.assertRaisesRegex(ValueError, '2..9'):
                self.prefill(model, request)
        for mutation in ('grid_count', 'grid_height', 'grid_time', 'offset', 'placeholder'):
            request = make_request(profile='navigation')
            ids, images = self.processor_payload(request)
            if mutation == 'grid_count':
                images[0]['image_tokens'] += 1
            elif mutation == 'grid_height':
                images[0]['image_grid_thw'][0, 1] += 1
            elif mutation == 'grid_time':
                images[0]['image_grid_thw'][0, 0] = 2
            elif mutation == 'offset':
                images[1]['offset'] = images[0]['offset']
            else:
                ids[request['spans'][0][0]] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                processor.preprocess_input(ids, images)
        for mutation in ('grid_count', 'span_count', 'span_length'):
            request = make_request(image_lengths=[8] * 9, profile='navigation')
            if mutation == 'grid_count':
                request['grid'] = request['grid'][:-1]
            elif mutation == 'span_count':
                request['mask'][:, request['spans'][0][1]:request['spans'][1][0]] = True
            else:
                request['grid'][0, 2] += 2
            with self.subTest(runtime_mutation=mutation), self.assertRaises(ValueError):
                self.prefill(model, request)
        self.assertEqual(model.native_outer_calls, 0)

    def test_navigation_rejects_oversize_history_chunks_and_profile_state_mismatch(self):
        model = self.make_model(0.75, profile='navigation')
        for mutation in ('long', 'history', 'position', 'split_size', 'partial_prefill'):
            request = make_request(image_lengths=[300] * 9, profile='navigation')
            context = request['context']
            if mutation == 'long':
                context.q_seqlens = context.kv_seqlens = torch.tensor([
                    profile_config('navigation')['max_prompt_tokens'] + 1])
            elif mutation == 'history':
                context.kv_seqlens += 1
            elif mutation == 'position':
                context.position_ids += 1
            elif mutation == 'split_size':
                context.cache_config.max_prefill_token_num = 1024
            else:
                context.q_seqlens = context.kv_seqlens = torch.tensor([512])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.prefill(model, request)
        self.assertEqual(model.native_outer_calls, 0)
        request = make_request(profile='navigation')
        self.prefill(model, request)
        state = request['context'].model_metas[0][self.runtime.STATE_KEY]
        state['profile'] = 'short'
        with self.assertRaisesRegex(ValueError, 'another input profile'):
            self.decode(model, request, 1)
        with self.assertRaisesRegex(ValueError, 'reuse prior request'):
            self.prefill(model, request)

    def test_navigation_global_pool_capacity_uses_tensor_blocks_and_reserved_budget(self):
        model = self.make_model(0.75, profile='navigation')
        limits = profile_config('navigation')
        required = limits['max_prompt_tokens'] + limits['max_new_tokens']
        self.assertEqual(required, 3584)
        required_blocks = math.ceil(required / 64)
        for mutation in ('whole_pool_small', 'one_high_layer_small', 'reserved_blocks'):
            request = make_request(profile='navigation')
            if mutation == 'whole_pool_small':
                request['cache'] = caches(required_blocks)
            elif mutation == 'one_high_layer_small':
                request['cache'][-1] = caches(required_blocks)[0]
            else:
                request['context'].cache_config.num_reserved_gpu_blocks = 64 - required_blocks + 1
            # A wide per-request block table cannot compensate for a small pool.
            self.assertEqual(request['context'].attn_metadata.block_offsets.shape[1], 64)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'global KV pool'):
                self.prefill(model, request)
        for mutation in ('high_layer_dtype', 'kv_shape', 'high_layer_tail_shape'):
            request = make_request(profile='navigation')
            key, value = request['cache'][-1]
            if mutation == 'high_layer_dtype':
                key, value = key.float(), value.float()
            elif mutation == 'kv_shape':
                value = value[:-1]
            else:
                key, value = key[..., :WIDTH - 1], value[..., :WIDTH - 1]
            request['cache'][-1] = key, value
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'every layer'):
                self.prefill(model, request)
        self.assertEqual(model.native_outer_calls, 0)
        # Exact minimum global pool (one reserved block) is sufficient even
        # when this short request needs only one allocated logical block.
        request = make_request(profile='navigation')
        request['cache'] = caches(required_blocks + 1)
        request['context'].attn_metadata.block_offsets = torch.tensor([[7]])
        self.prefill(model, request)
        event = [data for _, data in model.test_events if data['event'] == 'image_prefill'][-1]
        self.assertEqual(event['cache_pool']['usable_token_capacity'], required)
        self.assertEqual(event['cache_pool']['allocated_blocks_per_layer_min'], required_blocks + 1)
        self.assertFalse(event['cache_pool']['activation_memory_guaranteed'])
        # The short profile's native control remains valid with its existing
        # 63-block allocation; it does not inherit the navigation-only gate.
        short = self.make_model(1.0)
        request = make_request()
        request['cache'] = caches(63)
        self.prefill(short, request)
        self.assertEqual(short.language_model.native_calls, 1)
        event = [data for _, data in short.test_events if data['event'] == 'image_prefill'][-1]
        self.assertIsNone(event['cache_pool'])


if __name__ == '__main__':
    with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA is forbidden in CPU unit checks')):
        outcome = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(DartRuntimeChecks))
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    print(f'DART_RUNTIME_CPU_UNIT_OK tests={outcome.testsRun} (fake CPU control flow only; no CUDA/service validation)')
