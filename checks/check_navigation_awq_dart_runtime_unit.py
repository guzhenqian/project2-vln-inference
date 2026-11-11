"""CPU-only AWQ+DART control flow and guard tests; no real AWQ math or GPU.

Reuses the retained C synthetic requests/assertions against the independent D
runtime. Fake W4A16 modules carry native-shaped tensors and execute hooks;
their arithmetic is intentionally absent. This is NOT an AWQ kernel test.
"""

import copy
import importlib
import os
import sys
import types
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from torch import nn

import check_navigation_dart_runtime_unit as base
from navigation_awq_dart_profiles import AWQ_SPEC
from navigation_awq_dart_guard import (SOURCE_PROJECTIONS, execution_receipt,
                                      expected_source_names, inspect_awq_modules)


class AwqLinearW4A16Impl:
    pass


class AwqLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.w_bit, self.group_size = 4, 128
        self.qweight = nn.Parameter(torch.zeros(in_features, out_features // 8, dtype=torch.int32), requires_grad=False)
        self.qzeros = nn.Parameter(torch.zeros(in_features // 128, out_features // 8, dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.ones(in_features // 128, out_features, dtype=torch.float16), requires_grad=False)
        self.impl = AwqLinearW4A16Impl()

    def forward(self, x):
        return x  # Invocation receipt only; deliberately does not emulate AWQ.


class MergedAwqLinear(AwqLinear):
    pass


class QKVAwqLinear(MergedAwqLinear):
    pass


class FakeAwqNativeOuter(base.FakeNativeOuter):
    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        super().__init__(config, ctx_mgr, dtype, device)
        self.constructed_quant = copy.deepcopy(config.text_config.quantization_config)
        for layer in self.language_model.layers:
            layer.self_attn.qkv_proj = QKVAwqLinear(128, 192)
            layer.self_attn.o_proj = AwqLinear(128, 128)
            layer.mlp = nn.Module()
            layer.mlp.gate_up_proj = MergedAwqLinear(128, 512)
            layer.mlp.down_proj = AwqLinear(256, 128)
            original = layer.forward

            def forward(this, hidden, *args, _original=original, **kwargs):
                for projection in ('self_attn.qkv_proj', 'self_attn.o_proj', 'mlp.gate_up_proj', 'mlp.down_proj'):
                    module = this.get_submodule(projection)
                    module(torch.zeros(1, module.in_features, dtype=torch.float16).expand(hidden.shape[1], -1))
                return _original(hidden, *args, **kwargs)

            layer.forward = types.MethodType(forward, layer)

    def load_weights(self, weights):
        self.consumed = getattr(self, 'consumed', [])
        self.consumed.extend(name for name, tensor in weights)


def fake_modules():
    modules = base.fake_lmdeploy_modules()
    modules['lmdeploy.pytorch.models.qwen3_vl'].Qwen3VLForConditionalGeneration = FakeAwqNativeOuter
    for name in ('lmdeploy.pytorch.nn', 'lmdeploy.pytorch.nn.linear', 'lmdeploy.pytorch.nn.linear.awq'):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    module = modules['lmdeploy.pytorch.nn.linear.awq']
    module.AwqLinear, module.MergedAwqLinear, module.QKVAwqLinear = AwqLinear, MergedAwqLinear, QKVAwqLinear
    return modules


class AwqDartRuntimeChecks(base.DartRuntimeChecks):
    @classmethod
    def setUpClass(cls):
        cls.module_patch = mock.patch.dict(sys.modules, fake_modules())
        cls.module_patch.start()
        cls.runtime = importlib.import_module('navigation_awq_dart_runtime')

    @classmethod
    def tearDownClass(cls):
        cls.module_patch.stop()
        sys.modules.pop('navigation_awq_dart_runtime', None)

    def setUp(self):
        self.context = None
        self.manager = types.SimpleNamespace(current_context=lambda: self.context)
        self.config = types.SimpleNamespace(
            quantization_config=dict(AWQ_SPEC),
            text_config=types.SimpleNamespace(quantization_config=dict(AWQ_SPEC), num_hidden_layers=base.LAYERS,
                                             hidden_size=128, intermediate_size=256, head_dim=32,
                                             num_attention_heads=4, num_key_value_heads=1),
            vision_config=types.SimpleNamespace(deepstack_visual_indexes=[5, 11, 17], spatial_merge_size=2),
            image_token_id=999998, video_token_id=999999)
        self.patches = [mock.patch.object(self.runtime, 'version', return_value='0.11.1'),
                        mock.patch.object(self.runtime.AwqDartQwen3VLForConditionalGeneration,
                                          '_write_json', new=self.record_event)]
        for patch in self.patches:
            patch.start()

    def make_model(self, keep, profile='short', load=True):
        with mock.patch.dict(os.environ, {
            'PROJECT2_AWQ_DART_KEEP_RATIO': str(keep),
            'PROJECT2_AWQ_DART_RUN_DIR': str(base.ROOT / 'logs'),
            'PROJECT2_AWQ_DART_PROFILE': profile,
        }):
            model = self.runtime.AwqDartQwen3VLForConditionalGeneration(
                self.config, self.manager, dtype=torch.float16, device='cpu')
        if load:
            model.load_weights((('model.' + name, torch.tensor(0))
                                for name in sorted(model._awq_expected_weights)))
        return model

    def test_native_awq_config_never_removed_and_execution_receipts(self):
        model = self.make_model(0.75, 'navigation')
        self.assertEqual(model.constructed_quant, AWQ_SPEC)
        self.assertEqual(self.config.quantization_config, AWQ_SPEC)
        request = base.make_request(image_lengths=[300] * 9, profile='navigation',
                                    prefix=36, gaps=[9] * 8, suffix=7)
        self.prefill(model, request)
        self.decode(model, request, 1)
        events = [event for _, event in model.test_events]
        self.assertFalse(events[0]['weights_loaded'])
        self.assertEqual(events[0]['awq_modules']['runtime_module_count'], base.LAYERS * 4)
        for event in events:
            self.assertTrue(event['awq'])
            self.assertEqual(event['weight_dtype'], 'int4_packed+float16')
            self.assertEqual(event['activation_dtype'], 'float16')
            self.assertEqual(event['quantization_config'], AWQ_SPEC)
        prefill, decode = events[1:]
        for event in (prefill, decode):
            receipt = event['awq_execution']
            self.assertTrue(receipt['verified'])
            self.assertEqual(receipt['loaded_quantized_tensor_count'], base.LAYERS * 7 * 3)
            self.assertEqual(receipt['module_count'], base.LAYERS * 4)
        self.assertEqual([row['rows'] for row in prefill['awq_execution']['module_calls']],
                         [2815] * 8 + [2140] * 8)
        self.assertEqual([row['rows'] for row in decode['awq_execution']['module_calls']], [1] * 16)

    def test_rejects_fp16_only_mismatched_quant_and_quantized_vision(self):
        original = copy.deepcopy(self.config)
        for field in ('root_absent', 'text_absent', 'bits', 'zero_point', 'version', 'extra', 'vision'):
            self.config = copy.deepcopy(original)
            if field == 'root_absent':
                self.config.quantization_config = None
            elif field == 'text_absent':
                self.config.text_config.quantization_config = None
            elif field == 'vision':
                self.config.vision_config.quantization_config = dict(AWQ_SPEC)
            else:
                self.config.text_config.quantization_config[field] = 'invalid'
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.make_model(0.75)

    def test_rejects_missing_duplicate_unexpected_loaded_weights(self):
        model = self.make_model(0.75, load=False)
        with self.assertRaisesRegex(ValueError, 'Missing packed'):
            self.prefill(model, base.make_request())
        self.assertIsNone(model._dart_active)
        names = sorted(model._awq_expected_weights)
        for shard in (names[:len(names) // 2], names[len(names) // 2:]):
            model.load_weights((name, torch.tensor(0)) for name in shard)
        self.assertEqual(model._awq_loaded_weights, set(names))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            model.load_weights([(names[0], torch.tensor(0))])
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            model.load_weights([('visual.qweight', torch.tensor(0))])
        self.prefill(model, base.make_request())

    def test_rejects_bad_native_module_layout_and_unquantized_fallback(self):
        for mutation in ('class', 'shape', 'dtype', 'group', 'backend', 'extra'):
            model = self.make_model(0.75)
            module = model.language_model.layers[0].self_attn.o_proj
            if mutation == 'class':
                model.language_model.layers[0].self_attn.o_proj = nn.Identity()
            elif mutation == 'shape':
                module.qweight = nn.Parameter(module.qweight[:-1], requires_grad=False)
            elif mutation == 'dtype':
                module.scales = nn.Parameter(module.scales.float(), requires_grad=False)
            elif mutation == 'group':
                module.group_size = 64
            elif mutation == 'backend':
                module.impl = object()
            else:
                model.visual_extra = AwqLinear(128, 128)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                inspect_awq_modules(model)

    def test_execution_receipt_rejects_missing_repeated_or_wrong_rows(self):
        model = self.make_model(0.75)
        inventory = model._awq_inventory
        valid = {entry['name']: [100 if entry['layer_index'] < 2 else 75]
                 for entry in inventory['modules']}
        key = next(iter(valid))
        for mutation in ('missing', 'repeated', 'rows', 'unloaded'):
            calls = copy.deepcopy(valid)
            loaded = set(model._awq_loaded_weights)
            if mutation == 'missing':
                del calls[key]
            elif mutation == 'repeated':
                calls[key] *= 2
            elif mutation == 'rows':
                calls[key] = [75]
            else:
                loaded.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                execution_receipt(calls, inventory, loaded, model._awq_expected_weights, 100, 75)


if __name__ == '__main__':
    torch.set_num_threads(1)
    with mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(AwqDartRuntimeChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f'AWQ_DART_RUNTIME_CPU_UNIT_OK tests={result.testsRun}; fake CPU control flow only, no AWQ numerics')
