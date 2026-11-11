"""Standard-library SYNTHETIC corruption checks for the AWQ+DART audit.

Copies only saved JSON/JSONL evidence into isolated tmp directories. No original
logs are modified, and no HTTP, simulator, GPU or model imports are used.
Saved C traces supply realistic token/KV geometry only. AWQ evidence is fabricated
in isolated temporary fixtures to exercise acceptance/rejection; these fixtures
do NOT prove AWQ kernel execution or an actual D-group result.
"""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from audit_navigation_awq_dart_run import (AWQ_SPEC, CONFIG_SHA, HEADER_SHA,
                                           MODEL_RELATIVE, WEIGHT_SHA, audit_batch,
                                           audit_trace, expected_modules, served_model_name)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'logs/navigation_dart_single_learning_20261108T091551Z_4612'
SERVICE = ROOT / 'logs/navigation_dart_service_learning_20261108T065350824909Z_284bc210'
MODEL = 'SimpleVLN-Qwen3VL-4B-RL-AWQ-DART-keep75-navigation-20261108'


def synthetic_quantization():
    return dict(weight_dtype='int4_packed+float16', activation_dtype='float16',
                awq=True, quantization_config=copy.deepcopy(AWQ_SPEC))


def synthetic_execution(original_rows, retained_rows):
    return dict(verified=True, loaded_quantized_tensor_count=756, module_count=144,
                module_calls=[dict(name=name, layer_index=layer, projection=projection,
                                   rows=original_rows if layer < 2 else retained_rows, calls=1)
                              for name, (layer, projection, *_) in expected_modules().items()])


def synthetic_init():
    modules = []
    for name, (layer, projection, class_name, in_features, out_features) in expected_modules().items():
        modules.append(dict(name=name, layer_index=layer, projection=projection, **{'class': class_name},
                            backend_class='AwqLinearW4A16Impl',
                            in_features=in_features, out_features=out_features,
                            qweight_shape=[in_features, out_features // 8],
                            qzeros_shape=[in_features // 128, out_features // 8],
                            scales_shape=[in_features // 128, out_features],
                            qweight_dtype='int32', qzeros_dtype='int32', scales_dtype='float16'))
    return dict(event='runtime_initialized', runtime_class='AwqDartQwen3VLForConditionalGeneration',
                profile='navigation', keep_ratio=.75, **synthetic_quantization(),
                awq_modules=dict(runtime_module_count=144, checkpoint_projection_count=252, modules=modules),
                synthetic_fixture=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_json(path, value):
    info = path.stat()
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))


class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provenance = read(RUN / 'provenance.json')
        start = datetime.fromisoformat(provenance['started_utc'].replace('Z', '+00:00')).timestamp()
        end = (RUN / 'summary.json').stat().st_mtime
        all_prefills = [p for p in SERVICE.glob('prefill_*.json') if p.stat().st_mtime <= end]
        cls.before = {p.name for p in all_prefills if p.stat().st_mtime < start}
        cls.current = sorted([p for p in all_prefills if p.name not in cls.before],
                             key=lambda p: (p.stat().st_mtime_ns, p.name))
        if len(cls.before) != 1 or len(cls.current) != 9:
            raise ValueError('Expected one previous request and nine traces from the saved single episode')
        run_files = [RUN / name for name in ('manifest.json', 'summary.json', 'provenance.json')]
        run_files += list(RUN.glob('episode_*/steps.jsonl'))
        service_files = [SERVICE / 'launch.json'] + all_prefills
        service_files += [p.with_name(p.name.replace('prefill_', 'first_decode_')) for p in all_prefills]
        cls.run_files, cls.service_files = run_files, service_files
        cls.original_hashes = {path: digest(path) for path in run_files + service_files}
        (ROOT / 'tmp').mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        changed = [str(path) for path, expected in cls.original_hashes.items() if digest(path) != expected]
        if changed:
            raise AssertionError('Original evidence changed: ' + ', '.join(changed))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='awq_dart_audit_unit_', dir=ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        self.service = Path(self.temp.name) / 'service'
        for source_root, destination, paths in ((RUN, self.run, self.run_files),
                                                 (SERVICE, self.service, self.service_files)):
            for path in paths:
                target = destination / path.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        # Only copied fixtures are adapted; genuine C evidence stays unchanged.
        for path in self.service.glob('prefill_*.json'):
            record = read(path)
            record.update(synthetic_quantization(), synthetic_fixture=True,
                          awq_execution=synthetic_execution(record['original_length'], record['retained_length']))
            replace_json(path, record)
            decode_path = path.with_name(path.name.replace('prefill_', 'first_decode_'))
            decode = read(decode_path)
            decode.update(synthetic_quantization(), synthetic_fixture=True,
                          awq_execution=synthetic_execution(1, 1))
            replace_json(decode_path, decode)
        self.init = self.service / 'runtime_init_synthetic.json'
        self.init.write_text(json.dumps(synthetic_init()), encoding='utf-8')
        for path, key in ((self.run / 'manifest.json', 'model'),
                          (self.run / 'summary.json', 'model'),
                          (self.service / 'launch.json', 'model_name')):
            record = read(path)
            record[key] = MODEL
            if key == 'model_name':
                record['weight_quantization'] = copy.deepcopy(AWQ_SPEC)
                record.update(model_source=MODEL_RELATIVE, model_config_sha256=CONFIG_SHA,
                              weight_header_sha256=HEADER_SHA,
                              checkpoint_verification=dict(historical_full_weight_sha256=WEIGHT_SHA,
                                  full_weight_sha256_rechecked_this_run=True))
            record['synthetic_fixture'] = True
            replace_json(path, record)
        self.prefill = self.service / self.current[-1].name
        self.decode = self.prefill.with_name(self.prefill.name.replace('prefill_', 'first_decode_'))

    def audit(self):
        return audit_batch(self.run, self.service, before_names=self.before)

    def change_prefill(self, key, value):
        record = read(self.prefill)
        record[key] = value
        replace_json(self.prefill, record)

    def test_synthetic_awq_fixture_snapshot_and_completed_mtime_window(self):
        snapshot = self.audit()
        completed = audit_batch(self.run, self.service)
        self.assertTrue(snapshot['passed'])
        self.assertEqual(snapshot['actual_model_calls'], 7)
        self.assertEqual(snapshot['runtime_trace_count'], 9)
        self.assertEqual(snapshot['warmup_requests'], 2)
        self.assertEqual(snapshot['traces'], completed['traces'])
        self.assertEqual(snapshot['pruning_counts'], completed['pruning_counts'])

    def test_snapshot_does_not_depend_on_completed_summary_mtime(self):
        path = self.run / 'summary.json'
        os.utime(path, (1, 1))
        self.assertTrue(self.audit()['passed'])
        with self.assertRaisesRegex(ValueError, 'Trace count'):
            audit_batch(self.run, self.service)

    def test_missing_new_trace_is_rejected(self):
        self.prefill.unlink()
        with self.assertRaisesRegex(ValueError, 'Trace count'):
            self.audit()

    def test_missing_old_snapshot_trace_is_rejected(self):
        (self.service / next(iter(self.before))).unlink()
        with self.assertRaisesRegex(ValueError, 'disappeared'):
            self.audit()

    def test_missing_decode_is_rejected(self):
        self.decode.unlink()
        with self.assertRaises(FileNotFoundError):
            self.audit()

    def test_extra_new_trace_is_rejected(self):
        shutil.copy2(self.prefill, self.service / 'prefill_extra.json')
        with self.assertRaisesRegex(ValueError, 'Trace count'):
            self.audit()

    def test_fp16_and_kv_quantization_are_rejected(self):
        original = read(self.prefill)
        for key, value in (('awq', False), ('kv_quant_policy', 4), ('weight_dtype', 'float16'),
                           ('activation_dtype', 'bfloat16')):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[key] = value
                replace_json(self.prefill, changed)
                with self.assertRaisesRegex(ValueError, 'Expected'):
                    audit_trace(self.prefill)

    def test_unquantized_service_identity_is_rejected(self):
        path = self.service / 'launch.json'
        launch = read(path)
        launch['weight_quantization'] = None
        replace_json(path, launch)
        with self.assertRaisesRegex(ValueError, 'profile/quantization'):
            self.audit()

    def test_wrong_budget_even_with_matching_counts_is_rejected(self):
        p, d = read(self.prefill), read(self.decode)
        start, end = p['image_spans'][0]
        removed = next(i for i in p['kept_original_positions'] if start <= i < end)
        p['kept_original_positions'].remove(removed)
        p['retained_length'] -= 1
        p['retained_image_tokens_per_image'][0] -= 1
        p['retained_image_tokens'] -= 1
        p['removed'] += 1
        p['upper_layer_kv_length'] -= 1
        p['selection']['sequence_length_after'] -= 1
        p['selection']['visual_tokens_after'] -= 1
        d['upper_layer_kv_length'] -= 1
        replace_json(self.prefill, p)
        replace_json(self.decode, d)
        with self.assertRaisesRegex(ValueError, 'Keep-ratio budget'):
            audit_trace(self.prefill)

    def test_dropped_text_is_rejected(self):
        p = read(self.prefill)
        p['kept_original_positions'].remove(0)
        p['retained_length'] -= 1
        replace_json(self.prefill, p)
        with self.assertRaisesRegex(ValueError, 'Text/special tokens removed'):
            audit_trace(self.prefill)

    def test_wrong_decode_kv_is_rejected(self):
        d = read(self.decode)
        d['upper_layer_kv_length'] += 1
        replace_json(self.decode, d)
        with self.assertRaisesRegex(ValueError, 'Decode KV lengths'):
            audit_trace(self.prefill)

    def test_wrong_decode_scheduler_position_is_rejected(self):
        d = read(self.decode)
        d['scheduler_positions'][0][0] += 1
        replace_json(self.decode, d)
        with self.assertRaisesRegex(ValueError, 'scheduler position'):
            audit_trace(self.prefill)

    def test_wrong_decode_mrope_is_rejected(self):
        d = read(self.decode)
        d['mrope_positions'][0][0] += 1
        replace_json(self.decode, d)
        with self.assertRaisesRegex(ValueError, 'Decode mRoPE'):
            audit_trace(self.prefill)

    def test_consistently_shifted_mrope_is_rejected(self):
        p, d = read(self.prefill), read(self.decode)
        p['mrope_delta'] += 1
        p['semantic_decode_start'] += 1
        d['mrope_delta'] += 1
        d['mrope_positions'] = [[v[0] + 1] for v in d['mrope_positions']]
        replace_json(self.prefill, p)
        replace_json(self.decode, d)
        with self.assertRaisesRegex(ValueError, 'frozen image grid'):
            audit_trace(self.prefill)

    def test_insufficient_kv_capacity_is_rejected(self):
        p = read(self.prefill)
        pool = p['cache_pool']
        pool['allocated_blocks_per_layer_min'] = 40
        pool['usable_token_capacity'] = (40 - pool['reserved_gpu_blocks']) * pool['block_size']
        replace_json(self.prefill, p)
        with self.assertRaisesRegex(ValueError, 'fixed 64x64'):
            audit_trace(self.prefill)

    def test_fp16_original_trace_cannot_pass_d_group_audit(self):
        original = SERVICE / self.current[-1].name
        with self.assertRaisesRegex(ValueError, 'Expected native AWQ'):
            audit_trace(original)

    def test_awq_label_without_execution_evidence_is_rejected(self):
        record = read(self.prefill)
        del record['awq_execution']
        replace_json(self.prefill, record)
        with self.assertRaisesRegex(ValueError, 'not completely loaded'):
            audit_trace(self.prefill)

    def test_missing_quantized_weights_and_module_coverage_are_rejected(self):
        original = read(self.prefill)
        for kind in ('not_verified', 'weights', 'module_count', 'missing', 'duplicate'):
            record = copy.deepcopy(original)
            evidence = record['awq_execution']
            if kind == 'not_verified':
                evidence['verified'] = False
            elif kind == 'weights':
                evidence['loaded_quantized_tensor_count'] = 755
            elif kind == 'module_count':
                evidence['module_count'] = 143
            elif kind == 'missing':
                evidence['module_calls'].pop()
            else:
                evidence['module_calls'][-1] = copy.deepcopy(evidence['module_calls'][0])
            replace_json(self.prefill, record)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit_trace(self.prefill)

    def test_runtime_init_fused_module_shape_dtype_class_are_checked(self):
        original = read(self.init)
        for key, value in (('qweight_shape', [2560, 6144]), ('qzeros_dtype', 'float16'),
                           ('scales_dtype', 'float32'), ('class', 'Linear'),
                           ('backend_class', 'DenseLinearImpl')):
            record = copy.deepcopy(original)
            record['awq_modules']['modules'][0][key] = value
            replace_json(self.init, record)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mismatch'):
                audit_trace(self.prefill)

    def test_runtime_init_and_execution_quantization_must_match(self):
        original = read(self.init)
        record = copy.deepcopy(original)
        record['quantization_config']['group_size'] = 64
        replace_json(self.init, record)
        with self.assertRaisesRegex(ValueError, 'Expected native AWQ'):
            audit_trace(self.prefill)

    def test_missing_and_ambiguous_runtime_init_are_rejected(self):
        extra = self.init.with_name('runtime_init_extra.json')
        shutil.copy2(self.init, extra)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            audit_trace(self.prefill)
        extra.unlink()
        self.init.unlink()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            audit_trace(self.prefill)

    def test_prefill_awq_modules_must_receive_layer_specific_token_rows(self):
        original = read(self.prefill)
        for index, rows in ((0, original['retained_length']), (8, original['original_length'])):
            record = copy.deepcopy(original)
            record['awq_execution']['module_calls'][index]['rows'] = rows
            replace_json(self.prefill, record)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'row counts'):
                audit_trace(self.prefill)

    def test_decode_must_execute_loaded_awq_modules_once_per_token(self):
        original = read(self.decode)
        for key, value in (('rows', 2), ('calls', 0), ('calls', 2)):
            record = copy.deepcopy(original)
            record['awq_execution']['module_calls'][0][key] = value
            replace_json(self.decode, record)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'row counts'):
                audit_trace(self.prefill)

    def test_keep100_control_preserves_all_tokens_and_is_not_d_group(self):
        p, d, initialization = read(self.prefill), read(self.decode), read(self.init)
        original = p['original_length']
        p.update(keep_ratio=1., actual_mode='native_control', retained_length=original,
                 removed=0, retained_image_tokens=p['original_image_tokens'],
                 retained_image_tokens_per_image=p['image_tokens_per_image'],
                 kept_original_positions=list(range(original)), upper_layer_kv_length=original,
                 selection=None, awq_execution=synthetic_execution(original, original))
        d['upper_layer_kv_length'] = original + 1
        initialization['keep_ratio'] = 1.
        launch_path = self.service / 'launch.json'
        launch = read(launch_path)
        launch.update(keep_ratio=1., model_name=served_model_name(1., 'navigation'))
        replace_json(launch_path, launch)
        replace_json(self.prefill, p)
        replace_json(self.decode, d)
        replace_json(self.init, initialization)
        audited = audit_trace(self.prefill, keep_ratio=1.)
        self.assertFalse(audited['combined_awq_dart'])
        self.assertEqual(audited['removed'], 0)
        self.assertEqual(audited['launch_sha256'], digest(launch_path))
        with self.assertRaises(ValueError):
            audit_trace(self.prefill)

    def test_engine_prefix_cache_chunking_and_cache_pool_cannot_drift(self):
        path = self.service / 'launch.json'
        original = read(path)
        for key, value in (('enable_prefix_caching', True), ('max_prefill_token_num', 1024),
                           ('num_gpu_blocks', 128), ('session_len', 3072)):
            record = copy.deepcopy(original)
            record['engine_config'][key] = value
            replace_json(path, record)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'configuration'):
                self.audit()

    def test_original_b_checkpoint_identity_and_integrity_are_required(self):
        path = self.service / 'launch.json'
        original = read(path)
        for kind in ('model_source', 'model_config_sha256', 'weight_header_sha256', 'full_hash'):
            record = copy.deepcopy(original)
            if kind == 'full_hash':
                record['checkpoint_verification']['full_weight_sha256_rechecked_this_run'] = False
            else:
                record[kind] = 'wrong'
            replace_json(path, record)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'B-group'):
                self.audit()
            with self.subTest(kind=kind, entry='single_trace'), self.assertRaisesRegex(ValueError, 'B-group'):
                audit_trace(self.prefill)

    def test_single_trace_requires_matching_launch_identity_and_keep_ratio(self):
        path = self.service / 'launch.json'
        original = read(path)
        for key, value in (('model_name', served_model_name(1., 'navigation')), ('keep_ratio', 1.)):
            record = copy.deepcopy(original)
            record[key] = value
            replace_json(path, record)
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_trace(self.prefill)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            audit_trace(self.prefill)

    def test_model_and_episode_membership_are_rejected(self):
        path = self.run / 'summary.json'
        original = read(path)
        for kind in ('model', 'membership'):
            changed = copy.deepcopy(original)
            if kind == 'model':
                changed['model'] = 'wrong-model'
            else:
                changed['episodes'][0]['episode_id'] = '999999'
            replace_json(path, changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.audit()


if __name__ == '__main__':
    outcome = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AuditChecks))
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    print(f'AWQ_DART_AUDIT_SYNTHETIC_UNIT_OK tests={outcome.testsRun}; '
          'original evidence unchanged; NOT a real AWQ+DART execution result')
