"""Standard-library offline replay and corruption checks for the DART audit.

Copies only saved JSON/JSONL evidence into isolated tmp directories. No original
logs are modified, and no HTTP, simulator, GPU or model imports are used.
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

from audit_navigation_dart_run import audit_batch, audit_trace


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'logs/navigation_dart_single_learning_20261108T091551Z_4612'
SERVICE = ROOT / 'logs/navigation_dart_service_learning_20261108T065350824909Z_284bc210'


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
        self.temp = tempfile.TemporaryDirectory(prefix='dart_audit_unit_', dir=ROOT / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'
        self.service = Path(self.temp.name) / 'service'
        for source_root, destination, paths in ((RUN, self.run, self.run_files),
                                                 (SERVICE, self.service, self.service_files)):
            for path in paths:
                target = destination / path.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        self.prefill = self.service / self.current[-1].name
        self.decode = self.prefill.with_name(self.prefill.name.replace('prefill_', 'first_decode_'))

    def audit(self):
        return audit_batch(self.run, self.service, before_names=self.before)

    def change_prefill(self, key, value):
        record = read(self.prefill)
        record[key] = value
        replace_json(self.prefill, record)

    def test_saved_replay_snapshot_and_completed_mtime_window(self):
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

    def test_awq_and_kv_quantization_are_rejected(self):
        original = read(self.prefill)
        for key, value in (('awq', True), ('kv_quant_policy', 4), ('weight_dtype', 'bfloat16')):
            with self.subTest(key=key):
                changed = copy.deepcopy(original)
                changed[key] = value
                replace_json(self.prefill, changed)
                with self.assertRaisesRegex(ValueError, 'Expected FP16'):
                    audit_trace(self.prefill)

    def test_quantized_service_identity_is_rejected(self):
        path = self.service / 'launch.json'
        launch = read(path)
        launch['weight_quantization'] = 'awq'
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
        with self.assertRaisesRegex(ValueError, 'Keep75 budget'):
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
        with self.assertRaisesRegex(ValueError, 'Insufficient cache pool'):
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
    print(f'DART_AUDIT_OFFLINE_UNIT_OK tests={outcome.testsRun}; original evidence unchanged')
