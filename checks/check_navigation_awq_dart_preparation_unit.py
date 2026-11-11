"""CPU-only checks for independent D inputs, launch gates and resource isolation."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)

import check_navigation_dart_request_learning as c_request
import check_navigation_awq_dart_request as d_request
import navigation_awq_dart_preflight as preflight
import serve_navigation_awq_dart as service


class PreparationTests(unittest.TestCase):
    def test_inputs_only_change_model_identity(self):
        for keep in (1., .75):
            for sample in (0, 1):
                c = c_request.prepare_input(keep, sample, 'navigation')
                d = d_request.prepare_input(keep, sample, 'navigation')
                self.assertEqual(c[0], d[0])
                self.assertEqual(c[2:4], d[2:4])
                old, new = json.loads(c[-1]), json.loads(d[-1])
                self.assertNotEqual(old.pop('model'), new.pop('model'))
                self.assertEqual(old, new)

    def test_c_identity_cannot_pass_d_request(self):
        expected = d_request.model_name(.75, 'navigation')
        with self.assertRaises(ValueError):
            d_request.validate_model_listing({'data': [{'id': c_request.model_name(.75, 'navigation')}]}, expected)

    def test_running_c_blocks_before_report_read(self):
        with mock.patch.object(preflight, 'read', return_value={'status': 'running'}) as read:
            with self.assertRaisesRegex(ValueError, 'not completed'):
                preflight.require_c_finished()
            self.assertEqual(read.call_count, 1)

    def test_complete_c_requires_final_audit(self):
        with mock.patch.object(preflight, 'read', side_effect=[{'status': 'complete'}, {'status': 'partial_or_failed'}]), \
             mock.patch.object(Path, 'exists', return_value=False):
            with self.assertRaisesRegex(ValueError, 'final report'):
                preflight.require_c_finished()

    def test_launcher_default_does_not_acquire_resources(self):
        with mock.patch.object(sys, 'argv', ['serve_navigation_awq_dart.py']), \
             mock.patch.object(service, 'run_service', side_effect=AssertionError('service start forbidden')), \
             mock.patch.object(service, 'require_c_finished', side_effect=AssertionError('run gate reached')), \
             mock.patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            service.main()
        self.assertFalse(torch.cuda.is_initialized())

    def test_atomic_lock_rejects_duplicate_and_releases_owned(self):
        with tempfile.TemporaryDirectory(dir=preflight.ROOT / 'tmp', prefix='awq_dart_lock_test_') as tmp:
            root = Path(tmp)
            (root / 'logs').mkdir()
            with mock.patch.object(service, 'ROOT', root):
                with service.exclusive_service_lock():
                    self.assertTrue((root / 'logs/navigation_awq_dart_service.active.lock').is_file())
                    with self.assertRaises(FileExistsError):
                        with service.exclusive_service_lock():
                            self.fail('Duplicate lock acquired')
                self.assertFalse((root / 'logs/navigation_awq_dart_service.active.lock').exists())

    def test_engine_is_fixed_and_matches_c_navigation(self):
        engine = service.make_engine('navigation')
        self.assertEqual((engine.dtype, engine.num_gpu_blocks, engine.block_size, engine.max_prefill_token_num,
                          engine.session_len, engine.max_batch_size, engine.quant_policy),
                         ('float16', 64, 64, 3072, 4096, 1, 0))
        self.assertTrue(engine.eager_mode)
        self.assertFalse(engine.enable_prefix_caching)

    def test_frozen150_plan_and_single_policy_unchanged(self):
        plan = preflight.read(preflight.ROOT / 'setup/navigation-awq-dart-validation-plan-20261108.json')
        ids = []
        for item in plan['batches']:
            path = preflight.ROOT / item['original_manifest']
            self.assertEqual(preflight.digest(path), item['sha256'])
            original = preflight.read(path)
            self.assertEqual((original['episodes'], original['trajectory_ids']),
                             (item['episodes'], item['trajectory_ids']))
            ids.extend(item['episodes'])
        self.assertEqual(len(ids), 150)
        self.assertEqual(len(set(ids)), 150)
        c = preflight.read(preflight.ROOT / 'setup/navigation-dart-single-learning-20261108.json')
        d = preflight.read(preflight.ROOT / plan['single_candidate'])
        for key in ('version', 'scope', 'model', 'dtype', 'backend', 'source_precision',
                    'differences_from_author_benchmark', 'execution_gate', 'weight_quantization'):
            c.pop(key, None)
            d.pop(key, None)
        self.assertEqual(c, d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
