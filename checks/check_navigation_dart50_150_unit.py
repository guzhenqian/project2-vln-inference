"""Offline standard-library tests; no HTTP, model, GPU, service or simulator."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_navigation_dart50_150 as target


class Dart150Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = {'ROOT': self.root, 'PLAN': self.root / 'data/plan',
                      'ORIGINAL': self.root / 'data/original', 'BASELINE': self.root / 'baseline.json',
                      'SINGLE_CONFIG': self.root / 'single.json', 'SERVICE': self.root / 'logs/service',
                      'SINGLE_RUN': self.root / 'logs/single', 'SINGLE_AUDIT': self.root / 'single_audit.json'}
        self.patches = [mock.patch.object(target, name, value) for name, value in self.paths.items()]
        for patch in self.patches:
            patch.start()
        target.PLAN.mkdir(parents=True)
        target.ORIGINAL.mkdir(parents=True)
        target.SERVICE.mkdir(parents=True)
        self.single = {'model': target.MODEL, 'dtype': 'FP16', 'dart_keep_ratio': 0.5,
                       'dart_profile': 'navigation', 'backend': 'eager3072', 'source_precision': 'FP16 noAWQ'}
        target.write(target.SINGLE_CONFIG, self.single)
        self.originals, records, old = [], [], []
        for index in range(1, 4):
            ids = [str((index - 1) * 50 + value) for value in range(50)]
            batch = {'batch': index, 'episodes': ids, 'trajectory_ids': ['t' + eid for eid in ids],
                     'max_episode_steps': 400, 'success_distance_m': 3.0, 'seed': 7,
                     'temperature': 0.3, 'model_inputs': 'authorRGB', 'unexpected_core': {'preserve': True}}
            self.originals.append(batch)
            source = target.ORIGINAL / f'batch_{index:02d}.json'
            target.write(source, batch)
            path = target.PLAN / source.name
            target.write(path, target.clone_batch(batch, self.single))
            records.append({'batch': index, 'episodes': ids, 'trajectory_ids': batch['trajectory_ids'],
                            'path': target.rel(path), 'sha256': target.digest(path),
                            'source_manifest': target.rel(source), 'source_sha256': target.digest(source)})
            old.extend({'episode_id': eid, 'terminal_stop_source': 'model',
                        'metrics': {'success': 1, 'spl': 0.8}} for eid in ids)
        target.write(target.BASELINE, {'episodes': old})
        self.pins = {target.rel(target.ORIGINAL / f'batch_{i:02d}.json'):
                     target.digest(target.ORIGINAL / f'batch_{i:02d}.json') for i in range(1, 4)}
        self.pins[target.rel(target.SINGLE_CONFIG)] = target.digest(target.SINGLE_CONFIG)
        (self.root / 'checks').mkdir()
        for name in target.RUNNERS:
            path = self.root / 'checks' / name
            path.write_text('# fake original runner for offline tests\n')
            self.pins['checks/' + name] = target.digest(path)
        self.plan = {'planned_count': 150, 'model': target.MODEL, 'batches': records,
                     'service_directory': target.rel(target.SERVICE), 'source_checks': self.pins,
                     'latency': 'model_called=true only; warmups not in steps', 'memory': 'whole-card',
                     'order': 'unchanged manifest order; grouped by Habitat', 'old_timing_note': 'resumed FP16 batch1'}
        target.write(target.PLAN / 'manifest.json', self.plan)
        self.run_dir = self.root / 'logs/navigation_dart150_keep50_test'
        self.run_dir.mkdir()
        target.write(self.run_dir / 'frozen_plan.json', self.plan)
        target.write(self.run_dir / 'status.json', {'status': 'failed'})

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temp.cleanup()

    def save_episode(self, batch_index, eid, *, success=True, error=None, result=True):
        folder = self.run_dir / f'batch_{batch_index:02d}' / ('episode_' + eid)
        folder.mkdir(parents=True)
        rows = [{'episode_id': eid, 'step': 0, 'model_called': True, 'model_request_s': 2.0,
                 'action': 'MOVE_FORWARD', 'error': None},
                {'episode_id': eid, 'step': 1, 'model_called': False, 'model_request_s': 0.0,
                 'action': 'STOP', 'error': error}]
        (folder / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        if result:
            value = {'episode_id': eid, 'trajectory_id': 't' + eid, 'steps': 2,
                     'start_distance_m': 10.0, 'path_length_m': 20.0,
                     'metrics': {'success': int(success), 'spl': 0.5 if success else 0.0,
                                 'distance_to_goal': 1.0 if success else 5.0, 'collisions': {'count': 1}},
                     'terminal_stop_source': 'error_fallback' if error else 'model',
                     'error_count': int(bool(error))}
            target.write(folder / 'result.json', value)

    def test_clone_preserves_every_unknown_core_field_and_order(self):
        original = self.originals[0]
        snapshot = copy.deepcopy(original)
        clone = target.clone_batch(original, self.single)
        self.assertEqual(original, snapshot)
        self.assertEqual(clone['episodes'], original['episodes'])
        self.assertEqual(clone['unexpected_core'], original['unexpected_core'])
        self.assertEqual(clone['seed'], 7)
        self.assertEqual(clone['model'], target.MODEL)
        bad = dict(self.single, dart_keep_ratio=1.0)
        with self.assertRaises(ValueError):
            target.clone_batch(original, bad)

    def test_prepare_and_check_are_offline_and_refuse_overwrite_or_mutation(self):
        # Use a separate not-yet-existing plan; all inputs are synthetic files.
        target.PLAN = self.root / 'data/new_plan'
        with mock.patch.object(target, 'collect_inputs', return_value=(self.originals, self.single, self.pins)), \
                mock.patch.object(target.urllib.request, 'urlopen', side_effect=AssertionError('HTTP forbidden')), \
                mock.patch.object(target.subprocess, 'Popen', side_effect=AssertionError('process forbidden')), \
                contextlib.redirect_stdout(io.StringIO()):
            plan = target.prepare()
            self.assertEqual(plan['planned_count'], 150)
            self.assertEqual(target.check()['planned_count'], 150)
            with self.assertRaises(ValueError):
                target.prepare()
            path = target.PLAN / 'batch_01.json'
            changed = target.read(path)
            changed['seed'] = 8
            target.write(path, changed)
            with self.assertRaises(ValueError):
                target.check()

    def test_partial_summary_pairs_only_completed_and_counts_errors_without_call_flag(self):
        self.save_episode(1, '0', success=False)
        folder = self.run_dir / 'batch_01/episode_1'
        folder.mkdir()
        (folder / 'steps.jsonl').write_text(json.dumps({'episode_id': '1', 'step': 0,
            'action': 'STOP', 'error': 'HTTP timeout', 'model_request_s': 0.8}) + '\n')
        (self.run_dir / 'batch_01/console.log').write_text('WARMUP 1 STOP 9.0\nWARMUP 2 STOP 8.0\n')
        (self.run_dir / 'batch_01/gpu_samples.csv').write_text('now,12000,90\nnow,11700,50\n')
        report = target.summarize(self.run_dir, self.plan)
        self.assertEqual(report['completed_episode_count'], 1)
        self.assertEqual(report['actual_request_latency_s']['count'], 1)
        self.assertEqual(report['actual_request_latency_s']['median'], 2.0)
        self.assertEqual(report['paired_comparison']['count'], 1)
        self.assertEqual(report['paired_comparison']['fp16_sr'], 1)
        self.assertEqual(report['paired_comparison']['dart_sr'], 0)
        self.assertEqual(report['errors_missing_model_called_flag'], 1)
        self.assertEqual(report['error_count'], 1)
        self.assertEqual(report['saved_action_steps'], 3)
        self.assertEqual(report['sampled_whole_card_peak_mib'], 12000)
        self.assertEqual(len(report['pending_episodes']), 149)
        self.assertEqual(report['status'], 'partial_or_failed')

    def test_fallback_stop_never_counts_as_model_success(self):
        self.save_episode(1, '0', success=True, error='invalid action fallback')
        report = target.summarize(self.run_dir, self.plan)
        self.assertEqual(report['sr'], 0)
        self.assertEqual(report['spl'], 0)
        self.assertEqual(report['fallback_stop_success_count'], 1)
        self.assertEqual(report['error_count'], 1)

    def test_150_completion_requires_all_three_batches_and_runtime_audits(self):
        for item in self.plan['batches']:
            index = item['batch']
            for eid in item['episodes']:
                self.save_episode(index, eid)
            directory = self.run_dir / f'batch_{index:02d}'
            target.write(directory / 'summary.json', {'status': 'complete', 'episode_count': 50,
                         'episodes': [target.read(directory / ('episode_' + eid) / 'result.json')
                                      for eid in reversed(item['episodes'])],
                         'warmup_requests_excluded': 2, 'model': target.MODEL, 'error_count': 0})
            target.write(directory / 'manifest.json', target.read(target.root_path(item['path'])))
            target.write(directory / 'provenance.json', {'manifest_sha256': item['sha256'],
                         'scripts': {name: self.plan['source_checks']['checks/' + name] for name in target.RUNNERS}})
            target.write(directory / 'runtime_audit.json', {'passed': True, 'status': 'passed'})
        report = target.summarize(self.run_dir, self.plan)
        self.assertEqual(report['completed_episode_count'], 150)
        self.assertEqual(report['actual_request_latency_s']['count'], 150)
        self.assertEqual(report['status'], 'partial_or_failed')
        target.write(self.run_dir / 'status.json', {'status': 'complete'})
        self.assertEqual(target.summarize(self.run_dir, self.plan)['status'], 'complete')
        target.write(self.run_dir / 'batch_03/runtime_audit.json', {'passed': False, 'status': 'failed'})
        self.assertEqual(target.summarize(self.run_dir, self.plan)['status'], 'partial_or_failed')

    def test_request_error_stops_owned_evaluator_and_never_starts_batch_two(self):
        children = []

        class FakeChild:
            def __init__(fake, command, **kwargs):
                fake.pid, fake.code, fake.terminated = 123456, None, False
                fake.command = command
                directory = Path(command[command.index('--output') + 1])
                folder = directory / 'episode_0'
                folder.mkdir()
                (folder / 'steps.jsonl').write_text(json.dumps({'episode_id': '0', 'step': 0,
                    'action': 'STOP', 'error': 'injected HTTP error', 'model_request_s': 1.0}) + '\n')
                fake.stdout = io.StringIO('MODEL_STEP 0 1 STOP distance=4 request_s=1\n')
                children.append(fake)

            def poll(fake):
                return fake.code

            def wait(fake, timeout=None):
                fake.code = 0
                return fake.code

        def terminate(fake):
            fake.terminated, fake.code = True, -15

        with mock.patch.object(target, 'check', return_value=self.plan), \
                mock.patch.object(target, 'verify_service', return_value={'data': [{'id': target.MODEL}]}), \
                mock.patch.object(target.subprocess, 'Popen', side_effect=FakeChild), \
                mock.patch.object(target, 'terminate_owned', side_effect=terminate), \
                mock.patch.object(target, 'audit_completed_batch', side_effect=ValueError('incomplete batch')), \
                mock.patch.object(target.urllib.request, 'urlopen', side_effect=AssertionError('HTTP forbidden')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(target.run(), 1)
        self.assertEqual(len(children), 1)
        self.assertTrue(children[0].terminated)
        self.assertIn('checks/run_simplevln_diagnostic.py', children[0].command)
        self.assertEqual(children[0].command[children[0].command.index('--policy') + 1], 'memory')
        outputs = [path for path in (self.root / 'logs').glob('navigation_dart150_keep50_*')
                   if path != self.run_dir]
        self.assertEqual(len(outputs), 1)
        self.assertEqual(target.read(outputs[0] / 'status.json')['status'], 'failed')
        self.assertFalse((outputs[0] / 'batch_02').exists())
        self.assertEqual(target.read(outputs[0] / 'report.json')['completed_episode_count'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
