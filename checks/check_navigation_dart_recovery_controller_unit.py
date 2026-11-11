"""Offline recovery controller regressions: file writes and evidence accounting."""
import copy
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resume_navigation_dart150 as controller


class AtomicWriteTests(unittest.TestCase):
    def test_transient_permission_retry_preserves_old_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'status.json'
            target.write_text('{"old":true}')
            original = Path.replace
            seen = []

            def replace(source, destination):
                seen.append(source)
                if len(seen) < 3:
                    self.assertEqual(json.loads(target.read_text()), {'old': True})
                    raise PermissionError(errno.EACCES, 'fixture sharing conflict')
                return original(source, destination)

            with patch.object(Path, 'replace', replace), patch.object(controller.time, 'sleep') as sleep:
                controller.write(target, {'new': True})
            self.assertEqual(json.loads(target.read_text()), {'new': True})
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(len(set(seen)), 1)
            self.assertNotEqual(seen[0].name, 'status.json.tmp')
            self.assertEqual(list(Path(temporary).iterdir()), [target])

    def test_persistent_permission_failure_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'status.json'
            target.write_text('{"old":true}')
            with patch.object(Path, 'replace', side_effect=PermissionError(errno.EACCES, 'locked')) as replace:
                with patch.object(controller.time, 'sleep') as sleep:
                    with self.assertRaises(PermissionError):
                        controller.write(target, {'new': True}, attempts=3)
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(json.loads(target.read_text()), {'old': True})

    def test_unrelated_io_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'status.json'
            with patch.object(Path, 'replace', side_effect=OSError(errno.ENOSPC, 'disk full')):
                with patch.object(controller.time, 'sleep') as sleep:
                    with self.assertRaises(OSError):
                        controller.write(target, {})
            sleep.assert_not_called()

    def test_exclusive_lock_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'lock'
            controller.write(target, {'token': 'owner'}, exclusive=True)
            with self.assertRaises(FileExistsError):
                controller.write(target, {'token': 'intruder'}, exclusive=True)
            self.assertEqual(controller.read(target), {'token': 'owner'})


class RecoverySummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = controller.read(controller.base.PLAN / 'manifest.json')
        cls.baseline = {str(item['episode_id']): item for item in controller.read(controller.base.BASELINE)['episodes']}
        cls.completed = sorted(folder.name[8:] for folder in controller.PREFIX_BATCH.glob('episode_*')
                               if (folder / 'result.json').exists())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='recovery_controller_fixture_', dir=controller.ROOT / 'tmp')
        self.output = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        controller.write(self.output / 'frozen_plan.json', self.plan)
        controller.write(self.output / 'recovery_plan.json', controller.recovery_plan(self.plan, self.completed))
        controller.write(self.output / 'status.json', {'status': 'running'})

    def create_batch(self, item, complete=True):
        directory = self.output / ('batch_%02d' % item['batch'])
        directory.mkdir()
        (directory / 'console.log').write_text('WARMUP 1 STOP 0.1\nWARMUP 2 STOP 0.1\n')
        (directory / 'gpu_samples.csv').write_text('fixture, 1, 0\n')
        results, actual_calls = [], 0
        for eid in item['episodes']:
            old_folder = controller.PREFIX_BATCH / ('episode_' + eid)
            old_rows, _ = controller.base.step_rows(old_folder / 'steps.jsonl') if item['batch'] == 1 else ([], False)
            original = controller.read(old_folder / 'result.json') if item['batch'] == 1 and (old_folder / 'result.json').exists() else None
            if not complete and not old_rows:
                continue
            folder = directory / ('episode_' + eid)
            folder.mkdir()
            rows = [dict(row, prefix_reused=True) for row in old_rows]
            if original is not None:
                result = dict(original, prefix_reused_steps=len(rows), timing_note='Completed original episode retained; replay verification time excluded')
            elif complete:
                metrics = {'distance_to_goal': 10.0, 'success': 0.0, 'spl': 0.0,
                           'collisions': {'count': 0, 'is_collision': False}}
                rows.append({'step': len(rows), 'episode_id': eid, 'action': 'STOP',
                             'model_called': True, 'model_request_s': 0.25, 'error': None,
                             'metrics': metrics})
                result = {'episode_id': eid, 'trajectory_id': self.baseline[eid]['trajectory_id'],
                          'steps': len(rows), 'error_count': 0, 'metrics': metrics,
                          'start_distance_m': 10.0, 'path_length_m': 0.0,
                          'terminal_stop_source': 'model', 'elapsed_s': 1.0,
                          'prefix_reused_steps': len(old_rows)}
                if old_rows:
                    result['timing_note'] = 'Interrupted episode lower-bound loop time; excludes original image/log IO and downtime'
            else:
                result = None
            (folder / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            actual_calls += sum(row.get('model_called') is True for row in rows)
            if result is not None:
                controller.write(folder / 'result.json', result)
                results.append(result)
        if complete:
            batch = controller.read(controller.root_path(item['path']))
            controller.write(directory / 'manifest.json', batch)
            scripts = {name: self.plan['source_checks']['checks/' + name] for name in controller.base.RUNNERS}
            provenance = {'scripts': scripts, 'manifest_sha256': item['sha256']}
            if item['batch'] == 1:
                del scripts['run_simplevln_diagnostic.py']
                scripts['resume_overnight_batch.py'] = controller.digest(controller.ROOT / 'checks/resume_overnight_batch.py')
                provenance.update(prefix_run=str(controller.PREFIX_BATCH),
                                  prefix_manifest_sha256=controller.digest(controller.PREFIX_BATCH / 'manifest.json'))
            controller.write(directory / 'provenance.json', provenance)
            controller.write(directory / 'summary.json', {'status': 'complete', 'model': controller.base.MODEL,
                             'episode_count': len(results), 'episodes': results,
                             'error_count': 0, 'warmup_requests_excluded': 2})
            controller.write(directory / 'runtime_audit.json', {'status': 'passed', 'passed': True,
                             'actual_model_calls': actual_calls, 'warmup_requests': 4 if item['batch'] == 1 else 2})
        return directory

    def test_partial_prefix_counts_original_calls_once_and_retains_old_memory(self):
        self.create_batch(self.plan['batches'][0], complete=False)
        report = controller.summarize(self.output, self.plan)
        self.assertEqual(report['completed_episode_count'], 31)
        self.assertEqual(report['saved_action_steps'], 3639)
        self.assertEqual(report['actual_request_latency_s']['count'], 1014)
        self.assertEqual(report['warmup_requests_excluded'], 4)
        self.assertEqual(report['recovery']['warmup_requests_by_batch'], {'1': 4, '2': 0, '3': 0})
        self.assertEqual(report['sampled_whole_card_peak_mib'], max(controller.gpu_peaks(controller.PREFIX_BATCH / 'gpu_samples.csv')))
        self.assertEqual(report['status'], 'partial_or_failed')

    def test_complete_150_reports_eight_warmups_and_deduplicated_requests(self):
        for item in self.plan['batches']:
            self.create_batch(item)
        controller.write(self.output / 'status.json', {'status': 'complete'})
        report = controller.summarize(self.output, self.plan)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['completed_episode_count'], 150)
        self.assertEqual(report['warmup_requests_excluded'], 8)
        self.assertEqual(report['actual_request_latency_s']['count'], 1014 + 119)
        self.assertEqual(report['recovery']['reused_steps_in_saved_rows'], 3639)
        self.assertEqual(report['recovery']['reused_calls_in_saved_rows'], 1014)

    def test_changed_reused_row_is_rejected(self):
        old, _ = controller.base.step_rows(controller.PREFIX_BATCH / 'episode_1550/steps.jsonl')
        rows = [dict(row, prefix_reused=True) for row in old]
        rows[0]['model_request_s'] += 1
        with self.assertRaisesRegex(ValueError, 'Replayed prefix row changed'):
            controller.check_reused_rows('1550', rows)

    def test_changed_completed_elapsed_is_rejected(self):
        eid = self.completed[0]
        folder = controller.PREFIX_BATCH / ('episode_' + eid)
        rows, _ = controller.base.step_rows(folder / 'steps.jsonl')
        result = dict(controller.read(folder / 'result.json'), prefix_reused_steps=len(rows), timing_note='fixture')
        result['elapsed_s'] += 1
        with self.assertRaisesRegex(ValueError, 'Completed original metrics/elapsed changed'):
            controller.check_reused_rows(eid, [dict(row, prefix_reused=True) for row in rows], result)

    def test_failed_controller_cannot_claim_complete(self):
        for item in self.plan['batches']:
            self.create_batch(item)
        controller.write(self.output / 'status.json', {'status': 'failed'})
        self.assertEqual(controller.summarize(self.output, self.plan)['status'], 'partial_or_failed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
