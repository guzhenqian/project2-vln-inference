"""CPU-only dependency, persistence and safe handover tests for overnight C50."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import navigation_dart50_support as support
import run_navigation_dart50_night as sequence
import check_navigation_dart_request_learning as c75_request
import check_navigation_dart50_request as c50_request


class SequenceChecks(unittest.TestCase):
    def test_partial_c75_cannot_start_c50(self):
        with mock.patch.object(support, 'read', return_value={'status': 'partial_or_failed'}):
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                support.require_c75_complete(support.ROOT / 'logs/fake_report.json')

    def test_same_inputs_and_generation_settings_except_model(self):
        for index in (0, 1):
            original = c75_request.prepare_input(.75, index, 'navigation')
            candidate = c50_request.prepare_input(.5, index, 'navigation')
            self.assertEqual(original[0], candidate[0])
            self.assertEqual(original[2:4], candidate[2:4])
            a, b = json.loads(original[-1]), json.loads(candidate[-1])
            self.assertNotEqual(a.pop('model'), b.pop('model'))
            self.assertEqual(a, b)

    def test_transient_replace_conflict_retries_without_losing_data(self):
        with tempfile.TemporaryDirectory(dir=support.ROOT / 'tmp', prefix='dart50_write_test_') as tmp:
            path = Path(tmp) / 'status.json'
            path.write_text('{"before": true}')
            native = Path.replace
            calls = []

            def replace(source, target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError('synthetic sharing conflict')
                return native(source, target)

            with mock.patch.object(Path, 'replace', replace), mock.patch.object(support.time, 'sleep'):
                support.write(path, {'after': True})
            self.assertEqual(len(calls), 3)
            self.assertEqual(support.read(path), {'after': True})
            self.assertFalse(list(Path(tmp).glob('*.tmp')))

    def test_active_evaluator_blocks_service_termination_before_http(self):
        fake = {100: {'cwd': str(sequence.ROOT), 'words': ['python', 'checks/run_simplevln_diagnostic.py'],
                      'ppid': 1, 'start': '1'}}
        with mock.patch.object(sequence, 'processes', return_value=fake), \
             mock.patch.object(sequence, 'urlopen', side_effect=AssertionError('HTTP forbidden')), \
             mock.patch.object(sequence.os, 'kill', side_effect=AssertionError('signal forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'still active'):
                sequence.release_completed_c75(sequence.ROOT / 'tmp')

    def test_unexpected_service_identity_is_never_terminated(self):
        fake = {100: {'cwd': str(sequence.ROOT), 'words': ['python', 'checks/serve_navigation_dart_learning.py',
                 '--keep-ratio', '0.75', '--profile', 'navigation'], 'ppid': 1, 'start': '1'}}
        with mock.patch.object(sequence, 'processes', return_value=fake), \
             mock.patch.object(sequence, 'urlopen', return_value=io.StringIO('{"data":[{"id":"unexpected"}]}')), \
             mock.patch.object(sequence.os, 'kill', side_effect=AssertionError('signal forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'not the completed C75'):
                sequence.release_completed_c75(sequence.ROOT / 'tmp')


if __name__ == '__main__':
    unittest.main(verbosity=2)
