"""Offline exporter guards; synthetic D evidence stays in a temporary directory."""
import copy
import tempfile
import unittest
from pathlib import Path

import export_navigation_abcd150 as target


class ExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='abcd_export_unit_', dir=target.ROOT / 'tmp')
        self.run = Path(self.temp.name)
        self.c = target.read(target.SOURCES['C'])
        self.d = copy.deepcopy(self.c)
        self.d['controller_state']['model'] = target.served_model_name(.75)
        count = self.d['actual_request_latency_s']['count']
        counts = [count // 3, count // 3, count - 2 * (count // 3)]
        self.d['runtime_audits'] = [{'batch': i + 1, 'audit': {'passed': True,
            'actual_model_calls': n, 'warmup_requests': 2, 'runtime_trace_count': n + 2}}
            for i, n in enumerate(counts)]

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, value):
        target.write(self.run / 'report.json', value)
        return target.make_comparison(self.run)

    def test_paired_results_have_exact150_and_identical_c_is_zero_difference(self):
        result = self.evaluate(self.d)
        self.assertEqual(result['same_task_count'], 150)
        self.assertEqual(result['metrics']['D']['success_count'], 104)
        pair = result['paired_comparisons']['D_vs_C']
        self.assertEqual(pair['gained_success'], [])
        self.assertEqual(pair['lost_success'], [])
        self.assertEqual(pair['sr_change_percentage_points'], 0)
        self.assertEqual(pair['request_total_change_percent'], 0)

    def test_reject_partial_wrong_membership_or_inconsistent_runtime_evidence(self):
        for kind in ('partial', 'trajectory', 'missing_audit', 'wrong_call_count', 'duplicate_episode'):
            with self.subTest(kind=kind):
                value = copy.deepcopy(self.d)
                if kind == 'partial':
                    value['status'] = 'partial_or_failed'
                elif kind == 'trajectory':
                    value['episodes'][0]['trajectory_id'] = 'incorrect'
                elif kind == 'missing_audit':
                    value['runtime_audits'].pop()
                elif kind == 'wrong_call_count':
                    value['runtime_audits'][0]['audit']['actual_model_calls'] += 1
                else:
                    value['episodes'][-1] = value['episodes'][0]
                with self.assertRaises(ValueError):
                    self.evaluate(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
