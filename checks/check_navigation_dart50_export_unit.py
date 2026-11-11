"""Synthetic result-pairing checks; no real C50 results are created."""
import copy
from pathlib import Path
import tempfile
import unittest

from navigation_dart50_support import ROOT, C75_MODEL, read, write
from export_navigation_dart50_compare import make_comparison


class ExportChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tmp', prefix='dart50_export_synthetic_')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        report = copy.deepcopy(read(ROOT / 'reports/benchmark_overnight200_20261031.json'))
        report.update(status='complete', completed_episode_count=150, completed_batches=[1, 2, 3],
                      all_runtime_audits_passed=True, error_count=0, model=C75_MODEL,
                      synthetic_fixture=True)
        self.c75 = self.directory / 'c75.json'
        write(self.c75, report)
        self.c50 = self.directory / 'c50'
        self.c50.mkdir()
        changed = copy.deepcopy(report)
        self.lost_id = next(row['episode_id'] for row in changed['episodes'] if row['metrics']['success'])
        for row in changed['episodes']:
            if row['episode_id'] == self.lost_id:
                row['metrics']['success'] = 0
                row['metrics']['spl'] = 0
        write(self.c50 / 'report.json', changed)

    def test_exact_pairing_counts_one_lost_success(self):
        source = read(self.c75)
        source.update(warmup_requests_excluded=8,
                      recovery={'warmup_requests_by_batch': {'1': 4, '2': 2, '3': 2}})
        source['episodes'][0]['timing_note'] = 'Synthetic recovery timing limitation'
        write(self.c75, source)
        result = make_comparison(self.c75, self.c50)
        self.assertEqual(result['same_task_count'], 150)
        self.assertEqual(result['keep50_vs_keep75']['lost_success'], [str(self.lost_id)])
        self.assertEqual(result['keep50_vs_keep75']['gained_success'], [])
        self.assertAlmostEqual(result['keep50_vs_keep75']['sr_change_percentage_points'], -100 / 150)
        timing = result['source_timing_context']['FP16_DART75']
        self.assertEqual(timing['warmup_requests_excluded'], 8)
        self.assertEqual(timing['episode_timing_notes'][0]['timing_note'],
                         'Synthetic recovery timing limitation')

    def test_partial_or_duplicate_result_cannot_be_exported(self):
        path = self.c50 / 'report.json'
        original = read(path)
        changed = copy.deepcopy(original)
        changed['status'] = 'partial_or_failed'
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'not complete'):
            make_comparison(self.c75, self.c50)
        changed = copy.deepcopy(original)
        changed['episodes'][-1] = copy.deepcopy(changed['episodes'][0])
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'identity/denominator'):
            make_comparison(self.c75, self.c50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
