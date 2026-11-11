"""Stdlib-only regression checks for new delivery helpers, using temporary files."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from prepare_navigation_reproduction import ROOT, manifests
from restore_navigation_frozen_inputs import encode_subset
from restore_navigation_sources import digest, inside, install


class DeliveryTests(unittest.TestCase):
    def test_install_is_idempotent_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = dict(path='tmp/source.py', sha256=digest(b'original'))
            self.assertEqual(install(root, record, b'original'), 'restored')
            before = (root/record['path']).stat().st_mtime_ns
            self.assertEqual(install(root, record, b'original'), 'already_verified')
            self.assertEqual(before, (root/record['path']).stat().st_mtime_ns)
            changed = dict(record, sha256=digest(b'changed'))
            with self.assertRaises(ValueError):
                install(root, changed, b'changed')
            self.assertEqual((root/record['path']).read_bytes(), b'original')

    def test_checksum_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                install(root, dict(path='tmp/source.py', sha256=digest(b'expected')), b'bad')
            self.assertEqual(list(root.iterdir()), [])

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                inside(Path(folder), '../outside.py')

    def test_subset_preserves_vocabulary_and_requested_order(self):
        original = dict(instruction_vocab=['one'], episodes=[dict(episode_id=1), dict(episode_id=2)])
        value = encode_subset(original, ['2', '1'])
        restored = json.loads(gzip.decompress(value))
        self.assertEqual(restored['instruction_vocab'], ['one'])
        self.assertEqual([ep['episode_id'] for ep in restored['episodes']], [2, 1])
        self.assertEqual(value, encode_subset(original, ['2', '1']))
        self.assertEqual(original['episodes'][0]['episode_id'], 1)

    def test_same100_matches_published_roster_and_model_identity(self):
        rows = manifests(ROOT, 'awq', 100)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['model'], 'SimpleVLN-Qwen3VL-4B-RL-stage2-awq-20261109')
        frozen = json.loads((ROOT/'datasets/navigation_20261110/batches/batch_01.json').read_text(encoding='utf-8'))
        self.assertEqual(rows[0]['episodes'], frozen['episodes'])
        self.assertEqual(rows[0]['dataset_path'], frozen['dataset_path'])
        self.assertEqual(rows[0]['source_annotation_sha256'], frozen['source_annotation_sha256'])


if __name__ == '__main__':
    unittest.main()
