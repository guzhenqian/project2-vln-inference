"""Synthetic CPU checks for the layer-internal DART selector; no model loading."""

import json
import unittest

import torch

from navigation_dart_selection import select_dart_tokens


def make_inputs(spans, sequence_length):
    keys = torch.ones(sequence_length, 1, dtype=torch.float32)
    hidden = torch.ones(sequence_length, 2, dtype=torch.float32)
    mask = torch.zeros(sequence_length, dtype=torch.bool)
    for start, end in spans:
        mask[start:end] = True
    return keys, hidden, mask


class DartSelectionChecks(unittest.TestCase):
    def test_hand_computed_lowest_cosine_without_new_pivots(self):
        spans = [(1, 13)]
        keys, hidden, mask = make_inputs(spans, 19)
        keys[1:5, 0] = torch.tensor([40., 30., 20., 10.])
        keys[13:17, 0] = torch.tensor([40., 30., 20., 10.])
        # Four axis pivots, their antipodes, then four diagonal candidates.
        hidden[1:13] = torch.tensor([
            [1., 0.], [0., 1.], [-1., 0.], [0., -1.],
            [-1., 0.], [0., -1.], [1., 0.], [0., 1.],
            [-1., -1.], [1., -1.], [-1., 1.], [1., 1.],
        ])
        hidden[13] = torch.tensor([1., 1.])
        originals = [value.clone() for value in (keys, hidden, mask)]
        result = select_dart_tokens(keys, hidden, mask, spans)
        image = result['diagnostics']['images'][0]
        self.assertEqual(image['kept_sequence_indices'], list(range(1, 10)))
        self.assertEqual(image['pivot_order'], [1, 2, 3, 4, 13, 14, 15, 16])
        self.assertEqual(image['quotas'], [1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual([r['selected_indices'] for r in image['rounds'][:5]],
                         [[5], [6], [7], [8], [9]])
        # If newly selected token 5 became the fifth pivot, it would select
        # a positive-x diagonal, not token 9. This catches accidental re-pivoting.
        self.assertEqual(image['rounds'][4]['pivot_index'], 13)
        self.assertEqual(result['sequence_keep_indices'].tolist(),
                         [0] + list(range(1, 10)) + list(range(13, 19)))
        self.assertEqual(result['visual_keep_indices'].tolist(), list(range(9)))
        for original, current in zip(originals, (keys, hidden, mask)):
            self.assertTrue(torch.equal(original, current))
        json.dumps(result['diagnostics'])

    def test_ties_are_stable_and_use_lower_sequence_indices(self):
        spans = [(1, 9)]
        keys, hidden, mask = make_inputs(spans, 15)
        expected = None
        for _ in range(3):
            result = select_dart_tokens(keys, hidden, mask, spans)
            image = result['diagnostics']['images'][0]
            self.assertEqual(image['visual_pivot_indices'], [1, 2, 3, 4])
            self.assertEqual(result['diagnostics']['text_pivot_indices'], [9, 10, 11, 12])
            self.assertEqual(image['kept_sequence_indices'], [1, 2, 3, 4, 5, 6])
            serialized = json.dumps(result['diagnostics'], sort_keys=True)
            if expected is not None:
                self.assertEqual(expected, serialized)
            expected = serialized

    def test_per_image_ceil_budget_and_visual_row_mapping(self):
        spans = [(1, 8), (10, 20)]
        keys, hidden, mask = make_inputs(spans, 27)
        # A large key in the inter-image text must not enter the suffix pool.
        keys[9, 0] = 10000
        result = select_dart_tokens(keys, hidden, mask, spans)
        self.assertEqual([d['kept_count'] for d in result['diagnostics']['images']], [6, 8])
        self.assertEqual(result['diagnostics']['text_pivot_indices'], [20, 21, 22, 23])
        self.assertEqual(result['visual_keep_indices'].tolist(),
                         list(range(6)) + list(range(7, 15)))
        kept = result['sequence_keep_indices'].tolist()
        self.assertEqual(kept, [0] + list(range(1, 7)) + [8, 9]
                         + list(range(10, 18)) + list(range(20, 27)))

    def test_300_tokens_keep_exactly_225_including_pivots(self):
        spans = [(1, 301), (303, 603)]
        keys, hidden, mask = make_inputs(spans, 610)
        result = select_dart_tokens(keys, hidden, mask, spans)
        for image in result['diagnostics']['images']:
            self.assertEqual(image['kept_count'], 225)
            self.assertEqual(image['quotas'], [28, 28, 28, 28, 28, 27, 27, 27])
            self.assertEqual(len(set(image['kept_sequence_indices'])), 225)
        self.assertEqual(result['visual_keep_indices'].numel(), 450)
        self.assertEqual(result['diagnostics']['nonvisual_tokens_preserved'], 10)

    def test_identity_needs_no_pivots_and_preserves_all_rows(self):
        spans = [(1, 3)]
        keys, hidden, mask = make_inputs(spans, 3)
        result = select_dart_tokens(keys, hidden, mask, spans, keep_ratio=1)
        self.assertEqual(result['sequence_keep_indices'].tolist(), [0, 1, 2])
        self.assertEqual(result['visual_keep_indices'].tolist(), [0, 1])
        self.assertEqual(result['diagnostics']['mode'], 'identity')
        self.assertEqual(result['diagnostics']['images'][0]['rounds'], [])

    def test_invalid_ratios_and_pivot_capacity_fail(self):
        spans = [(1, 9)]
        args = make_inputs(spans, 15)
        for ratio in (0, -0.1, 1.01, float('nan'), float('inf'), True, '0.75'):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                select_dart_tokens(*args, spans, keep_ratio=ratio)
        with self.assertRaisesRegex(ValueError, 'budget'):
            select_dart_tokens(*args, spans, keep_ratio=0.1)
        with self.assertRaisesRegex(ValueError, 'visual pivot'):
            select_dart_tokens(*make_inputs([(1, 4)], 10), [(1, 4)])
        with self.assertRaisesRegex(ValueError, 'text pivot'):
            select_dart_tokens(*make_inputs([(1, 9)], 12), [(1, 9)])
        for count in (0, -1, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_dart_tokens(*args, spans, visual_pivots=count)

    def test_invalid_spans_masks_and_shapes_fail(self):
        spans = [(1, 9)]
        keys, hidden, mask = make_inputs(spans, 15)
        for invalid in ([], [(1, 1)], [(-1, 9)], [(1, 16)], [(1, 8)],
                        [(1, 6), (5, 9)], [(5, 9), (1, 5)], [(1.0, 9)]):
            with self.subTest(spans=invalid), self.assertRaises(ValueError):
                select_dart_tokens(keys, hidden, mask, invalid)
        gapped = mask.clone()
        gapped[4] = False
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            select_dart_tokens(keys, hidden, gapped, spans)
        with self.assertRaises(ValueError):
            select_dart_tokens(keys, hidden[:-1], mask, spans)
        with self.assertRaises(ValueError):
            select_dart_tokens(keys, hidden, mask.long(), spans)
        with self.assertRaises(ValueError):
            select_dart_tokens(keys.long(), hidden, mask, spans)

    def test_nonfinite_inputs_fail_even_for_identity(self):
        spans = [(1, 9)]
        for which in (0, 1):
            for value in (float('nan'), float('inf'), -float('inf')):
                for ratio in (0.75, 1.0):
                    args = list(make_inputs(spans, 15))
                    args[which][0, 0] = value
                    with self.subTest(which=which, value=value, ratio=ratio):
                        with self.assertRaisesRegex(ValueError, 'NaN or infinity'):
                            select_dart_tokens(*args, spans, keep_ratio=ratio)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DartSelectionChecks)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    print(f'DART_SELECTION_UNIT_OK tests={outcome.testsRun} (synthetic CPU only)')
