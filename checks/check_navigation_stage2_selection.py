"""Compare selected-query scoring with independent dense causal GQA attention."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
import math
import torch
from navigation_stage2_selection import select_attention_tokens, sparse_queries


class SelectionTests(unittest.TestCase):
    def test_dense_attention_and_disjoint_spans(self):
        torch.manual_seed(7)
        q, k = torch.randn(91, 4, 8), torch.randn(91, 2, 8)
        mask = torch.zeros(91, dtype=torch.bool)
        spans = [(5, 29), (36, 60)]
        for start, end in spans:
            mask[start:end] = True
        dense = torch.empty(4, 91, 91)
        for head in range(4):
            logits = q[:, head] @ k[:, head//2].t() / math.sqrt(8)
            logits.masked_fill_(torch.ones(91, 91, dtype=torch.bool).triu(1), -torch.inf)
            dense[head] = logits.softmax(-1)
        for method, queries in [('fastv', torch.tensor([90])), ('sparsevlm', torch.tensor([63, 78, 85]))]:
            for keep in (.75, .5):
                result = select_attention_tokens(q[queries], k, queries, mask, spans, keep, method)
                scores = dense[:, queries][:, :, mask].mean((0, 1))
                expected = scores.argsort(descending=True, stable=True)[:int(48*keep)].sort().values
                self.assertTrue(torch.equal(result['visual_keep_indices'], expected))
                selected = result['sequence_keep_indices']
                self.assertEqual(int(mask[selected].sum()), int(48*keep))
                self.assertTrue(set((~mask).nonzero().flatten().tolist()) <= set(selected.tolist()))

    def test_sparse_prepare_and_explicit_uniform_case(self):
        mask = torch.zeros(20, dtype=torch.bool)
        mask[2:10] = True
        queries, diagnostics = sparse_queries(torch.zeros(1, 20, 8), mask)
        self.assertEqual(len(queries), 0)
        self.assertTrue(diagnostics['uniform_score_fallback'])
        result = select_attention_tokens(torch.zeros(0, 4, 8), torch.zeros(20, 2, 8),
            queries, mask, [(2, 10)], .5, 'sparsevlm', diagnostics)
        self.assertEqual(result['visual_keep_indices'].tolist(), [0, 1, 2, 3])

    def test_reject_nonfinite_and_wrong_method(self):
        mask = torch.tensor([False] + [True]*8 + [False])
        for method, q in [('fastv', torch.full((1, 4, 8), float('nan'))), ('dart', torch.ones(1, 4, 8))]:
            with self.assertRaises(ValueError):
                select_attention_tokens(q, torch.ones(10, 2, 8), torch.tensor([9]), mask, [(1, 9)], .5, method)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
