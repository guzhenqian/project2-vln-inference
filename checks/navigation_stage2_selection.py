"""Pinned PriorTR FastV / SparseVLM-v1 scoring adapted to disjoint image spans.

Single pruning stage, no token merge. Scores/softmax use CPU FP32, explicitly
including transfer and scoring cost in request latency, as for existing DART.
"""
import math
import time
import torch

REFERENCE_COMMIT = 'e442a4c14af1b333067a6b8310c9111cad922033'


def sparse_queries(embeddings, image_mask):
    """PriorTR prepare(): above-average visual similarity of suffix text tokens."""
    mask = image_mask.cpu().bool()
    positions = mask.nonzero().flatten()
    if not len(positions):
        raise ValueError('SparseVLM requires image tokens')
    end = int(positions[-1]) + 1
    hidden = embeddings.reshape(mask.numel(), -1).detach().cpu().float()
    visual, text = hidden[mask], hidden[end:]
    if not len(text):
        raise ValueError('SparseVLM requires suffix text tokens')
    similarity = (visual @ text.t()).softmax(dim=-1).mean(dim=0)
    chosen = (similarity > similarity.mean()).nonzero().flatten() + end
    return chosen, {'suffix_start': end, 'suffix_count': len(text),
                    'important_text_count': len(chosen),
                    'important_text_original_positions': chosen.tolist(),
                    'uniform_score_fallback': len(chosen) == 0}


def select_attention_tokens(query, key, query_positions, image_mask, spans, keep_ratio,
                            method, prepare_diagnostics=None):
    started = time.perf_counter()
    if method not in ('fastv', 'sparsevlm') or keep_ratio not in (0.5, 0.75):
        raise ValueError('Unsupported stage2 token recipe')
    mask = image_mask.detach().cpu().bool()
    image_positions = mask.nonzero().flatten()
    if image_positions.numel() != sum(end-start for start, end in spans):
        raise ValueError('Mask and spans disagree')
    query, key = query.detach().cpu().float(), key.detach().cpu().float()
    query_positions = query_positions.cpu().long()
    if key.ndim != 3 or query.ndim != 3 or len(key) != len(mask):
        raise ValueError('Expected [tokens, heads, head_dim] rotated Q/K')
    if query.shape[1] % key.shape[1] or query.shape[-1] != key.shape[-1]:
        raise ValueError('Invalid grouped-query dimensions')
    if len(query) != len(query_positions):
        raise ValueError('Query selection mismatch')
    if not torch.isfinite(query).all() or not torch.isfinite(key).all():
        raise ValueError('Nonfinite Q/K')
    if len(query) == 0:
        if method != 'sparsevlm':
            raise ValueError('FastV must have a last-token query')
        scores = torch.ones(len(image_positions))
    else:
        repeated_key = key.repeat_interleave(query.shape[1] // key.shape[1], dim=1)
        logits = torch.einsum('qhd,khd->hqk', query, repeated_key) / math.sqrt(query.shape[-1])
        causal = torch.arange(len(mask))[None, :] > query_positions[:, None]
        logits.masked_fill_(causal.unsqueeze(0), float('-inf'))
        probabilities = logits.softmax(dim=-1)
        scores = probabilities[:, :, image_positions].mean(dim=(0, 1))
    budget = int(len(image_positions) * keep_ratio)
    if not 0 < budget < len(image_positions) or not torch.isfinite(scores).all():
        raise ValueError('Invalid selection budget or scores')
    # Deterministic original-position tie break; preserve source order after top-k.
    selected = torch.argsort(scores, descending=True, stable=True)[:budget].sort().values
    keep_mask = ~mask
    keep_mask[image_positions[selected]] = True
    kept = keep_mask.nonzero().flatten()
    if int(kept[-1]) != len(mask)-1 or int(mask[kept].sum()) != budget:
        raise ValueError('Text preservation or visual budget failed')
    return dict(sequence_keep_indices=kept, visual_keep_indices=selected,
        diagnostics=dict(method=method, reference_commit=REFERENCE_COMMIT,
            variant='single_layer_no_merge', scoring_device='cpu', scoring_dtype='float32',
            query_count=len(query_positions), query_original_positions=query_positions.tolist(),
            head_aggregation='mean', allocation='global_topk_across_disjoint_image_spans',
            original_visual_tokens=len(image_positions), retained_original_tokens=budget,
            merged_tokens=0, score_min=float(scores.min()), score_max=float(scores.max()),
            prepare=prepare_diagnostics, selection_including_transfer_s=time.perf_counter()-started))
