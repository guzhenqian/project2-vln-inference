"""Deterministic, exact-budget, multi-image adaptation of layer-internal DART.

This selector consumes decoder states, NOT input image embeddings. The caller
supplies post-key-norm/post-mRoPE keys and hidden states after the model's final
norm at the chosen pruning boundary. It never runs a model or changes a cache.

Reference mechanism:
https://github.com/ZichenWen1/DART/blob/main/Qwen2_5-VL/Qwen2_5VL_DART/modeling_qwen2_5_vl_self.py

Explicit adaptations from the reference implementation:
* each image has a separate ceil(N * keep_ratio) budget;
* four shared text pivots come from the suffix after the LAST image span,
  including its vision_end token and the chat suffix;
* the budget includes visual pivots; divmod distributes its remainder across
  the ORIGINAL pivots, visual first, then text;
* pivots rank by descending key L1 norm; ties use the lower sequence index;
  cosine ties also use the lower sequence index, with no set iteration.

Newly selected tokens never become pivots. All non-image tokens are preserved.
This is not a verbatim reproduction or an embedding-level DART variant.
"""

import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F


ALGORITHM = 'dart_layer_internal_multi_image_exact_budget_deterministic_v1'


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _validate_inputs(keys, normalized_hidden, image_token_mask, image_spans):
    for name, value in (('keys', keys), ('normalized_hidden', normalized_hidden)):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f'{name} must be a CPU floating-point Tensor')
        if value.device.type != 'cpu' or not value.is_floating_point():
            raise ValueError(f'{name} must be a CPU floating-point Tensor')
        if value.ndim != 2 or min(value.shape) <= 0:
            raise ValueError(f'{name} must have nonempty shape [sequence, features]')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} contains NaN or infinity')
    sequence_length = keys.shape[0]
    if normalized_hidden.shape[0] != sequence_length:
        raise ValueError('keys and normalized_hidden sequence lengths differ')
    if (not isinstance(image_token_mask, torch.Tensor)
            or image_token_mask.device.type != 'cpu'
            or image_token_mask.dtype != torch.bool
            or image_token_mask.shape != (sequence_length,)):
        raise ValueError('image_token_mask must be a CPU BoolTensor [sequence]')
    if not isinstance(image_spans, (list, tuple)) or not image_spans:
        raise ValueError('image_spans must contain at least one (start, end) interval')
    spans = []
    coverage = torch.zeros(sequence_length, dtype=torch.bool)
    previous_end = 0
    for item in image_spans:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError('Each image span must be a (start, end) pair')
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in item):
            raise ValueError('Image span boundaries must be integers')
        start, end = map(int, item)
        if not 0 <= start < end <= sequence_length:
            raise ValueError('Image spans must be nonempty and inside the sequence')
        if start < previous_end:
            raise ValueError('Image spans must be ordered and must not overlap')
        if not bool(image_token_mask[start:end].all()):
            raise ValueError('Each image span must contain only contiguous image tokens')
        coverage[start:end] = True
        spans.append((start, end))
        previous_end = end
    if not torch.equal(coverage, image_token_mask):
        raise ValueError('Image spans must cover exactly the image_token_mask')
    return sequence_length, spans


def _ranked_indices(scores, indices, count, *, descending):
    """Indices arrive in sequence order; stable sort breaks ties by that order."""
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite selector score')
    return indices[torch.argsort(scores, descending=descending, stable=True)[:count]]


@torch.no_grad()
def select_dart_tokens(keys, normalized_hidden, image_token_mask, image_spans,
                       *, keep_ratio=0.75, visual_pivots=4, text_pivots=4):
    """Return CPU keep indices and JSON-compatible selection diagnostics.

    Args:
        keys: [S, Dk] keys after architecture-specific key norm and mRoPE.
        normalized_hidden: [S, D] decoder hidden states after final model norm.
        image_token_mask: BoolTensor [S], true only for actual image placeholders.
        image_spans: ordered half-open (start, end) intervals, one per image.
        keep_ratio: fraction to retain in (0, 1]; 1 returns identity.

    Returns:
        sequence_keep_indices: LongTensor indexing the complete input sequence.
        visual_keep_indices: LongTensor indexing only the original visual rows,
            suitable for selecting rows from image and DeepStack feature arrays.
        diagnostics: JSON-compatible information about every pivot and selection.

    All shape/span/finiteness validation applies at keep_ratio=1. Pivot capacity
    validation is skipped for identity because no pivot selection takes place.
    Scoring uses CPU float32 unless an input is float64, then CPU float64.
    """
    if (isinstance(keep_ratio, bool) or not isinstance(keep_ratio, Real)
            or not math.isfinite(float(keep_ratio)) or not 0 < keep_ratio <= 1):
        raise ValueError('keep_ratio must be finite and in (0, 1]')
    keep_ratio = float(keep_ratio)
    visual_pivots = _positive_integer(visual_pivots, 'visual_pivots')
    text_pivots = _positive_integer(text_pivots, 'text_pivots')
    sequence_length, spans = _validate_inputs(
        keys, normalized_hidden, image_token_mask, image_spans)
    visual_positions = image_token_mask.nonzero(as_tuple=False).flatten()
    selected_mask = ~image_token_mask.clone()
    images = []
    text_start = spans[-1][1]
    identity = keep_ratio == 1.0

    if identity:
        selected_mask[:] = True
        text_indices = torch.empty(0, dtype=torch.long)
    else:
        if sequence_length - text_start < text_pivots:
            raise ValueError('Suffix after the last image has too few text pivot candidates')
        for start, end in spans:
            length = end - start
            if length < visual_pivots:
                raise ValueError('An image has too few visual pivot candidates')
            if math.ceil(length * keep_ratio) < visual_pivots:
                raise ValueError('An image budget is smaller than the visual pivot count')
        score_dtype = (torch.float64 if torch.float64 in (keys.dtype, normalized_hidden.dtype)
                       else torch.float32)
        score_keys = keys.to(dtype=score_dtype)
        score_hidden = normalized_hidden.to(dtype=score_dtype)
        key_norms = score_keys.abs().sum(dim=-1)
        suffix_indices = torch.arange(text_start, sequence_length)
        text_indices = _ranked_indices(key_norms[suffix_indices], suffix_indices,
                                       text_pivots, descending=True)

    for image_index, (start, end) in enumerate(spans):
        positions = torch.arange(start, end)
        length = end - start
        target = math.ceil(length * keep_ratio)
        rounds = []
        if identity:
            image_pivot_indices = torch.empty(0, dtype=torch.long)
            pivot_order = torch.empty(0, dtype=torch.long)
            quotas = []
        else:
            image_pivot_indices = _ranked_indices(
                key_norms[positions], positions, visual_pivots, descending=True)
            selected_mask[image_pivot_indices] = True
            available = torch.ones(length, dtype=torch.bool)
            available[image_pivot_indices - start] = False
            pivot_order = torch.cat((image_pivot_indices, text_indices))
            base, remainder = divmod(target - visual_pivots, pivot_order.numel())
            quotas = [base + int(i < remainder) for i in range(pivot_order.numel())]
            for pivot, quota in zip(pivot_order.tolist(), quotas):
                candidates = positions[available]
                if quota > candidates.numel():
                    raise RuntimeError('Internal quota exceeds the remaining image candidates')
                if quota:
                    similarities = F.cosine_similarity(
                        score_hidden[pivot].unsqueeze(0), score_hidden[candidates], dim=-1)
                    chosen = _ranked_indices(similarities, candidates, quota, descending=False)
                    selected_mask[chosen] = True
                    available[chosen - start] = False
                else:
                    chosen = torch.empty(0, dtype=torch.long)
                rounds.append({'pivot_index': pivot, 'quota': quota,
                               'selected_indices': chosen.tolist()})
        kept = positions[selected_mask[start:end]]
        if kept.numel() != target:
            raise RuntimeError('Selector did not meet the exact per-image budget')
        images.append({
            'image_index': image_index,
            'span': [start, end],
            'original_count': length,
            'target_count': target,
            'kept_count': kept.numel(),
            'actual_keep_ratio': kept.numel() / length,
            'visual_pivot_indices': image_pivot_indices.tolist(),
            'pivot_order': pivot_order.tolist(),
            'quotas': quotas,
            'rounds': rounds,
            'kept_sequence_indices': kept.tolist(),
        })

    sequence_indices = selected_mask.nonzero(as_tuple=False).flatten()
    visual_row_indices = selected_mask[visual_positions].nonzero(as_tuple=False).flatten()
    return {
        'sequence_keep_indices': sequence_indices,
        'visual_keep_indices': visual_row_indices,
        'diagnostics': {
            'algorithm': ALGORITHM,
            'mode': 'identity' if identity else 'selection',
            'adaptations': ['per_image_budget', 'shared_suffix_text_pivots',
                            'exact_budget_including_visual_pivots', 'deterministic_order_and_ties'],
            'requested_keep_ratio': keep_ratio,
            'budget_rounding': 'ceil per image',
            'pivot_order_rule': 'visual then text; descending key L1, ties lower sequence index',
            'similarity_rule': 'lowest cosine of final-normalized hidden; ties lower sequence index',
            'text_pool': {'start': text_start, 'end': sequence_length,
                          'definition': 'all positions after last image span, including special tokens'},
            'text_pivot_indices': text_indices.tolist(),
            'sequence_length_before': sequence_length,
            'sequence_length_after': sequence_indices.numel(),
            'visual_tokens_before': visual_positions.numel(),
            'visual_tokens_after': visual_row_indices.numel(),
            'nonvisual_tokens_preserved': int((~image_token_mask).sum()),
            'images': images,
        },
    }
