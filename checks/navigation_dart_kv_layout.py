"""Single-sequence, unquantized, packed KV layout for a DART runtime adapter.

Logical attention rows must be contiguous. Arbitrary holes in the original
token slots are not supported by LMDeploy's length-based attention kernels.
Scheduler allocation may still use the original length; this helper does not
change allocation, RoPE coordinates, model state, or the scheduler.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SingleSequenceKVLayout:
    original_prompt_length: int
    retained_prompt_length: int
    semantic_decode_start: int

    def __post_init__(self):
        if not 0 < self.retained_prompt_length <= self.original_prompt_length:
            raise ValueError('Expected 0 < retained length <= original length')
        if self.semantic_decode_start < 0:
            raise ValueError('Semantic decode position must be non-negative')

    @property
    def pruned_count(self):
        return self.original_prompt_length - self.retained_prompt_length

    def lengths_after_appends(self, appended):
        if not isinstance(appended, int) or appended < 0:
            raise ValueError('Appended token count must be a non-negative integer')
        return (self.original_prompt_length + appended,
                self.retained_prompt_length + appended)

    def semantic_decode_position(self, step_zero_based):
        if not isinstance(step_zero_based, int) or step_zero_based < 0:
            raise ValueError('Decode probe index must be a non-negative integer')
        return self.semantic_decode_start + step_zero_based


def compact_suffix_kv_length(original_kv_length, pruned_count):
    """Length for layers after pruning, while scheduler uses original lengths."""
    if pruned_count < 0 or original_kv_length <= pruned_count:
        raise ValueError('Invalid original length or pruned count')
    return original_kv_length - pruned_count


def build_single_batch_metadata(q_length, kv_length, block_offsets, *,
                                is_decoding, block_size=64, quant_policy=0,
                                metadata_type=None):
    """Build fresh metadata; do not mutate the lower layers' original metadata.

    An injected metadata_type (e.g. SimpleNamespace) permits CPU-only tests.
    The actual LMDeploy CUDA backend is imported only when this function is
    called without such an injected type, never at module import time.
    """
    import torch

    if quant_policy != 0:
        raise ValueError('Only unquantized FP16 KV caches are supported')
    if block_offsets.ndim != 2 or block_offsets.shape[0] != 1:
        raise ValueError('Only one sequence per batch is supported')
    if block_offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError('Block offsets must use integer indices')
    if block_size < 16 or block_size & (block_size - 1):
        raise ValueError('Block size must be a power of two and at least 16')
    if not isinstance(q_length, int) or not isinstance(kv_length, int):
        raise ValueError('Query and KV lengths must be Python integers')
    if not 0 < q_length <= kv_length <= block_offsets.shape[1] * block_size:
        raise ValueError('Query/KV length exceeds the contiguous cache capacity')
    if is_decoding and q_length != 1:
        raise ValueError('This helper supports only one decode token at a time')
    if metadata_type is None:
        from lmdeploy.pytorch.backends.cuda.attention import TritonAttentionMetadata
        metadata_type = TritonAttentionMetadata
    device = block_offsets.device
    q_start = torch.zeros(1, dtype=torch.int64, device=device)
    q_lens = torch.tensor([q_length], dtype=torch.int64, device=device)
    kv_lens = torch.tensor([kv_length], dtype=torch.int64, device=device)
    return metadata_type(
        is_decoding=bool(is_decoding), block_offsets=block_offsets,
        q_start_loc=q_start, q_seqlens=q_lens, kv_seqlens=kv_lens,
        kv_start_loc=None if is_decoding else q_start.clone(),
        kv_flatten_size=None if is_decoding else kv_length,
        fill_seqlens=None, quant_policy=0,
        cu_seqlens_q=torch.tensor([0, q_length], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, kv_length], dtype=torch.int32, device=device),
        max_kv_seqlen=kv_length,
    )
