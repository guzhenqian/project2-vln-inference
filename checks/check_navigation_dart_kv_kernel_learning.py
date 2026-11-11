"""User-run LMDeploy FP16 CUDA KV/attention check with synthetic QKV probes.

Uses the real saved DART selection and original mRoPE coordinates, but no
model weights, model generation, services, or navigation. --check-only exits
before any LMDeploy CUDA backend import or torch.cuda call. GPU execution is
reserved for the user's explicit run without --check-only.
"""

import argparse
import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from navigation_dart_kv_layout import (
    SingleSequenceKVLayout, build_single_batch_metadata, compact_suffix_kv_length,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = 'logs/navigation_dart_selection_learning_20261108T030530234089Z_ac3243f3/report.json'
REPORT_SHA256 = '517632818dd33e686a1f201ee407f302ff0b9b73ce23de515bb9a77ac7cc3148'
DECODE_STEPS = 26
BLOCK_SIZE = 64
MAX_ABS_TOL = 0.01
RELATIVE_L2_TOL = 0.01
SENTINEL = -7.25


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inside(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Evidence path must remain inside the project')
    return path


def bound_json(relative, expected_hash):
    path = inside(relative)
    if digest(path) != expected_hash:
        raise ValueError('Evidence SHA256 mismatch: ' + relative)
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def preflight():
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    report = bound_json(REPORT, REPORT_SHA256)
    if report['status'] != 'complete' or report['checkpoint_quantized']:
        raise ValueError('Expected the successful unquantized DART reference')
    selection_path = inside(REPORT).parent / 'selection.pt'
    if digest(selection_path) != report['selection_sha256']:
        raise ValueError('Saved selection SHA256 mismatch')
    selection = torch.load(selection_path, map_location='cpu', weights_only=True)
    recipe = report['recipe']
    feature_report = bound_json(recipe['feature_report'], recipe['feature_report_sha256'])
    feature_path = inside(recipe['feature_report']).parent / 'features.pt'
    if (digest(feature_path) != feature_report['feature_sha256']
            or feature_report['feature_sha256'] != report['source_feature_sha256']):
        raise ValueError('Original feature bundle SHA256 mismatch')
    features = torch.load(feature_path, map_location='cpu', weights_only=True)
    config_dict = bound_json(recipe['model'] + '/config.json', recipe['model_config_sha256'])
    for section in (config_dict, config_dict['text_config'], config_dict['vision_config']):
        if section.get('quantization_config') is not None:
            raise ValueError('This diagnostic requires the unquantized configuration')
    config = Qwen3VLConfig(**config_dict)
    ids, grid = features['input_ids'], features['image_grid_thw']
    positions, delta = Qwen3VLModel.get_rope_index(
        SimpleNamespace(config=config), input_ids=ids, image_grid_thw=grid,
        attention_mask=torch.ones_like(ids))
    keep = selection['sequence_keep_indices']
    if (keep.dtype != torch.long or keep.ndim != 1 or keep.numel() < 1
            or int(keep[0]) < 0 or int(keep[-1]) >= ids.shape[1]
            or not bool((keep[1:] > keep[:-1]).all())):
        raise ValueError('Keep indices must be a sorted unique subset of the original sequence')
    if not torch.equal(selection['input_ids'], ids[:, keep]):
        raise ValueError('Selected token IDs do not match original indices')
    if not torch.equal(selection['mrope_position_ids'], positions[:, :, keep]):
        raise ValueError('Selected mRoPE coordinates do not match the original full sequence')
    if not torch.equal(selection['original_rope_delta'], delta):
        raise ValueError('Original mRoPE delta mismatch')
    image_mask = ids[0] == config.image_token_id
    retained_image_mask = selection['input_ids'][0] == config.image_token_id
    if not torch.equal(keep[~retained_image_mask], (~image_mask).nonzero().flatten()):
        raise ValueError('A text or special token was removed or reordered')
    original_length, retained_length = ids.shape[1], keep.numel()
    if (original_length != report['before_sequence_tokens']
            or retained_length != report['after_sequence_tokens']):
        raise ValueError('Sequence lengths disagree with the selection report')
    semantic_start = original_length + int(delta.item())
    if semantic_start != int(positions.max()) + 1:
        raise ValueError('Original semantic decode start is inconsistent')
    layout = SingleSequenceKVLayout(original_length, retained_length, semantic_start)
    if (original_length, retained_length, layout.pruned_count) != (701, 551, 150):
        raise ValueError('This lesson is frozen to the verified 701 -> 551 input')
    if (config.text_config.num_attention_heads, config.text_config.num_key_value_heads,
            config.text_config.head_dim) != (32, 8, 128):
        raise ValueError('Unexpected Qwen3-VL attention dimensions')
    if (original_length // BLOCK_SIZE == (original_length + DECODE_STEPS) // BLOCK_SIZE
            or retained_length // BLOCK_SIZE == (retained_length + DECODE_STEPS) // BLOCK_SIZE):
        raise ValueError('Decode probe count must cross a page boundary in both paths')
    if compact_suffix_kv_length(original_length + DECODE_STEPS, layout.pruned_count) != retained_length + DECODE_STEPS:
        raise ValueError('Compacted suffix length calculation failed')
    return report, selection, config, positions, layout


def make_cpu_qkv(config, positions, semantic_start):
    """FP16 random probes with real original mRoPE; no trained parameters."""
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, rotate_half

    generator = torch.Generator(device='cpu').manual_seed(7)
    original_length = positions.shape[-1]
    total = original_length + DECODE_STEPS
    query = torch.randn(total, 32, 128, generator=generator).to(torch.float16)
    key = torch.randn(total, 8, 128, generator=generator).to(torch.float16)
    value = torch.randn(total, 8, 128, generator=generator).to(torch.float16)
    decode_positions = torch.arange(semantic_start, semantic_start + DECODE_STEPS).view(1, 1, -1).expand(3, 1, -1)
    all_positions = torch.cat((positions, decode_positions), dim=-1)
    rotary = Qwen3VLTextRotaryEmbedding(config.text_config, device=torch.device('cpu'))
    cos, sin = rotary(torch.empty(1, total, 128, dtype=torch.float32), all_positions)
    cos, sin = cos[0, :, None, :], sin[0, :, None, :]
    # Materialize precisely defined FP16 inputs, then use these same inputs for
    # both the CUDA kernels and the independent CPU FP32 reference below.
    query = (query.float() * cos + rotate_half(query.float()) * sin).half().contiguous()
    key = (key.float() * cos + rotate_half(key.float()) * sin).half().contiguous()
    return query, key, value.contiguous(), all_positions


def dense_cpu_reference(query, key, value):
    """Independent causal/GQA attention, computed once for all probe positions."""
    import torch

    if any(t.device.type != 'cpu' for t in (query, key, value)):
        raise ValueError('The dense reference must remain on CPU')
    q = query.float().transpose(0, 1)
    k = key.float().repeat_interleave(4, dim=1).transpose(0, 1)
    v = value.float().repeat_interleave(4, dim=1).transpose(0, 1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    future = torch.ones(query.shape[0], key.shape[0], dtype=torch.bool).triu(diagonal=1)
    scores.masked_fill_(future[None], float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v).transpose(0, 1).contiguous()


def error_metrics(actual, expected):
    import torch

    actual = actual.detach().float().cpu()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError('Invalid CUDA attention result')
    difference = actual - expected
    maximum = float(difference.abs().max())
    relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected).clamp_min(1e-12))
    return {'max_abs': maximum, 'relative_l2': relative,
            'pass': maximum <= MAX_ABS_TOL and relative <= RELATIVE_L2_TOL}


def run_case(name, query, key, value, reference, prompt_length, prefill_chunks,
             allocated_blocks, impl, results):
    import torch

    print(f'Running {name}; the first CUDA call may compile Triton kernels...', flush=True)
    offsets_cpu = torch.arange(allocated_blocks - 1, -1, -1, dtype=torch.int64)
    offsets = offsets_cpu.to(device='cuda', dtype=torch.int32)[None]
    shape = (allocated_blocks, BLOCK_SIZE, 8, 128)
    k_cache = torch.full(shape, SENTINEL, dtype=torch.float16, device='cuda')
    v_cache = torch.full(shape, SENTINEL, dtype=torch.float16, device='cuda')
    result = {'name': name, 'prompt_tokens': prompt_length, 'prefill_chunks': prefill_chunks,
              'decode_probes': DECODE_STEPS, 'allocated_blocks': allocated_blocks,
              'physical_block_offsets': offsets_cpu.tolist(), 'calls': [],
              'cache_checks': 0, 'status': 'running'}
    results.append(result)
    if sum(prefill_chunks) != prompt_length:
        raise ValueError('Prefill chunks must cover the complete selected sequence')
    written = 0

    def one_call(count, decoding):
        nonlocal written
        old_written, end = written, written + count
        old_k = k_cache.detach().cpu().index_select(0, offsets_cpu).reshape(-1, 8, 128)[:old_written].clone()
        old_v = v_cache.detach().cpu().index_select(0, offsets_cpu).reshape(-1, 8, 128)[:old_written].clone()
        metadata = build_single_batch_metadata(count, end, offsets, is_decoding=decoding, block_size=BLOCK_SIZE)
        actual = impl.forward(query[written:end].to('cuda'), key[written:end].to('cuda'),
                              value[written:end].to('cuda'), k_cache, v_cache, metadata, inplace=False)
        metrics = error_metrics(actual, reference[written:end])
        call = {'phase': 'decode_probe' if decoding else 'prefill',
                'q_length': count, 'kv_length': end, 'history_length': old_written,
                **metrics}
        result['calls'].append(call)
        if not metrics['pass']:
            raise ValueError(f'{name} attention exceeds fixed tolerance at KV length {end}: {metrics}')
        logical_k = k_cache.detach().cpu().index_select(0, offsets_cpu).reshape(-1, 8, 128)
        logical_v = v_cache.detach().cpu().index_select(0, offsets_cpu).reshape(-1, 8, 128)
        checks = {
            'valid_key_rows_exact': torch.equal(logical_k[:end], key[:end]),
            'valid_value_rows_exact': torch.equal(logical_v[:end], value[:end]),
            'old_key_prefix_unchanged': torch.equal(logical_k[:old_written], old_k),
            'old_value_prefix_unchanged': torch.equal(logical_v[:old_written], old_v),
            'unwritten_key_tail_unchanged': bool((logical_k[end:] == SENTINEL).all()),
            'unwritten_value_tail_unchanged': bool((logical_v[end:] == SENTINEL).all()),
        }
        call['cache_checks'] = checks
        if not all(checks.values()):
            raise ValueError(f'{name} KV row/overwrite/tail verification failed: {checks}')
        result['cache_checks'] += 1
        written = end

    for chunk in prefill_chunks:
        one_call(chunk, False)
    for _ in range(DECODE_STEPS):
        one_call(1, True)
    result['final_kv_length'] = written
    result['max_abs'] = max(call['max_abs'] for call in result['calls'])
    result['max_relative_l2'] = max(call['relative_l2'] for call in result['calls'])
    result['status'] = 'complete'
    print(f'{name}: prefill={prompt_length}, decode={DECODE_STEPS}, final_kv={written}, '
          f'max_abs={result["max_abs"]:.6g}, relative_l2={result["max_relative_l2"]:.6g}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    report, selection, config, positions, layout = preflight()
    print('DART_KV_PREFLIGHT_OK', flush=True)
    print(f'Original={layout.original_prompt_length}; retained={layout.retained_prompt_length}; '
          f'removed={layout.pruned_count}; semantic_decode_start={layout.semantic_decode_start}', flush=True)
    print('Synthetic FP16 QKV probes only; no model weights, generated actions, or service calls', flush=True)
    if args.check_only:
        print('CHECK_ONLY: CPU evidence/position checks only; CUDA backend not imported, no CUDA calls or output writes', flush=True)
        return

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid.uuid4().hex[:8]
    output = ROOT / 'logs' / ('navigation_dart_kv_kernel_learning_' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    provenance = {
        'scope': 'Synthetic QKV CUDA attention/cache validation with real fixed selection and mRoPE; not model inference',
        'selection_report': REPORT, 'selection_report_sha256': REPORT_SHA256,
        'selection_sha256': report['selection_sha256'],
        'script_sha256': digest(Path(__file__)),
        'helper_sha256': digest(Path(__file__).with_name('navigation_dart_kv_layout.py')),
        'versions': {name: version(name) for name in ('torch', 'transformers', 'lmdeploy')},
        'seed': 7, 'q_heads': 32, 'kv_heads': 8, 'head_dim': 128,
        'dtype': 'float16', 'reference_dtype': 'float32', 'reference_device': 'cpu',
        'kv_quant_policy': 0, 'block_size': BLOCK_SIZE, 'decode_probes': DECODE_STEPS,
        'original_prompt_length': layout.original_prompt_length,
        'retained_prompt_length': layout.retained_prompt_length, 'pruned_count': layout.pruned_count,
        'semantic_decode_start': layout.semantic_decode_start,
        'original_rope_delta': layout.semantic_decode_start - layout.original_prompt_length,
        'compact_length_rope_delta_if_needed': layout.semantic_decode_start - layout.retained_prompt_length,
        'max_abs_tolerance': MAX_ABS_TOL, 'relative_l2_tolerance': RELATIVE_L2_TOL,
        'allocation_policy': 'Both paths allocate by original prompt plus probes and two guard blocks; no allocation savings claimed',
        'limitations': ['No trained model layers or language generation run',
                        'No scheduler, service, or CUDA graph integration tested',
                        'Split prefill uses already-fixed indices; it does not solve global-pivot DART selection across real input chunks',
                        'Single batch, eager CUDA, unquantized KV only; no holes in logical cache slots',
                        'No performance benchmark or navigation evaluation'],
    }
    save_json(output / 'input.json', provenance)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        free, total = torch.cuda.mem_get_info()
        if free < 512 * 1024 ** 2:
            raise RuntimeError('At least 512 MiB currently free GPU memory is required')
        provenance['gpu_name'] = torch.cuda.get_device_name()
        provenance['gpu_free_bytes_before'] = free
        provenance['gpu_total_bytes'] = total
        # This is deliberately after the --check-only return and free-memory guard.
        from lmdeploy.pytorch.backends.cuda.attention import TritonAttentionImpl
        impl = TritonAttentionImpl(32, 128, scale=1 / math.sqrt(128), num_kv_heads=8,
                                  v_head_size=128, causal=True)
        allocated_blocks = math.ceil((layout.original_prompt_length + DECODE_STEPS) / BLOCK_SIZE) + 2
        provenance['allocated_blocks_per_case'] = allocated_blocks
        with torch.inference_mode():
            query, key, value, all_positions = make_cpu_qkv(config, positions, layout.semantic_decode_start)
            identity = torch.arange(layout.original_prompt_length)
            suffix = torch.arange(layout.original_prompt_length, layout.original_prompt_length + DECODE_STEPS)
            full_indices = torch.cat((identity, suffix))
            compact_indices = torch.cat((selection['sequence_keep_indices'], suffix))
            if not torch.equal(all_positions[:, :, compact_indices[:layout.retained_prompt_length]],
                               selection['mrope_position_ids']):
                raise ValueError('Probe compaction changed retained original mRoPE coordinates')
            if not torch.equal(all_positions[:, :, full_indices], all_positions):
                raise ValueError('100% keep failed identity check')
            full_reference = dense_cpu_reference(query, key, value)
            run_case('keep_100_percent', query, key, value, full_reference,
                     layout.original_prompt_length, [layout.original_prompt_length], allocated_blocks, impl, results)
            compact_q, compact_k, compact_v = (tensor[compact_indices].contiguous() for tensor in (query, key, value))
            compact_reference = dense_cpu_reference(compact_q, compact_k, compact_v)
            run_case('keep_75_percent', compact_q, compact_k, compact_v, compact_reference,
                     layout.retained_prompt_length, [layout.retained_prompt_length], allocated_blocks, impl, results)
            run_case('keep_75_percent_fixed_indices_split_prefill', compact_q, compact_k, compact_v,
                     compact_reference, layout.retained_prompt_length,
                     [511, layout.retained_prompt_length - 511], allocated_blocks, impl, results)
        report_out = dict(provenance, status='complete', cases=results,
                          attention_calls=sum(len(case['calls']) for case in results),
                          checks={'original_mrope_reconstructed': True, 'retained_mrope_preserved': True,
                                  'semantic_decode_positions_preserved': True, 'keep_100_percent_identity': True},
                          elapsed_seconds=time.perf_counter() - started)
        save_json(output / 'report.json', report_out)
        print('Saved:', output.relative_to(ROOT).as_posix(), flush=True)
        print('DART_LMDEPLOY_KV_KERNEL_OK', flush=True)
        print('Cache/attention probes passed; actual model, service, graph and navigation integration remain pending.', flush=True)
    except Exception as error:
        save_json(output / 'failure.json', dict(provenance, status='failed', error_type=type(error).__name__,
                                              error=str(error), cases=results,
                                              elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
