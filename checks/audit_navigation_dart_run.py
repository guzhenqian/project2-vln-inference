"""Read-only audit of saved DART navigation calls and runtime token/KV traces.

Only the frozen 640x480 navigation protocol (300 merged tokens per image) is
supported. This is not a general audit for other image resolutions or grids.
No model imports, GPU calls, HTTP requests, or changes to original run evidence.
The service is serial (batch size one). Trace association uses a frozen before
inventory plus chronological input signatures, not a request-ID claim.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


ALGORITHM = 'dart_layer_internal_multi_image_exact_budget_deterministic_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_trace(path):
    path = Path(path)
    record = read(path)
    trace = record.get('trace_id')
    require(path.name == 'prefill_' + str(trace) + '.json', 'Trace/file identity mismatch')
    prefix = path.name + ': '
    check = lambda condition, message: require(condition, prefix + message)
    check(record.get('event') == 'image_prefill', 'Not an image prefill')
    check(record.get('actual_mode') == ALGORITHM, 'Unexpected runtime algorithm')
    check(record.get('profile') == 'navigation' and record.get('keep_ratio') == .75,
          'Expected navigation keep75')
    check(record.get('prune_before_layer_index') == 2 and record.get('complete_prefill') is True,
          'Expected complete prefill, pruning before zero-based layer 2')
    check(record.get('weight_dtype') == 'float16' and record.get('awq') is False
          and record.get('kv_quant_policy') == 0, 'Expected FP16, no AWQ or KV quantization')
    original, retained = record['original_length'], record['retained_length']
    spans, kept = record['image_spans'], record['kept_original_positions']
    check(2 <= len(spans) == record['image_count'] <= 9, 'Invalid image count')
    check(0 < retained <= original <= 3072, 'Invalid sequence lengths')
    check(kept == sorted(set(kept)) and len(kept) == retained,
          'Retained indices must be unique, sorted, and match retained length')
    check(all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < original for i in kept),
          'Retained index out of bounds')
    check(kept[-1] == original - 1, 'Final prompt token not retained')
    image_positions = set()
    image_counts, retained_counts = [], []
    keep_set = set(kept)
    previous_end = 0
    for start, end in spans:
        check(previous_end <= start < end <= original, 'Overlapping/invalid image spans')
        positions = set(range(start, end))
        image_positions.update(positions)
        image_counts.append(end - start)
        retained_counts.append(len(positions & keep_set))
        previous_end = end
    check(image_counts == record['image_tokens_per_image'], 'Image span/count mismatch')
    check(image_counts == [300] * len(spans), 'Expected frozen 640x480/300-token image grids')
    check(retained_counts == record['retained_image_tokens_per_image'], 'Retained per-image mismatch')
    check(retained_counts == [math.ceil(n * .75) for n in image_counts], 'Keep75 budget mismatch')
    check(set(range(original)) - image_positions <= keep_set, 'Text/special tokens removed')
    check(sum(image_counts) == record['original_image_tokens'], 'Original image count mismatch')
    check(sum(retained_counts) == record['retained_image_tokens'], 'Retained image count mismatch')
    removed = original - retained
    check(removed == record['removed'] == sum(image_counts) - sum(retained_counts), 'Removed count mismatch')
    check(record['scheduler_length'] == record['lower_layer_kv_length'] == original,
          'Lower-layer/scheduler length mismatch')
    check(record['upper_layer_kv_length'] == retained, 'Upper-layer KV length mismatch')
    check(record.get('cache_allocation_reduced') is False, 'Unexpected allocator claim')
    pool = record['cache_pool']
    check(pool['usable_token_capacity'] ==
          (pool['allocated_blocks_per_layer_min'] - pool['reserved_gpu_blocks']) * pool['block_size'],
          'Cache pool capacity arithmetic mismatch')
    check(pool['usable_token_capacity'] >= pool['required_token_capacity'] >= 3584,
          'Insufficient cache pool capacity')
    selection = record['selection']
    check(selection['algorithm'] == ALGORITHM and
          selection['sequence_length_before'] == original and
          selection['sequence_length_after'] == retained and
          selection['visual_tokens_before'] == sum(image_counts) and
          selection['visual_tokens_after'] == sum(retained_counts), 'Selector/runtime mismatch')
    decode_path = path.with_name('first_decode_' + trace + '.json')
    decode = read(decode_path)
    check(decode.get('event') == 'first_decode' and decode.get('trace_id') == trace,
          'Decode trace mismatch')
    check(decode['query_length'] == 1 and decode['lower_layer_kv_length'] == original + 1
          and decode['upper_layer_kv_length'] == retained + 1, 'Decode KV lengths mismatch')
    check(decode['scheduler_positions'] == [[original]], 'Decode scheduler position mismatch')
    delta = record['mrope_delta']
    # Frozen processor grid [1,30,40], merge2 -> [1,15,20]: 300 image tokens
    # advance the semantic coordinate by20, hence -280 per image. This was
    # independently checked by the saved CPU position reference; do not extend
    # this formula to arbitrary resolutions merely by changing token counts.
    check(delta == -280 * len(spans), 'mRoPE delta disagrees with frozen image grid')
    semantic = original + delta
    check(record['semantic_decode_start'] == semantic and decode['mrope_delta'] == delta,
          'Semantic position/delta mismatch')
    check(decode['mrope_positions'] == [[semantic], [semantic], [semantic]] and
          decode.get('original_positions_preserved') is True, 'Decode mRoPE mismatch')
    return dict(trace_id=trace, image_count=len(spans), original_length=original,
                retained_length=retained, original_image_tokens=sum(image_counts),
                retained_image_tokens=sum(retained_counts), removed=removed,
                prefill_sha256=sha(path), first_decode_sha256=sha(decode_path))


def audit_batch(run_dir, service_dir, before_names=None):
    run_dir, service_dir = Path(run_dir), Path(service_dir)
    manifest = read(run_dir / 'manifest.json')
    summary = read(run_dir / 'summary.json')
    provenance = read(run_dir / 'provenance.json')
    launch = read(service_dir / 'launch.json')
    require(summary['status'] == 'complete' and summary['error_count'] == 0,
            'Runtime acceptance requires a complete zero-error batch')
    require(manifest['model'] == summary['model'] == launch['model_name'], 'Service/run model mismatch')
    require(launch['profile'] == 'navigation' and launch['keep_ratio'] == .75 and
            launch['weight_quantization'] is None, 'Unexpected service profile/quantization')
    engine = launch['engine_config']
    require(engine['dtype'] == 'float16' and engine['quant_policy'] == 0 and
            engine['eager_mode'] is True and engine['max_batch_size'] == 1,
            'Expected serial eager FP16 service')
    expected_ids = [str(eid) for eid in manifest['episodes']]
    actual_ids = [e['episode_id'] for e in summary['episodes']]
    require(len(expected_ids) == len(set(expected_ids)) and set(actual_ids) == set(expected_ids)
            and len(actual_ids) == len(expected_ids), 'Episode membership mismatch')
    calls, evidence = [], {}
    for episode in summary['episodes']:
        steps_path = run_dir / ('episode_' + episode['episode_id']) / 'steps.jsonl'
        rows = [json.loads(line) for line in steps_path.read_text(encoding='utf-8').splitlines()]
        require(len(rows) == episode['steps'] and all(r['step'] == i for i, r in enumerate(rows)),
                'Incomplete/nonsequential step records')
        require(all(not r.get('error') for r in rows), 'Step errors present')
        require(all(type(r.get('model_called')) is bool for r in rows), 'Missing model_called evidence')
        calls.extend(r for r in rows if r['model_called'])
        evidence[str(steps_path.relative_to(run_dir))] = sha(steps_path)
    require(calls and all(r['finish_reason'] == 'stop' and r['usage'] for r in calls),
            'Missing/abnormal HTTP completion evidence')
    require(summary['warmup_requests_excluded'] == 2, 'Expected two excluded warmups')
    files = list(service_dir.glob('prefill_*.json'))
    if before_names is not None:
        before_names = set(before_names)
        require(before_names <= {p.name for p in files}, 'Previously recorded service traces disappeared')
        files = [p for p in files if p.name not in before_names]
        method = 'frozen prefill inventory difference; chronological image/token signatures'
    else:
        start = datetime.fromisoformat(provenance['started_utc'].replace('Z', '+00:00')).timestamp()
        end = (run_dir / 'summary.json').stat().st_mtime
        files = [p for p in files if start <= p.stat().st_mtime <= end]
        method = 'saved run start/summary mtime window; chronological image/token signatures'
    files.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    require(len(files) == len(calls) + 2, 'Trace count does not equal actual HTTP calls + 2 warmups')
    traces = [audit_trace(path) for path in files]
    expected_signatures = [(calls[0]['image_count'], calls[0]['usage']['prompt_tokens'])] * 2
    expected_signatures.extend((r['image_count'], r['usage']['prompt_tokens']) for r in calls)
    require([(r['image_count'], r['original_length']) for r in traces] == expected_signatures,
            'Chronological runtime input signatures do not match warmups + navigation HTTP calls')
    counts = Counter((r['image_count'], r['original_image_tokens'], r['retained_image_tokens'],
                      r['original_length'], r['retained_length']) for r in traces)
    pruning = [dict(image_count=k[0], original_image_tokens=k[1], retained_image_tokens=k[2],
                    original_length=k[3], retained_length=k[4], requests=v)
               for k, v in sorted(counts.items())]
    return dict(status='passed', passed=True, run=str(run_dir), service_directory=str(service_dir),
                association_method=method, actual_model_calls=len(calls), warmup_requests=2,
                runtime_trace_count=len(traces), pruning_counts=pruning, traces=traces,
                evidence_sha256=dict(evidence, manifest=sha(run_dir / 'manifest.json'),
                                     summary=sha(run_dir / 'summary.json'),
                                     launch=sha(service_dir / 'launch.json')),
                limitations=['Serial request association by input signatures, not explicit HTTP request IDs.',
                             'Effective upper-layer token/KV lengths are reduced; allocated blocks are unchanged.',
                             'This validates runtime integration, not navigation success or a matched speedup.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--service-dir', type=Path, required=True)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_batch(args.run, args.service_dir, read(args.before) if args.before else None)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in
                      ('status', 'actual_model_calls', 'warmup_requests', 'runtime_trace_count', 'pruning_counts')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
