"""Read-only audit requiring both native AWQ execution and DART token/KV traces.

Only the frozen 640x480 navigation protocol (300 merged tokens per image) is
supported. This is not a general audit for other image resolutions or grids.
No model imports, GPU calls, HTTP requests, or changes to original run evidence.
The service is serial (batch size one). Trace association uses a frozen before
inventory plus chronological input signatures, not a request-ID claim.

Independent adaptation of the frozen C audit SHA256
f040c21c622d73579190f4f4bbb9cd0908a08af1a077f2a85b8ce889846a0374.
Synthetic unit fixtures are not evidence of a completed GPU combination run.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from navigation_awq_dart_preflight import CONFIG_SHA, HEADER_SHA, WEIGHT_SHA
from navigation_awq_dart_profiles import MODEL_RELATIVE, served_model_name

ALGORITHM = 'dart_layer_internal_multi_image_exact_budget_deterministic_v1'
AWQ_SPEC = dict(quant_method='awq', version='gemm', bits=4,
                group_size=128, zero_point=True)
PROJECTIONS = {
    'self_attn.qkv_proj': ('QKVAwqLinear', 2560, 6144),
    'self_attn.o_proj': ('AwqLinear', 4096, 2560),
    'mlp.gate_up_proj': ('MergedAwqLinear', 2560, 19456),
    'mlp.down_proj': ('AwqLinear', 9728, 2560),
}


def expected_modules():
    return {f'language_model.layers.{layer}.{projection}':
            (layer, projection, module_class, in_features, out_features)
            for layer in range(36)
            for projection, (module_class, in_features, out_features) in PROJECTIONS.items()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_quantization(record, check):
    check(record.get('weight_dtype') == 'int4_packed+float16'
          and record.get('activation_dtype') == 'float16' and record.get('awq') is True
          and record.get('quantization_config') == AWQ_SPEC,
          'Expected native AWQ W4A16 packed weights, not a FP16 or label-only trace')


def audit_launch(service_dir, keep_ratio=.75):
    path = Path(service_dir) / 'launch.json'
    launch = read(path)
    require(launch.get('profile') == 'navigation' and launch.get('keep_ratio') == keep_ratio
            and launch.get('weight_quantization') == AWQ_SPEC,
            'Unexpected service profile/quantization')
    require(launch.get('model_name') == served_model_name(keep_ratio, 'navigation')
            and launch.get('model_source') == MODEL_RELATIVE,
            'Expected original B-group AWQ checkpoint and D-group service identity')
    require(launch.get('model_config_sha256') == CONFIG_SHA
            and launch.get('weight_header_sha256') == HEADER_SHA,
            'Original B-group AWQ config/header fingerprint mismatch')
    checkpoint = launch.get('checkpoint_verification', {})
    require(checkpoint.get('historical_full_weight_sha256') == WEIGHT_SHA
            and checkpoint.get('full_weight_sha256_rechecked_this_run') is True,
            'Original B-group full AWQ weight integrity was not rechecked for this launch')
    engine = launch['engine_config']
    require(engine['dtype'] == 'float16' and engine['quant_policy'] == 0 and
            engine['eager_mode'] is True and engine['max_batch_size'] == 1,
            'Expected serial eager W4A16 service with FP16 activation/KV')
    require(engine['num_gpu_blocks'] == 64 and engine['block_size'] == 64
            and engine['max_prefill_token_num'] == 3072 and engine['session_len'] == 4096
            and engine['enable_prefix_caching'] is False and engine['enable_microbatch'] is False,
            'Unexpected D-group full-prefill/cache configuration')
    return launch


def audit_runtime_init(service_dir, keep_ratio=.75):
    paths = list(Path(service_dir).glob('runtime_init_*.json'))
    require(len(paths) == 1, 'Expected exactly one AWQ runtime initialization record')
    path = paths[0]
    record = read(path)
    check = lambda condition, message: require(condition, path.name + ': ' + message)
    check(record.get('event') == 'runtime_initialized'
          and record.get('runtime_class') == 'AwqDartQwen3VLForConditionalGeneration',
          'Expected the independent AWQ+DART runtime class')
    check(record.get('profile') == 'navigation' and record.get('keep_ratio') == keep_ratio,
          'Runtime initialization profile/keep-ratio mismatch')
    audit_quantization(record, check)
    evidence = record.get('awq_modules', {})
    check(evidence.get('runtime_module_count') == 144
          and evidence.get('checkpoint_projection_count') == 252,
          'Expected 144 native fused AWQ modules covering 252 checkpoint projections')
    modules = evidence.get('modules', [])
    expected = expected_modules()
    check(len(modules) == 144 and {m.get('name') for m in modules} == set(expected),
          'Missing/duplicate/unexpected AWQ modules')
    for module in modules:
        layer, projection, class_name, in_features, out_features = expected[module['name']]
        check(module.get('layer_index') == layer and module.get('projection') == projection
              and module.get('class') == class_name
              and module.get('backend_class') == 'AwqLinearW4A16Impl',
              'AWQ module path/class mismatch')
        check(module.get('in_features') == in_features and module.get('out_features') == out_features,
              'AWQ module dimensions mismatch')
        check(module.get('qweight_shape') == [in_features, out_features // 8]
              and module.get('qzeros_shape') == [in_features // 128, out_features // 8]
              and module.get('scales_shape') == [in_features // 128, out_features],
              'AWQ packed tensor shape mismatch')
        check(module.get('qweight_dtype') == 'int32' and module.get('qzeros_dtype') == 'int32'
              and module.get('scales_dtype') == 'float16', 'AWQ packed tensor dtype mismatch')
    return dict(path=path.name, sha256=sha(path), module_count=144,
                checkpoint_projection_count=252)


def audit_awq_execution(record, original_rows, retained_rows, check):
    audit_quantization(record, check)
    execution = record.get('awq_execution', {})
    check(execution.get('verified') is True
          and execution.get('loaded_quantized_tensor_count') == 756
          and execution.get('module_count') == 144,
          'AWQ weights were not completely loaded or module execution was not verified')
    calls = execution.get('module_calls', [])
    expected = expected_modules()
    check(len(calls) == 144 and {m.get('name') for m in calls} == set(expected),
          'Missing/duplicate/unexpected executed AWQ modules')
    for call in calls:
        layer, projection, *_ = expected[call['name']]
        rows = original_rows if layer < 2 else retained_rows
        check(call.get('layer_index') == layer and call.get('projection') == projection
              and call.get('rows') == rows and call.get('calls') == 1,
              'AWQ module execution row counts differ from DART token/KV lengths')


def audit_trace(path, *, keep_ratio=.75):
    path = Path(path)
    record = read(path)
    trace = record.get('trace_id')
    require(path.name == 'prefill_' + str(trace) + '.json', 'Trace/file identity mismatch')
    prefix = path.name + ': '
    check = lambda condition, message: require(condition, prefix + message)
    check(record.get('event') == 'image_prefill', 'Not an image prefill')
    check(keep_ratio in (.75, 1.), 'Only keep75 and keep100 controls are supported')
    mode = ALGORITHM if keep_ratio == .75 else 'native_control'
    check(record.get('actual_mode') == mode, 'Unexpected runtime algorithm')
    check(record.get('profile') == 'navigation' and record.get('keep_ratio') == keep_ratio,
          'Expected navigation with the selected keep ratio')
    check(record.get('prune_before_layer_index') == 2 and record.get('complete_prefill') is True,
          'Expected complete prefill, pruning before zero-based layer 2')
    audit_quantization(record, check)
    check(record.get('kv_quant_policy') == 0, 'Expected unquantized FP16 KV cache')
    audit_launch(path.parent, keep_ratio)
    initialization = audit_runtime_init(path.parent, keep_ratio)
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
    check(retained_counts == [math.ceil(n * keep_ratio) for n in image_counts], 'Keep-ratio budget mismatch')
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
    check(pool['allocated_blocks_per_layer_min'] == 64 and pool['block_size'] == 64
          and pool['reserved_gpu_blocks'] == 1, 'Expected fixed 64x64 pool with one reserved block')
    check(pool['usable_token_capacity'] ==
          (pool['allocated_blocks_per_layer_min'] - pool['reserved_gpu_blocks']) * pool['block_size'],
          'Cache pool capacity arithmetic mismatch')
    check(pool['usable_token_capacity'] >= pool['required_token_capacity'] >= 3584,
          'Insufficient cache pool capacity')
    selection = record['selection']
    if keep_ratio == .75:
        check(selection['algorithm'] == ALGORITHM and
              selection['sequence_length_before'] == original and
              selection['sequence_length_after'] == retained and
              selection['visual_tokens_before'] == sum(image_counts) and
              selection['visual_tokens_after'] == sum(retained_counts), 'Selector/runtime mismatch')
    else:
        check(selection is None and kept == list(range(original)) and removed == 0,
              'Keep100 must be a native AWQ control with identity token indices')
    audit_awq_execution(record, original, retained, check)
    decode_path = path.with_name('first_decode_' + trace + '.json')
    decode = read(decode_path)
    check(decode.get('event') == 'first_decode' and decode.get('trace_id') == trace,
          'Decode trace mismatch')
    audit_awq_execution(decode, 1, 1, check)
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
                keep_ratio=keep_ratio, combined_awq_dart=keep_ratio == .75,
                awq_runtime=initialization, native_awq_module_count=144,
                launch_sha256=sha(path.parent / 'launch.json'),
                prefill_sha256=sha(path), first_decode_sha256=sha(decode_path))


def audit_batch(run_dir, service_dir, before_names=None):
    run_dir, service_dir = Path(run_dir), Path(service_dir)
    manifest = read(run_dir / 'manifest.json')
    summary = read(run_dir / 'summary.json')
    provenance = read(run_dir / 'provenance.json')
    launch = audit_launch(service_dir)
    require(summary['status'] == 'complete' and summary['error_count'] == 0,
            'Runtime acceptance requires a complete zero-error batch')
    require(manifest['model'] == summary['model'] == launch['model_name'], 'Service/run model mismatch')
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
                             'AWQ execution evidence records loaded tensors and module calls; it is not a kernel numerical equivalence test.',
                             'Effective upper-layer token/KV lengths are reduced; allocated blocks are unchanged.',
                             'This validates runtime integration, not navigation success or a matched speedup.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', type=Path)
    mode.add_argument('--trace', type=Path)
    parser.add_argument('--keep-ratio', type=float, choices=(.75, 1.), default=.75,
                        help='keep100 is only an AWQ native control, not a combined result')
    parser.add_argument('--service-dir', type=Path)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.trace:
        if args.service_dir or args.before:
            parser.error('--trace discovers its runtime initialization in the same directory')
        result = dict(status='passed', passed=True,
                      trace=audit_trace(args.trace, keep_ratio=args.keep_ratio))
    else:
        if not args.service_dir or args.keep_ratio != .75:
            parser.error('--run requires --service-dir and the D-group keep75 mode')
        result = audit_batch(args.run, args.service_dir, read(args.before) if args.before else None)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in
                      ('status', 'actual_model_calls', 'warmup_requests', 'runtime_trace_count', 'pruning_counts', 'trace')
                      if key in result},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
