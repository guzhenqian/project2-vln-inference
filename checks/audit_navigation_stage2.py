"""Read-only stage2 runtime audit; includes real native INT4 execution receipts."""
from pathlib import Path
from navigation_stage2_common import ROOT, GROUPS, SOURCES, read, digest, model_name


def require(value, message):
    if not value:
        raise ValueError(message)


def audit_service(service, group):
    launch = read(service / 'launch.json')
    recipe = GROUPS[group]
    require(launch['group'] == group and launch['recipe'] == recipe and launch['model_name'] == model_name(group), 'Service identity mismatch')
    engine = launch['engine_config']
    for key, value in dict(dtype='float16', eager_mode=True, max_prefill_token_num=3072,
        max_batch_size=1, session_len=4096, num_gpu_blocks=64, block_size=64,
        quant_policy=0, enable_prefix_caching=False).items():
        require(engine[key] == value, 'Engine mismatch: ' + key)
    require(set(launch['sources_sha256']) == set(SOURCES), 'Unpinned runtime dependency')
    for name, expected in launch['sources_sha256'].items():
        require(digest(ROOT / 'checks' / name) == expected, 'Runtime changed: ' + name)
    init = list(service.glob('runtime_init_*.json'))
    require(len(init) == 1, 'Expected one runtime initialization')
    record = read(init[0])
    require(record['algorithm'] == recipe['method'] and record['keep_ratio'] == recipe['keep'], 'Runtime recipe mismatch')
    if recipe['method'] in ('awq', 'gptq'):
        require(record['awq_modules']['runtime_module_count'] == 144, 'Missing native INT4 modules')
    return launch


def audit_trace(path, group):
    row = read(path)
    recipe = GROUPS[group]
    original, retained = row['original_length'], row['retained_length']
    kept, spans = row['kept_original_positions'], row['image_spans']
    require(2 <= row['image_count'] == len(spans) <= 9 and original <= 3072, 'Input limit violation')
    require(row['algorithm'] == recipe['method'] and row['keep_ratio'] == recipe['keep'], 'Algorithm mismatch')
    expected_mode = 'native_control' if recipe['keep'] == 1 else recipe['method'] + '_single_layer_no_merge'
    require(row['actual_mode'] == expected_mode and row['complete_prefill'] is True, 'Wrong runtime path')
    require(kept == sorted(set(kept)) and len(kept) == retained and kept[-1] == original-1, 'Invalid retained indices')
    require(all(isinstance(i, int) and 0 <= i < original for i in kept), 'Out-of-range indices')
    visual = set()
    last = 0
    for start, end in spans:
        require(last <= start < end <= original and end-start == 300, 'Incorrect image spans')
        visual.update(range(start, end))
        last = end
    require(set(range(original))-visual <= set(kept), 'Text or special tokens removed')
    per_image = [len(set(range(start, end)) & set(kept)) for start, end in spans]
    budget = int(300 * len(spans) * recipe['keep'])
    require(sum(per_image) == budget == row['retained_image_tokens'], 'Visual budget mismatch')
    require(per_image == row['retained_image_tokens_per_image'], 'Per-image counters mismatch')
    require(original-retained == row['removed'] == len(visual)-budget, 'Removed count mismatch')
    require(row['scheduler_length'] == row['lower_layer_kv_length'] == original and row['upper_layer_kv_length'] == retained, 'Prefill cache mismatch')
    require(row['cache_allocation_reduced'] is False and row['kv_quant_policy'] == 0, 'Unexpected cache behavior')
    require(row['cache_pool']['usable_token_capacity'] >= 3584, 'Insufficient KV capacity')
    if recipe['keep'] < 1:
        require(row['prune_before_layer_index'] == recipe['prune_layer'], 'Wrong prune layer')
        selection = row['selection']
        require(selection['method'] == recipe['method'] and selection['retained_original_tokens'] == budget
            and selection['merged_tokens'] == 0 and selection['variant'] == 'single_layer_no_merge', 'Selector disagreement')
    if recipe['method'] in ('awq', 'gptq'):
        receipt = row['awq_execution']
        require(receipt['verified'] and receipt['module_count'] == 144
            and receipt['loaded_quantized_tensor_count'] == 756, 'INT4 native execution missing')
        require(all(item['rows'] == original for item in receipt['module_calls']), 'INT4 token count mismatch')
    decode_path = path.with_name('first_decode_' + row['trace_id'] + '.json')
    decode = read(decode_path)
    require(decode['query_length'] == 1 and decode['lower_layer_kv_length'] == original+1
        and decode['upper_layer_kv_length'] == retained+1, 'Decode KV mismatch')
    delta = -280 * len(spans)
    require(row['mrope_delta'] == decode['mrope_delta'] == delta, 'mRoPE delta mismatch')
    require(decode['mrope_positions'] == [[original+delta]]*3 and decode['scheduler_positions'] == [[original]]
        and decode['original_positions_preserved'], 'Semantic coordinates changed')
    return dict(trace_id=row['trace_id'], image_count=len(spans), original_length=original,
        retained_length=retained, prefill_sha256=digest(path), first_decode_sha256=digest(decode_path))


def audit_batch(folder, service, group, before):
    audit_service(service, group)
    summary, manifest = read(folder / 'summary.json'), read(folder / 'manifest.json')
    require(summary['status'] == 'complete' and summary['error_count'] == 0, 'Incomplete or erroneous batch')
    require(summary['model'] == manifest['model'] == model_name(group), 'Model identity changed')
    require(summary['warmup_requests_excluded'] == 2, 'Warmups changed')
    expected = dict(zip(map(str, manifest['episodes']), map(str, manifest['trajectory_ids'])))
    require({str(row['episode_id']): str(row['trajectory_id']) for row in summary['episodes']} == expected
        and len(summary['episodes']) == len(expected), 'Case membership mismatch')
    calls, evidence = [], {}
    for result in summary['episodes']:
        path = folder / ('episode_' + str(result['episode_id'])) / 'steps.jsonl'
        rows = [__import__('json').loads(line) for line in path.read_text().splitlines()]
        require(len(rows) == result['steps'] and all(row['step'] == i and not row.get('error') for i, row in enumerate(rows)), 'Step record mismatch')
        require(all(type(row.get('model_called')) is bool for row in rows), 'Missing real-request flags')
        calls.extend(row for row in rows if row['model_called'])
        evidence[str(path.relative_to(folder))] = digest(path)
    fresh = [path for path in service.glob('prefill_*.json') if path.name not in set(before)]
    fresh.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    require(len(fresh) == len(calls)+2, 'Trace count differs from real calls + warmups')
    traces = [audit_trace(path, group) for path in fresh]
    signatures = [(calls[0]['image_count'], calls[0]['usage']['prompt_tokens'])]*2
    signatures += [(row['image_count'], row['usage']['prompt_tokens']) for row in calls]
    require([(row['image_count'], row['original_length']) for row in traces] == signatures, 'Request/trace association mismatch')
    return dict(passed=True, group=group, episodes=len(expected), actual_calls=len(calls),
        warmups=2, traces=traces, step_files_sha256=evidence,
        summary_sha256=digest(folder/'summary.json'), launch_sha256=digest(service/'launch.json'))
