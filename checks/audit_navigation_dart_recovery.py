"""Read-only runtime audit of a resumed DART batch with an unchanged prefix.

Old and new service trace inventories are deliberately kept separate: reused
model responses have old runtime evidence, while recovery has two new warmups.
No inference, GPU calls, simulator imports, or mutation of saved run evidence.
"""
from collections import Counter
from pathlib import Path

from audit_navigation_dart_run import audit_trace, read, require, sha


def rows_at(path):
    import json
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def signature(row):
    require(row.get('model_called') is True and row.get('finish_reason') == 'stop'
            and row.get('usage'), 'Incomplete/abnormal model completion evidence')
    return row['image_count'], row['usage']['prompt_tokens']


def audit_segment(service_dir, names, calls, warmup_signature):
    files = [service_dir / name for name in names]
    require(all(p.parent == service_dir and p.name.startswith('prefill_')
                and p.suffix == '.json' for p in files), 'Invalid trace inventory names')
    files.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    require(len(files) == len(calls) + 2, 'Segment trace count != actual calls + two warmups')
    traces = [audit_trace(path) for path in files]
    expected = [warmup_signature] * 2 + [signature(row) for row in calls]
    require([(row['image_count'], row['original_length']) for row in traces] == expected,
            'Segment chronological input signatures do not match warmups + actual calls')
    return traces


def audit_resumed_batch(new_batch_dir, old_batch_dir, service_dir, new_before_names):
    run, old, service = map(Path, (new_batch_dir, old_batch_dir, service_dir))
    manifest = read(run / 'manifest.json')
    summary = read(run / 'summary.json')
    launch = read(service / 'launch.json')
    require(manifest == read(old / 'manifest.json'), 'Recovery changed the frozen batch manifest')
    require(summary['status'] == 'complete' and summary['error_count'] == 0,
            'Recovery runtime audit requires a complete zero-error batch')
    require(manifest['model'] == summary['model'] == launch['model_name'], 'Model/service mismatch')
    require(launch['profile'] == 'navigation' and launch['keep_ratio'] == .75
            and launch['weight_quantization'] is None, 'Unexpected profile/weight quantization')
    engine = launch['engine_config']
    require(engine['dtype'] == 'float16' and engine['quant_policy'] == 0
            and engine['eager_mode'] is True and engine['max_batch_size'] == 1,
            'Expected serial eager FP16 service')
    expected = [str(eid) for eid in manifest['episodes']]
    actual = [str(row['episode_id']) for row in summary['episodes']]
    require(len(expected) == len(set(expected)) == len(actual) == len(set(actual))
            and set(expected) == set(actual), 'Recovered episode membership mismatch')
    require(summary['warmup_requests_excluded'] == 2,
            'Recovery evaluator must report its two new warmups, separately from old warmups')
    old_calls, new_calls, all_calls = [], [], []
    evidence, prefix_steps, completed_prefix = {}, 0, 0
    partial_prefix_ids = []
    new_steps_seen = False
    for episode in summary['episodes']:
        eid = str(episode['episode_id'])
        folder, old_folder = run / ('episode_' + eid), old / ('episode_' + eid)
        path, old_path = folder / 'steps.jsonl', old_folder / 'steps.jsonl'
        rows = rows_at(path)
        prior = rows_at(old_path) if old_path.exists() else []
        require(len(rows) == episode['steps'] and
                all(r['step'] == i and str(r.get('episode_id')) == eid for i, r in enumerate(rows)),
                'Incomplete/nonsequential recovered records: ' + eid)
        require(all(not r.get('error') and type(r.get('model_called')) is bool for r in rows),
                'Step errors or missing model_called: ' + eid)
        require(len(rows) >= len(prior) and episode.get('prefix_reused_steps') == len(prior),
                'Prefix length mismatch: ' + eid)
        if prior:
            require(not new_steps_seen, 'Old prefix is not contiguous in actual episode order')
            require(rows[:len(prior)] == [dict(r, prefix_reused=True) for r in prior],
                    'Previously saved prefix rows changed: ' + eid)
            require(all(r.get('prefix_reused') is True for r in rows[:len(prior)]),
                    'Prefix reuse markers must be boolean true: ' + eid)
            require(all(not r.get('prefix_reused') for r in prior), 'Nested recovery not supported')
            old_calls.extend(r for r in prior if r['model_called'])
            prefix_steps += len(prior)
            evidence['old/' + str(old_path.relative_to(old))] = sha(old_path)
        suffix = rows[len(prior):]
        require(all(not r.get('prefix_reused') for r in suffix), 'New steps falsely marked reused')
        new_steps_seen = new_steps_seen or bool(suffix)
        new_calls.extend(r for r in suffix if r['model_called'])
        all_calls.extend(r for r in rows if r['model_called'])
        evidence['new/' + str(path.relative_to(run))] = sha(path)
        if (old_folder / 'result.json').exists():
            original = read(old_folder / 'result.json')
            require(len(prior) == len(rows), 'Completed old episode received new steps')
            for key in ('episode_id', 'trajectory_id', 'steps', 'metrics', 'error_count',
                        'terminal_stop_source', 'elapsed_s', 'start_distance_m', 'path_length_m'):
                require(episode[key] == original[key], 'Completed prefix result changed: ' + eid + '/' + key)
            evidence['old/' + str((old_folder / 'result.json').relative_to(old))] = sha(old_folder / 'result.json')
            completed_prefix += 1
        elif prior:
            partial_prefix_ids.append(eid)
            require('lower-bound' in episode.get('timing_note', ''), 'Missing interruption timing caveat')
    require(old_calls and new_calls and len(partial_prefix_ids) <= 1, 'Unsupported/empty recovery segments')
    require(len(all_calls) == len(old_calls) + len(new_calls), 'Prefix model calls counted more than once')
    old_before = set(read(old / 'runtime_before.json'))
    new_before = set(new_before_names)
    current = {path.name for path in service.glob('prefill_*.json')}
    require(old_before <= new_before <= current, 'Runtime trace inventory disappeared or reordered')
    decodes = {path.name for path in service.glob('first_decode_*.json')}
    require(decodes == {name.replace('prefill_', 'first_decode_', 1) for name in current},
            'Missing or orphan first_decode runtime evidence')
    warmup_sig = signature(all_calls[0])
    old_traces = audit_segment(service, new_before - old_before, old_calls, warmup_sig)
    new_traces = audit_segment(service, current - new_before, new_calls, warmup_sig)
    traces = old_traces + new_traces
    require(len({row['trace_id'] for row in traces}) == len(traces), 'Duplicate trace association')
    counts = Counter((r['image_count'], r['original_image_tokens'], r['retained_image_tokens'],
                      r['original_length'], r['retained_length']) for r in traces)
    pruning = [dict(image_count=k[0], original_image_tokens=k[1], retained_image_tokens=k[2],
                    original_length=k[3], retained_length=k[4], requests=v)
               for k, v in sorted(counts.items())]
    evidence.update(manifest=sha(run / 'manifest.json'), summary=sha(run / 'summary.json'),
                    launch=sha(service / 'launch.json'), old_before=sha(old / 'runtime_before.json'))
    return dict(status='passed', passed=True, run=str(run), prefix_run=str(old),
                service_directory=str(service), actual_model_calls=len(all_calls),
                prefix_model_calls=len(old_calls), continuation_model_calls=len(new_calls),
                prefix_reused_steps=prefix_steps, completed_prefix_episodes=completed_prefix,
                interrupted_episode_ids=partial_prefix_ids, warmup_requests=4,
                old_warmup_requests=2, new_warmup_requests=2, runtime_trace_count=len(traces),
                old_runtime_trace_count=len(old_traces), new_runtime_trace_count=len(new_traces),
                association_method='Separate frozen old/new prefill inventories; chronological image/token signatures within each serial session; reused responses associated once with old traces',
                pruning_counts=pruning, traces=traces, evidence_sha256=evidence,
                limitations=['Association uses input signatures and inventory boundaries, not explicit HTTP request IDs.',
                             'Interrupted episode loop time is a lower bound; it excludes original prefix IO and downtime.',
                             'Effective upper-layer token/KV lengths are reduced; allocated blocks are unchanged.',
                             'This validates runtime integration, not navigation success or a matched speedup.'])
