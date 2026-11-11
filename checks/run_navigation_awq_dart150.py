"""Freeze, check, run or summarize the original 150 tasks with AWQ + DART75.

Standard library only. --prepare and --check-only never contact a service or
import a simulator/model. --run uses the existing service and unchanged author
policy evaluator; it never starts/stops a model service or retries a batch.
"""

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from uuid import uuid4


from navigation_awq_dart_profiles import AWQ_SPEC

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_SOURCE = None
ORIGINAL = ROOT / 'data/baseline/overnight200_fp16_20261031'
PLAN = ROOT / 'data/baseline/navigation_awq_dart150_keep75_20261109'
BASELINE = ROOT / 'reports/benchmark_overnight200_20261031.json'
OLD_LOGS = ROOT / 'logs/overnight200_fp16_20261031'
OLD_BATCHES = ('batch_01_resumed', 'batch_02', 'batch_03')
SINGLE_CONFIG = ROOT / 'setup/navigation-awq-dart-single-20261108.json'
SINGLE_RUN = ROOT / 'logs/navigation_dart_single_learning_20261108T091551Z_4612'
SINGLE_AUDIT = ROOT / 'reports/navigation_dart_single_audit_20261108T092417251229Z_e502335c.json'
SERVICE = ROOT / 'logs/navigation_dart_service_learning_20261108T065350824909Z_284bc210'
NINE_TRACE = 'c5f5a4d4b72f410c9708e5f11cbc9b33'
MODEL = 'SimpleVLN-Qwen3VL-4B-RL-AWQ-DART-keep75-navigation-20261108'
ENDPOINT = 'http://127.0.0.1:23333/v1'
RUNNERS = {'run_simplevln_diagnostic.py', 'simplevln_env.py',
           'simplevln_policy.py', 'run_closed_loop_baseline.py'}
ADAPTER_FILES = ('navigation_awq_dart_runtime.py', 'navigation_awq_dart_module_map.py',
                 'navigation_awq_dart_profiles.py', 'navigation_dart_selection.py',
                 'navigation_dart_kv_layout.py', 'serve_navigation_awq_dart.py',
                 'audit_navigation_awq_dart_run.py', 'run_navigation_awq_dart150.py', 'navigation_dart50_support.py',
                 'navigation_awq_dart_guard.py', 'navigation_awq_dart_preflight.py', 'navigation_dart_profiles.py')
CHANGED_FIELDS = {'version', 'scope', 'model', 'dtype', 'backend', 'source_precision',
                  'differences_from_author_benchmark', 'pre_registered',
                  'dart_keep_ratio', 'dart_profile', 'execution_gate', 'weight_quantization'}
COMPARISON_NOTE = ('Old FP16 used CUDA graphs and max-prefill/chunk 1024; this run uses eager '
                   'execution and complete prefill up to 3072. Closed-loop inputs and trajectories '
                   'can differ. D also records144 native AWQ module calls per forward. These instrumentation and execution differences prevent a pure algorithm speedup claim.')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def stamp():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]


def write(path, value, *, exclusive=False):
    from navigation_dart50_support import write as durable_write
    return durable_write(path, value, exclusive=exclusive)


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def root_path(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT.resolve()), 'Evidence path leaves the project: ' + relative)
    return path


def stats(values):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * 0.95
    low = int(index)
    return {'count': len(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'total': sum(values),
            'p95': values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)}


def clone_batch(original, single):
    """Change descriptive/model fields only; retain every original task field."""
    require(single['model'] == MODEL and single['dtype'] == 'W4A16' and single['weight_quantization'] == AWQ_SPEC
            and single['dart_keep_ratio'] == 0.75 and single['dart_profile'] == 'navigation',
            'Single-episode protocol is not the required AWQ DART75 navigation configuration')
    result = copy.deepcopy(original)
    result.update(version='navigation_awq_dart150_keep75_20261109', model=single['model'],
                  dtype=single['dtype'], backend=single['backend'],
                  source_precision=single['source_precision'], dart_keep_ratio=0.75,
                  dart_profile='navigation', weight_quantization=AWQ_SPEC,
                  scope='Exact original completed FP16 150-task development subset; three frozen batches of 50; AWQ + DART75',
                  differences_from_author_benchmark='Habitat0.1.7/LMDeploy; AWQ W4A16 + layer-internal DART75; fixed150, not official full1839',
                  pre_registered='Original membership and manifest order frozen20261031; exact completed150; no outcome filtering',
                  execution_gate='Existing AWQ-DART75 navigation service only; passed fixed nine-image and full single-episode evidence; per-batch runtime traces must be audited')
    require(all(result[key] == original[key] for key in original if key not in CHANGED_FIELDS),
            'A core original protocol field changed')
    require(set(result) - set(original) <= CHANGED_FIELDS, 'Unexpected new protocol fields')
    return result


def collect_inputs():
    """Verify old evidence and all source/data hashes before any plan write."""
    pins = {}

    def pin(path, expected=None):
        actual = digest(path)
        require(expected is None or actual == expected, 'SHA256 mismatch: ' + rel(path))
        pins[rel(path)] = actual
        return actual

    originals = [read(ORIGINAL / f'batch_{i:02d}.json') for i in range(1, 4)]
    ids = [str(eid) for batch in originals for eid in batch['episodes']]
    require(all(len(batch['episodes']) == len(set(batch['episodes'])) == 50 for batch in originals)
            and len(ids) == len(set(ids)) == 150, 'Expected three disjoint batches of 50')
    baseline = read(BASELINE)
    base = {str(item['episode_id']): item for item in baseline['episodes']}
    require(baseline['completed_episode_count'] == len(baseline['episodes']) == len(base) == 150
            and set(base) == set(ids) and not baseline['errors'], 'Old baseline does not contain the exact completed150')
    pin(BASELINE)
    provenance = read(OLD_LOGS / 'batch_02/provenance.json')
    require(set(provenance['scripts']) == RUNNERS, 'Unexpected baseline evaluator provenance')
    for name, expected in provenance['scripts'].items():
        pin(ROOT / 'checks' / name, expected)
    pin(OLD_LOGS / 'batch_02/provenance.json')
    pin(ROOT / 'scripts/run_habitat_task_python.sh')
    for i, original in enumerate(originals, 1):
        pin(ORIGINAL / f'batch_{i:02d}.json')
        pin(ROOT / original['dataset_path'].format(split=original['split']), original['source_annotation_sha256'])
        pin(ROOT / 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py', original['author_policy_sha256'])
        for scene in original['scene_files']:
            if scene['path'] not in pins:
                pin(ROOT / scene['path'], scene['sha256'])
            else:
                require(pins[scene['path']] == scene['sha256'], 'Conflicting scene hashes')
        old_run = OLD_LOGS / OLD_BATCHES[i - 1]
        summary = read(old_run / 'summary.json')
        actual = [str(item['episode_id']) for item in summary['episodes']]
        require(summary['status'] == 'complete' and summary['episode_count'] == 50
                and len(actual) == len(set(actual)) == 50 and set(actual) == set(original['episodes'])
                and summary['error_count'] == 0 and summary['warmup_requests_excluded'] == 2,
                'Old batch is incomplete or changed: ' + old_run.name)
        pin(old_run / 'summary.json')
        for eid in original['episodes']:
            folder = old_run / ('episode_' + str(eid))
            result = read(folder / 'result.json')
            for key in ('episode_id', 'trajectory_id', 'steps', 'metrics', 'terminal_stop_source', 'error_count'):
                require(result[key] == base[str(eid)][key], f'Old report/result mismatch: {eid}/{key}')
            pin(folder / 'result.json')
            pin(folder / 'steps.jsonl', base[str(eid)]['steps_sha256'])

    single = read(SINGLE_CONFIG)
    single_summary = read(SINGLE_RUN / 'summary.json')
    require(single_summary['status'] == 'complete' and single_summary['model'] == MODEL
            and single_summary['episode_count'] == 1 and single_summary['error_count'] == 0
            and [str(e['episode_id']) for e in single_summary['episodes']] == ['1250'],
            'Required single-episode integration evidence is missing or failed')
    single_provenance = read(SINGLE_RUN / 'provenance.json')
    require(single_provenance['scripts'] == provenance['scripts'], 'Single episode used another evaluator')
    pin(SINGLE_CONFIG, single_provenance['manifest_sha256'])
    for name in ('summary.json', 'manifest.json', 'provenance.json',
                 'episode_1250/result.json', 'episode_1250/steps.jsonl'):
        pin(SINGLE_RUN / name)
    from audit_navigation_awq_dart_run import audit_batch
    single_audit = read(SINGLE_AUDIT)
    checked = audit_batch(SINGLE_RUN, SERVICE, read(SINGLE_RUN / 'runtime_before.json'))
    require(single_audit['passed'] is True and checked['passed'] is True
            and single_audit['traces'] == checked['traces'],
            'Single-episode keep75 runtime audit is missing or changed')
    pin(SINGLE_AUDIT)
    pin(SINGLE_RUN / 'runtime_before.json')
    for trace in single_audit['traces']:
        for kind in ('prefill', 'first_decode'):
            pin(SERVICE / (kind + '_' + trace['trace_id'] + '.json'), trace[kind + '_sha256'])
    nine = read(SERVICE / ('prefill_' + NINE_TRACE + '.json'))
    decode = read(SERVICE / ('first_decode_' + NINE_TRACE + '.json'))
    require(nine['profile'] == 'navigation' and nine['keep_ratio'] == 0.75
            and nine['image_count'] == 9 and nine['original_length'] == 2815
            and nine['retained_length'] == 2140 and nine['retained_image_tokens'] == 2025
            and nine['retained_image_tokens_per_image'] == [225] * 9 and nine['awq'] is True
            and decode['lower_layer_kv_length'] == 2816 and decode['upper_layer_kv_length'] == 2141
            and decode['original_positions_preserved'], 'Required nine-image DART/KV evidence did not pass')
    launch = read(SERVICE / 'launch.json')
    engine = launch['engine_config']
    require(launch['model_name'] == MODEL and launch['keep_ratio'] == 0.75
            and launch['profile'] == 'navigation' and engine['eager_mode'] is True
            and engine['max_prefill_token_num'] == 3072 and engine['quant_policy'] == 0
            and engine['dtype'] == 'float16' and engine['num_gpu_blocks'] == 64
            and not engine['enable_prefix_caching'], 'Service provenance is not the frozen DART configuration')
    for name in ('launch.json', 'prefill_' + NINE_TRACE + '.json', 'first_decode_' + NINE_TRACE + '.json'):
        pin(SERVICE / name)
    model_config = root_path(launch['model_source']) / 'config.json'
    if model_config.exists():
        pin(model_config, launch['model_config_sha256'])
    for name in ADAPTER_FILES:
        path = ROOT / 'checks' / name
        # The existing service must use the same runtime source, rather than a
        # same-named model label hiding a different adapter revision.
        expected = launch.get('sources_sha256', {}).get(path.name)
        pin(path, expected)
    require('navigation_awq_dart_runtime.py' in launch['sources_sha256'],
            'Service launch did not bind the runtime source hash')
    from audit_navigation_awq_dart_run import audit_trace, audit_launch, audit_runtime_init
    audit_launch(SERVICE, .75)
    audit_runtime_init(SERVICE, .75)
    audit_trace(SERVICE / ('prefill_' + NINE_TRACE + '.json'), keep_ratio=.75)
    for path in SERVICE.glob('runtime_init_*.json'):
        pin(path)
    pin(root_path(launch['model_source']) / 'model.safetensors',
        launch['checkpoint_verification']['historical_full_weight_sha256'])
    evidence = read(EVIDENCE_SOURCE)
    pin(EVIDENCE_SOURCE)
    validation = root_path(evidence['validation_directory'])
    for label, ratio in (('keep100', 1.), ('keep75', .75)):
        record = read(validation / (label + '_smoke_audit.json'))
        require(record['passed'] is True and len(record['traces']) == 2, 'Missing GPU smoke pair')
        service = root_path(record['service'])
        pin(validation / (label + '_smoke_audit.json'))
        pin(service / 'launch.json')
        for item in service.glob('runtime_init_*.json'):
            pin(item)
        for trace in record['traces']:
            checked = audit_trace(service / ('prefill_' + trace['trace_id'] + '.json'), keep_ratio=ratio)
            require(checked == trace, 'GPU control/combination trace changed')
            for kind in ('prefill', 'first_decode'):
                pin(service / (kind + '_' + trace['trace_id'] + '.json'), trace[kind + '_sha256'])
    return originals, single, pins


def prepare():
    require(not PLAN.exists(), 'Frozen plan already exists; use --check-only (never overwrite it)')
    originals, single, pins = collect_inputs()
    batches = [clone_batch(original, single) for original in originals]
    PLAN.mkdir(parents=True, exist_ok=False)
    records = []
    for index, batch in enumerate(batches, 1):
        path = PLAN / f'batch_{index:02d}.json'
        write(path, batch, exclusive=True)
        records.append({'batch': index, 'episodes': batch['episodes'], 'trajectory_ids': batch['trajectory_ids'],
                        'path': rel(path), 'sha256': digest(path),
                        'source_manifest': rel(ORIGINAL / path.name),
                        'source_sha256': pins[rel(ORIGINAL / path.name)]})
    plan = {'created_utc': now(), 'planned_count': 150, 'model': MODEL, 'keep_ratio': 0.75,
            'profile': 'navigation', 'batches': records, 'source_checks': pins,
            'baseline_report': rel(BASELINE), 'service_directory': rel(SERVICE),
            'single_episode_evidence': rel(SINGLE_RUN), 'nine_image_trace': NINE_TRACE,
            'single_episode_runtime_audit': rel(SINGLE_AUDIT),
            'protocol': 'Unchanged author policy, seed7, max400 primitive actions; model STOP and distance<3m; three fixed50 batches',
            'order': 'Manifest listing/order unchanged; unchanged Habitat GROUP_BY_SCENE deterministically groups execution by scene',
            'latency': 'steps.jsonl model_called is True only; warmups are not written to steps, so do not discard first two real calls',
            'memory': 'Whole-card memory.used sampled every200ms; includes desktop, engine cache and activations; not model-exclusive',
            'comparison_note': COMPARISON_NOTE,
            'old_timing_note': 'Old batch1 was resumed; episode735 loop timing omits prefix IO; no exact total-time comparison',
            'precision_note': 'Old source_precision text mentions BF16, but validated baseline runtime was FP16',
            'runtime_audit': 'Per-batch new prefill/first_decode traces must match all actual requests plus2warmups; current150 traces are not yet collected',
            'failure_policy': 'Stop on evaluator/request/parse/audit error; preserve partial results; no automatic retry or service operation'}
    write(PLAN / 'manifest.json', plan, exclusive=True)
    print('DART150_PLAN_FROZEN', rel(PLAN), '150 tasks; no HTTP/model/simulation', flush=True)
    return plan


def check():
    plan = read(PLAN / 'manifest.json')
    require(plan['planned_count'] == 150 and plan['model'] == MODEL
            and plan['service_directory'] == rel(SERVICE), 'Frozen plan identity changed')
    for relative, expected in plan['source_checks'].items():
        require(digest(root_path(relative)) == expected, 'Frozen source/evidence changed: ' + relative)
    single = read(SINGLE_CONFIG)
    ids = []
    require(len(plan['batches']) == 3, 'Expected exactly three frozen batches')
    for index, item in enumerate(plan['batches'], 1):
        require(item['batch'] == index, 'Batch order changed')
        path = root_path(item['path'])
        require(digest(path) == item['sha256'], 'Frozen DART batch changed')
        batch = read(path)
        require(batch == clone_batch(read(ORIGINAL / f'batch_{index:02d}.json'), single),
                'Frozen batch does not match its original core protocol')
        require(batch['episodes'] == item['episodes'] and batch['trajectory_ids'] == item['trajectory_ids'],
                'Frozen task order or trajectory binding changed')
        ids.extend(batch['episodes'])
    require(len(ids) == len(set(ids)) == 150, 'Frozen membership is not exactly150')
    return plan


def step_rows(path, *, allow_partial=False):
    """Only an interrupted file's unfinished final line may be omitted."""
    if not path.exists():
        return [], False
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    rows, truncated = [], False
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if allow_partial and index == len(lines) - 1 and not text.endswith('\n'):
                truncated = True
            else:
                raise
    return rows, truncated


def summarize(run_dir, plan=None):
    """Read actual results, including partial batches; never infer 150 completion."""
    plan = plan or check()
    require(read(Path(run_dir) / 'frozen_plan.json') == plan, 'Run does not bind the current frozen plan')
    baseline = read(BASELINE)
    old = {str(e['episode_id']): e for e in baseline['episodes']}
    episodes, calls, errors, pending, peaks, completed_batches, audits = [], [], [], [], [], [], []
    missing_call_flag_errors = 0
    for item in plan['batches']:
        batch = read(root_path(item['path']))
        directory = Path(run_dir) / f"batch_{item['batch']:02d}"
        expected_ids = [str(eid) for eid in batch['episodes']]
        trajectories = dict(zip(expected_ids, map(str, batch['trajectory_ids'])))
        require(all(path.name[8:] in expected_ids for path in directory.glob('episode_*') if path.is_dir()),
                'Unexpected episode directory outside frozen membership')
        for eid in expected_ids:
            folder = directory / ('episode_' + eid)
            result_path = folder / 'result.json'
            rows, truncated = step_rows(folder / 'steps.jsonl', allow_partial=not result_path.exists())
            require(all(row['step'] == index and str(row['episode_id']) == eid for index, row in enumerate(rows)),
                    'Step ordering/episode identity mismatch: ' + eid)
            for row in rows:
                if row.get('error'):
                    errors.append({'batch': item['batch'], 'episode_id': eid,
                                   'step': row['step'], 'error': row['error']})
                    missing_call_flag_errors += int('model_called' not in row)
                if row.get('model_called') is True:
                    value = float(row['model_request_s'])
                    require(math.isfinite(value) and value > 0, 'Invalid real HTTP latency: ' + eid)
                    calls.append(value)
            if not result_path.exists():
                pending.append({'batch': item['batch'], 'episode_id': eid,
                                'status': 'started_incomplete' if folder.exists() else 'not_started',
                                'saved_steps': len(rows), 'truncated_final_step_line': truncated})
                continue
            result = read(result_path)
            require(str(result['episode_id']) == eid and str(result['trajectory_id']) == trajectories[eid]
                    and len(rows) == result['steps'] and 0 < len(rows) <= batch['max_episode_steps'],
                    'Result/trajectory/step count mismatch: ' + eid)
            metric_success = rows[-1]['action'] == 'STOP' and result['metrics']['distance_to_goal'] < batch['success_distance_m']
            require(bool(result['metrics']['success']) == metric_success
                    and result['error_count'] == sum(bool(row.get('error')) for row in rows),
                    'Result success/error accounting mismatch: ' + eid)
            expected_stop_source = (('error_fallback' if rows[-1].get('error') else 'model')
                                    if rows[-1]['action'] == 'STOP' else 'step_limit')
            require(result['terminal_stop_source'] == expected_stop_source, 'Terminal STOP source mismatch: ' + eid)
            shortest, length = float(result['start_distance_m']), float(result['path_length_m'])
            require(math.isfinite(shortest) and math.isfinite(length) and shortest >= 0
                    and length >= 0 and max(shortest, length) > 0, 'Invalid route lengths: ' + eid)
            expected_spl = float(metric_success) * shortest / max(shortest, length)
            require(abs(float(result['metrics']['spl']) - expected_spl) < 1e-5, 'SPL arithmetic mismatch: ' + eid)
            success = int(metric_success and result['terminal_stop_source'] == 'model')
            episodes.append(dict(result, batch=item['batch'], protocol_success=success,
                                 protocol_spl=float(result['metrics']['spl']) if success else 0.0,
                                 actual_model_calls=sum(row.get('model_called') is True for row in rows),
                                 steps_sha256=digest(folder / 'steps.jsonl')))
        summary_path = directory / 'summary.json'
        if summary_path.exists():
            summary = read(summary_path)
            run_manifest = read(directory / 'manifest.json')
            provenance = read(directory / 'provenance.json')
            expected_scripts = {name: plan['source_checks']['checks/' + name] for name in RUNNERS}
            require(run_manifest == batch and provenance['manifest_sha256'] == item['sha256']
                    and provenance['scripts'] == expected_scripts, 'Batch manifest/evaluator provenance mismatch')
            actual_ids = [str(e['episode_id']) for e in summary['episodes']]
            require(summary['status'] == 'complete' and summary['episode_count'] == len(expected_ids)
                    and len(actual_ids) == len(set(actual_ids)) == len(expected_ids)
                    and set(actual_ids) == set(expected_ids) and summary['warmup_requests_excluded'] == 2,
                    'Invalid completed batch summary')
            require(summary['model'] == MODEL and summary['error_count'] ==
                    sum(entry['error_count'] for entry in episodes if entry['batch'] == item['batch']),
                    'Batch model/error summary mismatch')
            require(all((directory / ('episode_' + eid) / 'result.json').exists() for eid in expected_ids),
                    'Summary claims results that are absent')
            result_map = {str(entry['episode_id']): entry for entry in episodes if entry['batch'] == item['batch']}
            for entry in summary['episodes']:
                saved = result_map[str(entry['episode_id'])]
                for key in ('episode_id', 'trajectory_id', 'steps', 'metrics', 'terminal_stop_source',
                            'error_count', 'start_distance_m', 'path_length_m'):
                    require(entry[key] == saved[key], f'Batch summary/result mismatch: {entry["episode_id"]}/{key}')
            completed_batches.append(item['batch'])
        gpu = directory / 'gpu_samples.csv'
        if gpu.exists():
            with gpu.open(encoding='utf-8') as stream:
                for row in csv.reader(stream):
                    try:
                        value = float(row[1])
                    except (IndexError, ValueError):
                        continue
                    if math.isfinite(value):
                        peaks.append(value)
        audit_path = directory / 'runtime_audit.json'
        audits.append({'batch': item['batch'], 'audit': read(audit_path) if audit_path.exists() else None})
    paired_old = [old[str(item['episode_id'])] for item in episodes]
    mean = lambda values: statistics.mean(values) if values else None
    old_success = lambda item: int(bool(item['metrics']['success']) and item['terminal_stop_source'] == 'model')
    paired = {'count': len(episodes), 'fp16_sr': mean([old_success(item) for item in paired_old]),
              'fp16_spl': mean([item['metrics']['spl'] if old_success(item) else 0.0 for item in paired_old]),
              'dart_sr': mean([item['protocol_success'] for item in episodes]),
              'dart_spl': mean([item['protocol_spl'] for item in episodes]),
              'gained_success': [item['episode_id'] for item in episodes if item['protocol_success'] and not old_success(old[str(item['episode_id'])])],
              'lost_success': [item['episode_id'] for item in episodes if not item['protocol_success'] and old_success(old[str(item['episode_id'])])]}
    state = read(Path(run_dir) / 'status.json') if (Path(run_dir) / 'status.json').exists() else {'status': 'unknown'}
    audited = all(item['audit'] and item['audit'].get('passed') is True for item in audits)
    complete = len(episodes) == 150 and len(completed_batches) == 3 and not errors and audited and state['status'] == 'complete'
    return {'updated_utc': now(), 'status': 'complete' if complete else 'partial_or_failed',
            'controller_state': state, 'planned_count': 150, 'completed_episode_count': len(episodes),
            'completed_batches': completed_batches, 'pending_episodes': pending, 'episodes': episodes,
            'sr': paired['dart_sr'], 'spl': paired['dart_spl'], 'paired_comparison': paired,
            'actual_request_latency_s': stats(calls), 'actual_request_scope': 'All saved steps, including incomplete episodes; model_called=true only',
            'errors': errors, 'error_count': len(errors), 'errors_missing_model_called_flag': missing_call_flag_errors,
            'completed_episode_steps': sum(item['steps'] for item in episodes),
            'saved_action_steps': sum(item['steps'] for item in episodes) + sum(item['saved_steps'] for item in pending),
            'collisions_completed_episodes': sum(item['metrics']['collisions']['count'] for item in episodes),
            'model_stop_success_count': sum(item['protocol_success'] for item in episodes),
            'fallback_stop_success_count': sum(bool(item['metrics']['success']) and item['terminal_stop_source'] == 'error_fallback' for item in episodes),
            'sampled_whole_card_peak_mib': max(peaks) if peaks else None,
            'runtime_audits': audits, 'all_runtime_audits_passed': audited,
            'plan_sha256': digest(PLAN / 'manifest.json'),
            'notes': [plan['latency'], plan['memory'], plan['order'], plan['old_timing_note'], COMPARISON_NOTE,
                      'Partial quality comparisons use only the same completed episode IDs; unfinished tasks remain listed.',
                      'A model label or API Usage alone does not prove pruning; runtime traces are audited independently.']}


def verify_service():
    # This is the only controller HTTP request; generation belongs solely to the unchanged evaluator.
    with urllib.request.urlopen(ENDPOINT + '/models', timeout=10) as response:
        models = json.load(response)
    require(MODEL in [item['id'] for item in models['data']], 'Existing service does not advertise the frozen AWQ-DART75 model')
    return models


def terminate_owned(process):
    """Only a Popen created here in a new session can reach this function."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except ProcessLookupError:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def audit_completed_batch(directory, before):
    from audit_navigation_awq_dart_run import audit_batch
    try:
        audit = audit_batch(directory, SERVICE, before_names=before)
        require(audit.get('passed') is True and audit.get('status') == 'passed', 'Runtime audit did not pass')
    except Exception as error:
        write(directory / 'runtime_audit.json', {'status': 'failed', 'passed': False, 'error': str(error)})
        raise
    write(directory / 'runtime_audit.json', audit)
    return audit


def run():
    require(os.name == 'posix', '--run must be executed in the existing Ubuntu/WSL environment')
    plan = check()
    output = ROOT / 'logs' / ('navigation_awq_dart150_keep75_' + stamp())
    output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'starting', 'started_utc': now(), 'model': MODEL, 'planned_count': 150,
             'completed_batches': [], 'controller_pid': os.getpid(), 'reason': '',
             'service_directory': rel(SERVICE), 'automatic_retry': False}
    write(output / 'status.json', state)
    write(output / 'frozen_plan.json', plan, exclusive=True)
    child = None
    lock = ROOT / 'logs/navigation_awq_dart150_keep75.active.lock'
    lock_token = uuid4().hex
    lock_owned = False
    current = before = None
    print('DART150_RUN_DIRECTORY', rel(output), flush=True)
    try:
        write(lock, {'token': lock_token, 'controller_pid': os.getpid(), 'run_directory': rel(output)}, exclusive=True)
        lock_owned = True
        write(output / 'service_models.json', verify_service(), exclusive=True)
        env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', PYTHONUNBUFFERED='1')
        env.pop('DISPLAY', None)
        for item in plan['batches']:
            # Read-only integrity check before every new batch; never adopt or rerun an old evaluator.
            check()
            current = output / f"batch_{item['batch']:02d}"
            current.mkdir(exist_ok=False)
            before = sorted(path.name for path in SERVICE.glob('prefill_*.json'))
            write(current / 'runtime_before.json', before, exclusive=True)
            command = ['bash', 'scripts/run_habitat_task_python.sh', 'checks/run_simplevln_diagnostic.py',
                       '--policy', 'memory', '--manifest', str(root_path(item['path'])),
                       '--output', str(current), '--endpoint', ENDPOINT]
            state.update(status='running', current_batch=item['batch'], batch_started_utc=now(), command=command)
            write(output / 'status.json', state)
            offsets = {}
            with (current / 'console.log').open('x', encoding='utf-8') as console:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                         bufsize=1, start_new_session=True)
                state['evaluator_pid'] = child.pid
                write(output / 'status.json', state)
                for line in child.stdout:
                    console.write(line)
                    console.flush()
                    print(line, end='', flush=True)
                    if line.startswith('MODEL_STEP '):
                        eid = line.split()[1]
                        require(eid in item['episodes'], 'Evaluator emitted an unexpected episode ID')
                        path = current / ('episode_' + eid) / 'steps.jsonl'
                        with path.open(encoding='utf-8') as stream:
                            stream.seek(offsets.get(eid, 0))
                            new_rows = [json.loads(value) for value in stream.read().splitlines()]
                            offsets[eid] = stream.tell()
                        for row in new_rows:
                            if row.get('error'):
                                raise RuntimeError(f"Request/policy error in episode{eid}, step{row['step']}: {row['error']}")
                        require(new_rows, 'MODEL_STEP was printed without a new saved step')
                        state.update(updated_utc=now(), current_episode=eid, current_step=new_rows[-1]['step'] + 1)
                        write(output / 'status.json', state)
                returncode = child.wait()
                child = None
            require(returncode == 0, f'Evaluator batch{item["batch"]} exited with code{returncode}')
            report = summarize(output, plan)
            require(item['batch'] in report['completed_batches'] and not report['errors'],
                    'Evaluator did not complete its exact50 tasks without errors')
            audit_completed_batch(current, before)
            state['completed_batches'].append(item['batch'])
            state.update(updated_utc=now(), completed_count=report['completed_episode_count'])
            write(output / 'status.json', state)
            write(output / 'report.json', summarize(output, plan))
        state.update(status='complete', reason='All three fixed50 batches and runtime audits passed', finished_utc=now())
    except BaseException as error:
        state.update(status='failed', reason=f'{type(error).__name__}: {error}',
                     traceback=traceback.format_exc(), finished_utc=now())
        print('DART150_FAILED', state['reason'], flush=True)
    finally:
        try:
            terminate_owned(child)
        except Exception as error:
            state.update(status='failed', reason=state['reason'] + '; owned evaluator cleanup failed: ' + str(error))
        write(output / 'status.json', state)
        if current is not None and not (current / 'runtime_audit.json').exists():
            try:
                audit_completed_batch(current, before)
            except Exception:
                pass  # Failure evidence was written; evaluator/controller failure remains authoritative.
        try:
            report = summarize(output, plan)
            write(output / 'report.json', report)
            print('DART150_STATUS', state['status'], f"{report['completed_episode_count']}/150",
                  'report=' + rel(output / 'report.json'), flush=True)
        except Exception:
            (output / 'summary_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            state.update(status='failed', reason=state['reason'] + '; offline summary validation failed')
            write(output / 'status.json', state)
        if (lock_owned and (child is None or child.poll() is not None)
                and lock.exists() and read(lock).get('token') == lock_token):
            lock.unlink()
    return 0 if state['status'] == 'complete' else 1


def configure(evidence_path):
    global SERVICE, SINGLE_RUN, SINGLE_AUDIT, NINE_TRACE
    evidence = read(evidence_path)
    SERVICE = root_path(evidence['service_directory'])
    SINGLE_RUN = root_path(evidence['single_episode_evidence'])
    SINGLE_AUDIT = root_path(evidence['single_episode_runtime_audit'])
    NINE_TRACE = evidence['nine_image_trace']
    global EVIDENCE_SOURCE
    EVIDENCE_SOURCE = Path(evidence_path).resolve()
    require(EVIDENCE_SOURCE.is_relative_to(ROOT), 'Evidence must stay inside the project')
    from navigation_awq_dart_preflight import require_c_finished
    require_c_finished()
    validation = read(root_path(evidence['validation_directory']) / 'status.json')
    require(validation.get('gpu_combination_verified') is True and validation.get('evidence') == evidence,
            'Actual AWQ+DART GPU integration must pass before D150')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true', help='Freeze the exact150 protocol and local provenance; offline')
    modes.add_argument('--check-only', action='store_true', help='Recheck frozen hashes and task protocol; offline')
    modes.add_argument('--run', action='store_true', help='Run3x50 against the existing AWQ-DART75 service; no retries/service operations')
    modes.add_argument('--summarize', type=Path, metavar='RUN_DIRECTORY', help='Write a new offline summary snapshot of this controller run')
    parser.add_argument('--evidence', type=Path, default=ROOT / 'setup/navigation-awq-dart75-evidence-20261109.json')
    args = parser.parse_args()
    configure(args.evidence)
    if args.prepare:
        prepare()
    elif args.check_only:
        plan = check()
        print('DART150_CHECK_ONLY_OK', plan['planned_count'], 'No HTTP/model/simulation', flush=True)
    elif args.run:
        return run()
    else:
        directory = args.summarize.resolve()
        require(directory.is_relative_to((ROOT / 'logs').resolve())
                and directory.name.startswith('navigation_awq_dart150_keep75_'), 'Choose a DART150 run under project logs')
        report = summarize(directory)
        path = directory / ('summary_' + stamp() + '.json')
        write(path, report, exclusive=True)
        print('DART150_SUMMARY_SAVED', rel(path), report['completed_episode_count'], '/150', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f'DART150_ERROR {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1)
