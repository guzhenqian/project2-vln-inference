"""Resume the audited interrupted DART150 run without modifying frozen evidence.

Only --run contacts the existing service or launches an evaluator. Batch 1 uses
the unchanged pixel/pose replay runner; batches 2/3 use the unchanged evaluator.
"""
import argparse
import csv
import errno
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
from uuid import uuid4

import run_navigation_dart150 as base

ROOT = base.ROOT
PREFIX = ROOT / 'logs/navigation_dart150_keep75_20261108T100921887552Z_0ad70d02'
PREFIX_BATCH = PREFIX / 'batch_01'
PREFIX_AUDIT = ROOT / 'reports/recovery_prefix_audit_20261108T112639Z_01a86a49.json'
PREFIX_AUDIT_SHA256 = 'd059b83d022913d6924802df9fe578cb8649aedc6a8b0b4afb0907863bd9ed6e'
LOCK = ROOT / 'logs/navigation_dart150_keep75.active.lock'
RECOVERY_SOURCES = ('resume_navigation_dart150.py', 'resume_overnight_batch.py',
                    'audit_navigation_dart_recovery.py')
read, digest, require, now, stamp = base.read, base.digest, base.require, base.now, base.stamp
rel, root_path = base.rel, base.root_path


def write(path, value, *, exclusive=False, attempts=16):
    """Unique temporary file; retry only Windows sharing/permission conflicts.

    The previous target remains valid until replace succeeds. A persistent
    failure is surfaced; it never truncates the target or silently loses state.
    """
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
        return
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(content)
        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return
            except OSError as error:
                if not (isinstance(error, PermissionError) or error.errno == errno.EACCES):
                    raise
                if attempt + 1 == attempts:
                    raise
                time.sleep(min(0.1 * 2 ** attempt, 1.0))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Retain a uniquely named diagnostic fragment if locked.


def check_prefix(plan):
    """Rehash the already CPU-audited original evidence; no simulation/HTTP."""
    require(digest(PREFIX_AUDIT) == PREFIX_AUDIT_SHA256, 'Prefix audit SHA changed')
    audit = read(PREFIX_AUDIT)
    require(audit['status'] == 'passed' and audit['completed_episode_count'] == 31
            and audit['recorded_steps'] == 3639 and audit['actual_model_calls'] == 1014
            and audit['warmup_requests'] == 2 and not audit['observed_inflight_evidence']
            and audit['partial_episode']['episode_id'] == '1550'
            and audit['partial_episode']['recorded_steps'] == 8,
            'Prefix audit is not the reviewed interrupted run')
    evidence = (audit['original_evidence_files'] + audit['service_evidence_files']
                + list(audit['audited_script_files'].values()) + [audit['service_snapshot']])
    for item in evidence:
        path = root_path(item['path'])
        require(path.stat().st_size == item['size'] and digest(path) == item['sha256'],
                'Original recovery evidence changed: ' + item['path'])
    require(read(PREFIX / 'frozen_plan.json') == plan, 'Interrupted run has another frozen plan')
    state = read(PREFIX / 'status.json')
    require(state['status'] == 'failed' and 'PermissionError' in state['reason']
            and state['current_batch'] == 1 and state['current_episode'] == '1550',
            'Unexpected original interruption')
    item = plan['batches'][0]
    require(item['batch'] == 1 and read(PREFIX_BATCH / 'manifest.json') == read(root_path(item['path'])),
            'Prefix manifest differs from original full batch 1')
    completed, partial, rows_total, calls_total = [], [], 0, 0
    expected = set(map(str, item['episodes']))
    for folder in sorted(PREFIX_BATCH.glob('episode_*')):
        if not folder.is_dir():
            continue
        eid = folder.name[8:]
        require(eid in expected, 'Unexpected prefix episode: ' + eid)
        rows, truncated = base.step_rows(folder / 'steps.jsonl')
        require(rows and not truncated and all(str(row['episode_id']) == eid
                and row['step'] == i and not row.get('error') for i, row in enumerate(rows)),
                'Invalid prefix steps: ' + eid)
        for row in rows:
            frame = (PREFIX_BATCH / row['frame']).resolve()
            require(frame.is_relative_to(PREFIX_BATCH.resolve()) and frame.stat().st_size > 0,
                    'Missing/empty referenced prefix RGB: ' + eid)
        rows_total += len(rows)
        calls_total += sum(row.get('model_called') is True for row in rows)
        if (folder / 'result.json').exists():
            result = read(folder / 'result.json')
            require(result['steps'] == len(rows) and result['error_count'] == 0,
                    'Prefix result mismatch: ' + eid)
            completed.append(eid)
        else:
            partial.append((eid, len(rows)))
    require(len(completed) == 31 and partial == [('1550', 8)]
            and rows_total == 3639 and calls_total == 1014, 'Original prefix accounting changed')
    return audit, completed


def check():
    plan = base.check()
    audit, completed = check_prefix(plan)
    for name in RECOVERY_SOURCES:
        require((ROOT / 'checks' / name).is_file(), 'Missing recovery source: ' + name)
    return plan, audit, completed


def recovery_plan(plan, completed):
    return {'created_utc': now(), 'original_plan_sha256': digest(base.PLAN / 'manifest.json'),
            'prefix_run': rel(PREFIX), 'prefix_batch': rel(PREFIX_BATCH),
            'prefix_audit_path': rel(PREFIX_AUDIT), 'prefix_audit_sha256': PREFIX_AUDIT_SHA256,
            'completed_prefix_episode_ids': completed, 'partial_prefix_episode_id': '1550',
            'partial_prefix_steps': 8, 'warmup_requests_by_batch': {'1': 4, '2': 2, '3': 2},
            'model': base.MODEL, 'endpoint': base.ENDPOINT,
            'recovery_source_checks': {rel(ROOT / 'checks' / name): digest(ROOT / 'checks' / name)
                                       for name in RECOVERY_SOURCES},
            'recovery_semantics': 'Replay original actions with exact pixel/pose checks; reuse old request rows once; restore RGB history and queued actions before continuation',
            'automatic_retry': False}


def check_recovery_plan(output, plan):
    frozen = read(Path(output) / 'recovery_plan.json')
    require(read(Path(output) / 'frozen_plan.json') == plan, 'Recovery frozen plan changed')
    require(frozen['original_plan_sha256'] == digest(base.PLAN / 'manifest.json')
            and frozen['prefix_run'] == rel(PREFIX) and frozen['prefix_batch'] == rel(PREFIX_BATCH)
            and frozen['prefix_audit_path'] == rel(PREFIX_AUDIT)
            and frozen['prefix_audit_sha256'] == digest(PREFIX_AUDIT) == PREFIX_AUDIT_SHA256
            and frozen['model'] == base.MODEL and frozen['endpoint'] == base.ENDPOINT,
            'Recovery provenance changed')
    require(set(frozen['recovery_source_checks']) == {'checks/' + name for name in RECOVERY_SOURCES},
            'Recovery source snapshot is incomplete')
    for name, expected in frozen['recovery_source_checks'].items():
        require(digest(root_path(name)) == expected, 'Recovery source SHA changed: ' + name)
    return frozen


def check_reused_rows(eid, rows, result=None):
    """Old rows must be reproduced exactly, apart from the explicit reuse flag."""
    old_folder = PREFIX_BATCH / ('episode_' + eid)
    old_rows, _ = base.step_rows(old_folder / 'steps.jsonl')
    reused = 0
    for index, row in enumerate(rows):
        if index < len(old_rows):
            require(row.get('prefix_reused') is True, 'Missing reused-prefix marker: ' + eid)
            stripped = dict(row)
            stripped.pop('prefix_reused')
            require(stripped == old_rows[index], 'Replayed prefix row changed: ' + eid)
            reused += 1
        else:
            require(not row.get('prefix_reused'), 'Fresh step marked as reused: ' + eid)
    if result is not None:
        require(reused == len(old_rows) == result['prefix_reused_steps'], 'Incomplete prefix replay: ' + eid)
        if (old_folder / 'result.json').exists():
            original = read(old_folder / 'result.json')
            require(all(result[key] == value for key, value in original.items()),
                    'Completed original metrics/elapsed changed: ' + eid)
        if reused:
            require(bool(result.get('timing_note')), 'Recovery timing caveat missing: ' + eid)
    return reused


def warmups_observed(directory):
    path = directory / 'console.log'
    return sum(line.startswith('WARMUP ') for line in path.read_text(encoding='utf-8').splitlines()) if path.exists() else 0


def gpu_peaks(path):
    values = []
    if path.exists():
        with path.open(encoding='utf-8') as stream:
            for row in csv.reader(stream):
                try:
                    value = float(row[1])
                except (ValueError, IndexError):
                    continue
                if math.isfinite(value):
                    values.append(value)
    return values


def audit_completed_batch(directory, before):
    try:
        if directory.name == 'batch_01':
            from audit_navigation_dart_recovery import audit_resumed_batch
            audit = audit_resumed_batch(directory, PREFIX_BATCH, base.SERVICE, before)
        else:
            from audit_navigation_dart_run import audit_batch
            audit = audit_batch(directory, base.SERVICE, before_names=before)
        require(audit.get('passed') is True and audit.get('status') == 'passed', 'Runtime audit did not pass')
    except Exception as error:
        write(directory / 'runtime_audit.json', {'status': 'failed', 'passed': False, 'error': str(error)})
        raise
    write(directory / 'runtime_audit.json', audit)
    return audit


def run():
    require(os.name == 'posix', '--run requires the existing Ubuntu/WSL environment')
    plan, _, completed = check()
    require(not LOCK.exists(), 'Original DART150 lock is held; no process will be stopped automatically')
    output = ROOT / 'logs' / ('navigation_dart150_keep75_resumed_' + stamp())
    output.mkdir(exist_ok=False)
    state = {'status': 'starting', 'started_utc': now(), 'model': base.MODEL, 'planned_count': 150,
             'completed_batches': [], 'controller_pid': os.getpid(), 'reason': '',
             'service_directory': rel(base.SERVICE), 'automatic_retry': False,
             'prefix_run': rel(PREFIX)}
    write(output / 'status.json', state)
    write(output / 'frozen_plan.json', plan, exclusive=True)
    write(output / 'recovery_plan.json', recovery_plan(plan, completed), exclusive=True)
    child = None
    lock_token, lock_owned = uuid4().hex, False
    current = before = None
    print('DART150_RECOVERY_RUN_DIRECTORY', rel(output), flush=True)
    try:
        write(LOCK, {'token': lock_token, 'controller_pid': os.getpid(), 'run_directory': rel(output)}, exclusive=True)
        lock_owned = True
        write(output / 'service_models.json', base.verify_service(), exclusive=True)
        env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', PYTHONUNBUFFERED='1')
        env.pop('DISPLAY', None)
        for item in plan['batches']:
            # Original sources/data and all new recovery sources are checked each batch.
            require(base.check() == plan, 'Original plan changed during recovery')
            check_prefix(plan)
            check_recovery_plan(output, plan)
            current = output / f"batch_{item['batch']:02d}"
            current.mkdir(exist_ok=False)
            before = sorted(path.name for path in base.SERVICE.glob('prefill_*.json'))
            write(current / 'runtime_before.json', before, exclusive=True)
            script = 'resume_overnight_batch.py' if item['batch'] == 1 else 'run_simplevln_diagnostic.py'
            command = ['bash', 'scripts/run_habitat_task_python.sh', 'checks/' + script,
                       '--policy', 'memory', '--manifest', str(root_path(item['path'])),
                       '--output', str(current), '--endpoint', base.ENDPOINT]
            if item['batch'] == 1:
                command.extend(['--prefix', str(PREFIX_BATCH)])
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
                        require(eid in item['episodes'], 'Evaluator emitted an unexpected episode')
                        path = current / ('episode_' + eid) / 'steps.jsonl'
                        with path.open(encoding='utf-8') as stream:
                            stream.seek(offsets.get(eid, 0))
                            rows = [json.loads(value) for value in stream.read().splitlines()]
                            offsets[eid] = stream.tell()
                        require(rows, 'MODEL_STEP without saved rows')
                        for row in rows:
                            require(not row.get('error'), f"Request/policy error in {eid}/{row['step']}: {row.get('error')}")
                        state.update(updated_utc=now(), current_episode=eid,
                                     current_step=rows[-1]['step'] + 1, phase='evaluation')
                        write(output / 'status.json', state)
                    elif line.startswith('PREFIX_REPLAY_VERIFIED '):
                        parts = line.split()
                        state.update(updated_utc=now(), current_episode=parts[1],
                                     current_step=int(parts[2]), phase='prefix_replay')
                        write(output / 'status.json', state)
                returncode = child.wait()
                child = None
            require(returncode == 0, f"Evaluator batch {item['batch']} exited {returncode}")
            report = summarize(output, plan)
            require(item['batch'] in report['completed_batches'] and not report['errors'],
                    'Evaluator did not complete its exact 50 tasks without errors')
            audit_completed_batch(current, before)
            state['completed_batches'].append(item['batch'])
            state.update(updated_utc=now(), completed_count=report['completed_episode_count'])
            write(output / 'status.json', state)
            write(output / 'report.json', summarize(output, plan))
        state.update(status='complete', reason='All original 150 tasks and all runtime audits passed', finished_utc=now())
    except BaseException as error:
        state.update(status='failed', reason=f'{type(error).__name__}: {error}',
                     traceback=traceback.format_exc(), finished_utc=now())
        print('DART150_RECOVERY_FAILED', state['reason'], flush=True)
    finally:
        try:
            base.terminate_owned(child)
        except Exception as error:
            state.update(status='failed', reason=state['reason'] + '; owned evaluator cleanup failed: ' + str(error))
        try:
            write(output / 'status.json', state)
            if current is not None and not (current / 'runtime_audit.json').exists():
                try:
                    audit_completed_batch(current, before)
                except Exception:
                    pass
            report = summarize(output, plan)
            write(output / 'report.json', report)
            print('DART150_RECOVERY_STATUS', state['status'], f"{report['completed_episode_count']}/150",
                  'report=' + rel(output / 'report.json'), flush=True)
        except Exception:
            (output / 'summary_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
            state.update(status='failed', reason=state['reason'] + '; final state/summary validation failed')
            try:
                write(output / 'status.json', state)
            except OSError:
                pass
        finally:
            if (lock_owned and (child is None or child.poll() is not None)
                    and LOCK.exists() and read(LOCK).get('token') == lock_token):
                LOCK.unlink()
    return 0 if state['status'] == 'complete' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--summarize', type=Path, metavar='RUN_DIRECTORY')
    args = parser.parse_args()
    if args.check_only:
        plan, audit, completed = check()
        print('DART150_RECOVERY_CHECK_ONLY_OK', json.dumps({'planned': plan['planned_count'],
              'completed_prefix': len(completed), 'partial_episode': '1550', 'partial_steps': 8,
              'prefix_model_calls': audit['actual_model_calls'], 'lock_present': LOCK.exists(),
              'http_model_simulation': False}), flush=True)
    elif args.run:
        raise SystemExit(run())
    else:
        plan, _, _ = check()
        output = args.summarize.resolve()
        require(output.is_relative_to((ROOT / 'logs').resolve())
                and output != PREFIX, 'Summary target must be a new recovery run under logs')
        report = summarize(output, plan)
        path = output / ('report_snapshot_' + stamp() + '.json')
        write(path, report, exclusive=True)
        print('DART150_RECOVERY_SUMMARY', rel(path), report['status'], report['completed_episode_count'], flush=True)


def summarize(run_dir, plan=None):
    """Read actual results, including partial batches; never infer 150 completion."""
    plan = plan or base.check()
    recovery = check_recovery_plan(run_dir, plan)
    require(read(Path(run_dir) / 'frozen_plan.json') == plan, 'Run does not bind the current frozen plan')
    baseline = read(base.BASELINE)
    old = {str(e['episode_id']): e for e in baseline['episodes']}
    episodes, calls, errors, pending, peaks, completed_batches, audits = [], [], [], [], [], [], []
    missing_call_flag_errors = 0
    peaks.extend(gpu_peaks(PREFIX_BATCH / 'gpu_samples.csv'))
    warmup_counts = {}
    reused_steps = reused_calls = 0
    for item in plan['batches']:
        batch = read(root_path(item['path']))
        directory = Path(run_dir) / f"batch_{item['batch']:02d}"
        expected_ids = [str(eid) for eid in batch['episodes']]
        observed_warmups = warmups_observed(directory)
        require(0 <= observed_warmups <= 2, 'Unexpected warmup count')
        warmup_counts[str(item['batch'])] = observed_warmups + (2 if item['batch'] == 1 else 0)
        trajectories = dict(zip(expected_ids, map(str, batch['trajectory_ids'])))
        require(all(path.name[8:] in expected_ids for path in directory.glob('episode_*') if path.is_dir()),
                'Unexpected episode directory outside frozen membership')
        for eid in expected_ids:
            folder = directory / ('episode_' + eid)
            result_path = folder / 'result.json'
            rows, truncated = base.step_rows(folder / 'steps.jsonl', allow_partial=not result_path.exists())
            require(all(row['step'] == index and str(row['episode_id']) == eid for index, row in enumerate(rows)),
                    'Step ordering/episode identity mismatch: ' + eid)
            if item['batch'] == 1:
                reused_steps += check_reused_rows(eid, rows)
                reused_calls += sum(row.get('prefix_reused') is True and row.get('model_called') is True for row in rows)
            else:
                require(not any(row.get('prefix_reused') for row in rows), 'Unexpected prefix in fresh batch')
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
            if item['batch'] == 1:
                check_reused_rows(eid, rows, result)
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
            expected_scripts = {name: plan['source_checks']['checks/' + name] for name in base.RUNNERS}
            if item['batch'] == 1:
                del expected_scripts['run_simplevln_diagnostic.py']
                expected_scripts['resume_overnight_batch.py'] = recovery['recovery_source_checks']['checks/resume_overnight_batch.py']
                require(provenance['prefix_manifest_sha256'] == digest(PREFIX_BATCH / 'manifest.json')
                        and Path(provenance['prefix_run']).resolve() == PREFIX_BATCH.resolve(),
                        'Resume evaluator prefix provenance changed')
            require(run_manifest == batch and provenance['manifest_sha256'] == item['sha256']
                    and provenance['scripts'] == expected_scripts, 'Batch manifest/evaluator provenance mismatch')
            actual_ids = [str(e['episode_id']) for e in summary['episodes']]
            require(summary['status'] == 'complete' and summary['episode_count'] == len(expected_ids)
                    and len(actual_ids) == len(set(actual_ids)) == len(expected_ids)
                    and set(actual_ids) == set(expected_ids) and summary['warmup_requests_excluded'] == 2
                    and observed_warmups == 2,
                    'Invalid completed batch summary')
            require(summary['model'] == base.MODEL and summary['error_count'] ==
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
    audited = all(item['audit'] and item['audit'].get('passed') is True
                  and item['audit'].get('status') == 'passed' for item in audits)
    if audited:
        require([entry['audit']['warmup_requests'] for entry in audits] == [4, 2, 2],
                'Runtime audit warmup counts differ from the four sessions')
        require(sum(entry['audit']['actual_model_calls'] for entry in audits) == len(calls),
                'Runtime audits and deduplicated request counts differ')
    if len(completed_batches) == 3:
        require(warmup_counts == {'1': 4, '2': 2, '3': 2} and reused_steps == 3639
                and reused_calls == 1014, 'Final recovery accounting mismatch')
    complete = len(episodes) == 150 and len(completed_batches) == 3 and not errors and audited and state['status'] == 'complete'
    return {'updated_utc': now(), 'status': 'complete' if complete else 'partial_or_failed',
            'controller_state': state, 'planned_count': 150, 'completed_episode_count': len(episodes),
            'completed_batches': completed_batches, 'pending_episodes': pending, 'episodes': episodes,
            'sr': paired['dart_sr'], 'spl': paired['dart_spl'], 'paired_comparison': paired,
            'actual_request_latency_s': base.stats(calls), 'actual_request_scope': 'All saved steps, including incomplete episodes; model_called=true only',
            'errors': errors, 'error_count': len(errors), 'errors_missing_model_called_flag': missing_call_flag_errors,
            'completed_episode_steps': sum(item['steps'] for item in episodes),
            'saved_action_steps': sum(item['steps'] for item in episodes) + sum(item['saved_steps'] for item in pending),
            'collisions_completed_episodes': sum(item['metrics']['collisions']['count'] for item in episodes),
            'model_stop_success_count': sum(item['protocol_success'] for item in episodes),
            'fallback_stop_success_count': sum(bool(item['metrics']['success']) and item['terminal_stop_source'] == 'error_fallback' for item in episodes),
            'sampled_whole_card_peak_mib': max(peaks) if peaks else None,
            'runtime_audits': audits, 'all_runtime_audits_passed': audited,
            'plan_sha256': digest(base.PLAN / 'manifest.json'),
            'recovery_plan_sha256': digest(Path(run_dir) / 'recovery_plan.json'),
            'warmup_requests_excluded': sum(warmup_counts.values()),
            'recovery': dict(recovery, warmup_requests_by_batch=warmup_counts,
                             reused_steps_in_saved_rows=reused_steps, reused_calls_in_saved_rows=reused_calls),
            'memory_source_files': [rel(PREFIX_BATCH / 'gpu_samples.csv')] +
                [rel(Path(run_dir) / f"batch_{i:02d}/gpu_samples.csv") for i in range(1, 4)
                 if (Path(run_dir) / f"batch_{i:02d}/gpu_samples.csv").exists()],
            'notes': [plan['latency'], plan['memory'], plan['order'], plan['old_timing_note'], base.COMPARISON_NOTE,
                      'Partial quality comparisons use only the same completed episode IDs; unfinished tasks remain listed.',
                      'A model label or API Usage alone does not prove pruning; runtime traces are audited independently.',
                      'Old prefix model_called=true rows are counted exactly once; no prefix model replay requests.',
                      'Eight warmups for a complete run: old batch1 two, recovery batch1 two, batches2/3 two each.',
                      'Original 31 completed elapsed times are retained. Episode1550 elapsed is a lower-bound loop time: original decision+sim plus continuation, excluding original prefix image/log IO and downtime.',
                      'Whole-card peak combines original batch1 and all resumed batches; recovery/replay intervals are not pure inference measurements.']}



if __name__ == '__main__':
    main()
