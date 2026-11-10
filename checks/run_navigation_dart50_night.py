"""After audited C75 completion: switch service, verify C50, run exact150.

No service or evaluator is touched until a completed C75 report is supplied.
The only service stopped is the exact C75 project process (and its descendants)
or the C50 process group created by this program. Failures retain all evidence.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

from navigation_dart50_support import ROOT, C75_MODEL, read, write, digest, require_c75_complete
from navigation_dart50_profiles import served_model_name
from audit_navigation_dart50_run import audit_trace, audit_batch

PYTHON = str(Path.home() / '.venvs/project2/bin/python')
MODEL = served_model_name(.5, 'navigation')
API = 'http://127.0.0.1:23333/v1/models'


def now():
    return datetime.now(timezone.utc).isoformat()


def processes():
    found = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            words = [x for x in (path / 'cmdline').read_bytes().decode().split('\0') if x]
            found[int(path.name)] = {'ppid': int(stat[1]), 'start': stat[19],
                                     'words': words, 'cwd': str((path / 'cwd').resolve())}
        except (OSError, UnicodeError, ValueError):
            pass
    return found


def same_process(pid, snapshot):
    current = processes().get(pid)
    return current is not None and current['start'] == snapshot['start'] and bool(current['words'])


def release_completed_c75(output):
    inventory = processes()
    active = []
    owners = []
    for pid, record in inventory.items():
        if record['cwd'] != str(ROOT):
            continue
        words = record['words']
        script = next((Path(word).name for word in words if word.endswith('.py')), '')
        if ('checks/serve_navigation_dart_learning.py' in words
                and '--keep-ratio' in words and words[words.index('--keep-ratio') + 1] == '0.75'
                and '--profile' in words and words[words.index('--profile') + 1] == 'navigation'):
            owners.append(pid)
        if (script in ('run_simplevln_diagnostic.py', 'resume_overnight_batch.py',
                       'resume_navigation_run.py', 'run_navigation_dart150.py', 'resume_navigation_dart150.py')
                or ('dart' in script and 'recover' in script)):
            active.append(pid)
    if active:
        raise RuntimeError('C75 evaluator/recovery/controller still active: ' + str(active))
    if len(owners) > 1:
        raise RuntimeError('More than one C75 service owner; refuse ambiguous termination')
    if owners:
        # Read-only endpoint identity guards against a changed same-named process.
        with urlopen(API, timeout=15) as response:
            listing = json.load(response)
        if [row['id'] for row in listing['data']] != [C75_MODEL]:
            raise RuntimeError('Current service is not the completed C75 model')
        owner = owners[0]
        descendants = {owner}
        while True:
            more = {pid for pid, value in inventory.items() if value['ppid'] in descendants}
            if more <= descendants:
                break
            descendants |= more
        owned = {pid: inventory[pid] for pid in descendants}
        write(output / 'c75_service_release.json', {'started_utc': now(), 'owner_pid': owner,
              'processes': owned, 'reason': 'C75 exact150 and audits complete; user authorized C50 next'}, exclusive=True)
        for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            targets = [owner] if sig == signal.SIGINT else list(owned)
            for pid in targets:
                if same_process(pid, owned[pid]):
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline and any(same_process(pid, snap) for pid, snap in owned.items()):
                time.sleep(.5)
            if not any(same_process(pid, snap) for pid, snap in owned.items()):
                break
        if any(same_process(pid, snap) for pid, snap in owned.items()):
            raise RuntimeError('C75 owned processes did not exit; no C50 service will start')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                          '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    write(output / 'gpu_after_c75.json', {'utc': now(), 'memory_used_free_mib': gpu.stdout.strip()}, exclusive=True)


def stop_owned(child):
    if child is None or child.poll() is not None:
        return
    for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(child.pid, sig)
            child.wait(timeout=seconds)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass


def validate_service(service):
    launch = read(service / 'launch.json')
    engine = launch['engine_config']
    if not (launch['model_name'] == MODEL and launch['keep_ratio'] == .5
            and launch['profile'] == 'navigation' and launch['weight_quantization'] is None
            and engine['dtype'] == 'float16' and engine['max_prefill_token_num'] == 3072
            and engine['num_gpu_blocks'] == engine['block_size'] == 64
            and engine['eager_mode'] is True and engine['quant_policy'] == 0
            and engine['enable_prefix_caching'] is False):
        raise ValueError('Actual C50 launch identity/configuration mismatch')
    for name, expected in launch['sources_sha256'].items():
        if digest(ROOT / 'checks' / name) != expected:
            raise ValueError('Loaded C50 source changed: ' + name)
    required_sources = {'navigation_dart50_runtime.py', 'navigation_dart50_module_map.py',
                        'navigation_dart50_profiles.py', 'navigation_dart_selection.py',
                        'navigation_dart_kv_layout.py', 'serve_navigation_dart50.py'}
    if not required_sources <= set(launch['sources_sha256']):
        raise ValueError('C50 launch does not bind every runtime dependency')
    initializations = list(service.glob('runtime_init_*.json'))
    if len(initializations) != 1:
        raise ValueError('Expected one real C50 runtime initialization')
    init = read(initializations[0])
    if not (init['runtime_class'] == 'Dart50Qwen3VLForConditionalGeneration'
            and init['keep_ratio'] == .5 and init['awq'] is False and init['weight_dtype'] == 'float16'):
        raise ValueError('C50 runtime was not actually instantiated with FP16')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c75-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/navigation_dart50_night_20261108')
    args = parser.parse_args()
    output = args.output.resolve()
    if os.name != 'posix' or not output.is_relative_to(ROOT / 'logs'):
        raise ValueError('Run in WSL with a dedicated project logs directory')
    output.mkdir(exist_ok=True)
    state = read(output / 'status.json') if (output / 'status.json').exists() else {}
    if state.get('c50_model_started') or state.get('c50_navigation_started'):
        raise ValueError('Sequence has already started; inspect and recover its evidence, do not duplicate')
    c75 = require_c75_complete(args.c75_report)
    lock = ROOT / 'logs/navigation_dart50_night.active.lock'
    token = f'{os.getpid()}-{time.time_ns()}'
    write(lock, {'pid': os.getpid(), 'token': token, 'directory': str(output)}, exclusive=True)
    state.update(status='switching_after_c75', c75_report=c75, controller_pid=os.getpid(), started_utc=now())
    service_child = task_child = None
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', LIBGL_ALWAYS_SOFTWARE='1')
    env.pop('DISPLAY', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)

    def update(**values):
        state.update(values, updated_utc=now())
        write(output / 'status.json', state)
        print('DART50_SEQUENCE', state['status'], flush=True)

    def command(argv, name, phase, timeout=10 * 3600):
        nonlocal task_child
        update(status=phase, command=argv)
        with (output / (name + '.log')).open('x', encoding='utf-8') as stream:
            task_child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            state['active_child_pid'] = task_child.pid
            start = time.monotonic()
            while task_child.poll() is None:
                if service_child is not None and service_child.poll() is not None:
                    raise RuntimeError('C50 model service exited during ' + phase)
                if time.monotonic() - start > timeout:
                    raise TimeoutError('Bounded phase timeout: ' + phase)
                state['updated_utc'] = now()
                write(output / 'status.json', state)
                time.sleep(10)
            code = task_child.returncode
            task_child = None
            if code:
                raise RuntimeError(f'{name} exited {code}; evidence retained in {name}.log')

    try:
        update()
        release_completed_c75(output)
        before_launches = set((ROOT / 'logs').glob('navigation_dart50_service_*/launch.json'))
        with (output / 'model_service.log').open('x', encoding='utf-8') as stream:
            service_child = subprocess.Popen([PYTHON, '-B', '-u', 'checks/serve_navigation_dart50.py',
                '--run', '--keep-ratio', '0.5', '--profile', 'navigation', '--c75-report', str(args.c75_report.resolve())],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        update(status='loading_c50', c50_model_started=True, service_pid=service_child.pid)
        deadline = time.monotonic() + 600
        while True:
            if service_child.poll() is not None:
                raise RuntimeError('C50 service exited before readiness')
            try:
                with urlopen(API, timeout=3) as response:
                    listing = json.load(response)
                if [row['id'] for row in listing['data']] != [MODEL]:
                    raise ValueError('Unexpected model identity on port23333')
                break
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('C50 service readiness timeout')
                time.sleep(3)
        launches = set((ROOT / 'logs').glob('navigation_dart50_service_*/launch.json')) - before_launches
        if len(launches) != 1:
            raise ValueError('C50 service launch evidence is ambiguous')
        service = launches.pop().parent
        validate_service(service)
        update(service_directory=service.relative_to(ROOT).as_posix())
        traces = []
        for sample in (0, 1):
            before = {p.name for p in service.glob('prefill_*.json')}
            command([PYTHON, '-B', '-u', 'checks/check_navigation_dart50_request.py', '--run',
                     '--keep-ratio', '0.5', '--profile', 'navigation', '--sample-index', str(sample)],
                    f'smoke_{sample}', f'smoke_sample_{sample}', 300)
            added = [p for p in service.glob('prefill_*.json') if p.name not in before]
            if len(added) != 1:
                raise ValueError('Each C50 smoke request must create exactly one prefill trace')
            trace = audit_trace(added[0])
            if (trace['image_count'], trace['original_length'], trace['retained_length']) != (
                    (2, 701, 401) if sample == 0 else (9, 2815, 1465)):
                raise ValueError('Actual C50 smoke token counts differ from frozen input budget')
            traces.append(trace)
        write(output / 'smoke_audit.json', {'passed': True, 'traces': traces}, exclusive=True)
        single = output / 'single_1250'
        single.mkdir(exist_ok=False)
        before = sorted(p.name for p in service.glob('prefill_*.json'))
        write(single / 'runtime_before.json', before, exclusive=True)
        command(['bash', 'scripts/run_habitat_task_python.sh', '-u', 'checks/run_simplevln_diagnostic.py',
                 '--policy', 'memory', '--manifest', 'setup/navigation-dart50-single-20261108.json',
                 '--output', str(single)], 'single_1250', 'single_navigation_1250', 3600)
        single_audit = audit_batch(single, service, before)
        write(output / 'single_audit.json', single_audit, exclusive=True)
        evidence = {'service_directory': service.relative_to(ROOT).as_posix(),
                    'single_episode_evidence': single.relative_to(ROOT).as_posix(),
                    'single_episode_runtime_audit': (output / 'single_audit.json').relative_to(ROOT).as_posix(),
                    'nine_image_trace': traces[1]['trace_id'], 'c75_report': c75['path']}
        write(ROOT / 'setup/navigation-dart50-evidence-20261108.json', evidence, exclusive=True)
        command([PYTHON, '-B', '-u', 'checks/run_navigation_dart50_150.py', '--prepare'],
                'freeze150', 'freezing_same150', 1200)
        before_runs = set((ROOT / 'logs').glob('navigation_dart150_keep50_*'))
        update(c50_navigation_started=True)
        command([PYTHON, '-B', '-u', 'checks/run_navigation_dart50_150.py', '--run'],
                'navigation150', 'running_same150')
        added_runs = set((ROOT / 'logs').glob('navigation_dart150_keep50_*')) - before_runs
        if len(added_runs) != 1:
            raise ValueError('C50 formal run directory is ambiguous')
        run = added_runs.pop()
        report = read(run / 'report.json')
        if not (report['status'] == 'complete' and report['completed_episode_count'] == 150
                and report['all_runtime_audits_passed'] is True and report['error_count'] == 0):
            raise ValueError('C50 final evidence did not establish completed150')
        update(c50_run_directory=run.relative_to(ROOT).as_posix())
        command([PYTHON, '-B', 'checks/export_navigation_dart50_compare.py',
                 '--c75-report', str(args.c75_report.resolve()), '--c50-run', str(run),
                 '--output', str(output / 'comparison.json')], 'comparison', 'exporting_comparison', 600)
        update(status='complete', completed_utc=now(), c50_completed_count=150,
               comparison=str((output / 'comparison.json').relative_to(ROOT)))
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        stop_owned(task_child)
        stop_owned(service_child)
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()


if __name__ == '__main__':
    main()
