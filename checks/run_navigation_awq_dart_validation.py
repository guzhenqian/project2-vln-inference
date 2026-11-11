"""Authorized D75 GPU acceptance: native AWQ100 controls, D75 smokes, full1250.

Successful validation leaves only its own D75 service ready for the formal150.
Failures stop owned children and preserve every log; no other service is killed.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from urllib.request import urlopen

from navigation_dart50_support import ROOT, read, write, digest
from navigation_awq_dart_preflight import require_c_finished
from navigation_awq_dart_profiles import served_model_name
from audit_navigation_awq_dart_run import audit_launch, audit_runtime_init, audit_trace, audit_batch
from run_navigation_dart50_night import processes, stop_owned

PYTHON = str(Path.home() / '.venvs/project2/bin/python')
OUTPUT = ROOT / 'logs/navigation_awq_dart75_sequence_20261109'
API = 'http://127.0.0.1:23333/v1/models'

def main():
    require_c_finished()
    assert not OUTPUT.exists(), 'Existing validation evidence must not be overwritten'
    active = [pid for pid, item in processes().items() if item['cwd'] == str(ROOT)
              and any(Path(word).name in ('run_simplevln_diagnostic.py', 'resume_overnight_batch.py',
                  'run_navigation_dart50_150.py', 'resume_navigation_dart150.py') for word in item['words'])]
    assert not active, 'Existing evaluator still active: ' + str(active)
    OUTPUT.mkdir()
    state = {'status': 'starting_validation', 'pid': os.getpid(), 'started_utc': now(),
             'target_keep_ratio': .75, 'formal150_started': False}
    service_child = task_child = None
    passed = False
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LIBGL_ALWAYS_SOFTWARE='1')
    env.pop('DISPLAY', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)

    def update(**values):
        state.update(values, updated_utc=now())
        write(OUTPUT / 'status.json', state)
        print('AWQ_DART_VALIDATION', state['status'], flush=True)

    def command(argv, name, timeout=3600):
        nonlocal task_child
        update(status=name, command=argv)
        with (OUTPUT / (name + '.log')).open('x', encoding='utf-8') as stream:
            task_child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
        update(active_child_pid=task_child.pid)
        deadline = time.monotonic() + timeout
        while task_child.poll() is None:
            if service_child is not None and service_child.poll() is not None:
                raise RuntimeError('Owned model service exited during ' + name)
            if time.monotonic() > deadline:
                raise TimeoutError(name)
            time.sleep(5)
        code = task_child.returncode
        task_child = None
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; original evidence retained')

    try:
        update()
        services, smokes = {}, {}
        for keep in (1., .75):
            label = 'keep100' if keep == 1 else 'keep75'
            before = set((ROOT / 'logs').glob('navigation_awq_dart_service_*/launch.json'))
            with (OUTPUT / (label + '_service.log')).open('x', encoding='utf-8') as stream:
                service_child = subprocess.Popen([PYTHON, '-B', '-u', 'checks/serve_navigation_awq_dart.py',
                    '--run', '--keep-ratio', str(keep), '--profile', 'navigation'], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            update(status='loading_' + label, service_pid=service_child.pid, service_keep_ratio=keep)
            deadline = time.monotonic() + 600
            while True:
                if service_child.poll() is not None:
                    raise RuntimeError(label + ' service exited before readiness')
                try:
                    with urlopen(API, timeout=3) as response:
                        listing = json.load(response)
                    assert [row['id'] for row in listing['data']] == [served_model_name(keep)]
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('Service startup: ' + label)
                    time.sleep(3)
            added = set((ROOT / 'logs').glob('navigation_awq_dart_service_*/launch.json')) - before
            assert len(added) == 1
            service = added.pop().parent
            launch = audit_launch(service, keep)
            audit_runtime_init(service, keep)
            for name, expected in launch['sources_sha256'].items():
                assert digest(ROOT / 'checks' / name) == expected, name
            services[label] = service.relative_to(ROOT).as_posix()
            update(service_directory=services[label])
            traces = []
            for sample in (0, 1):
                old = {path.name for path in service.glob('prefill_*.json')}
                command([PYTHON, '-B', '-u', 'checks/check_navigation_awq_dart_request.py', '--run',
                    '--keep-ratio', str(keep), '--profile', 'navigation', '--sample-index', str(sample)],
                    f'{label}_sample{sample}', 300)
                fresh = [path for path in service.glob('prefill_*.json') if path.name not in old]
                assert len(fresh) == 1
                trace = audit_trace(fresh[0], keep_ratio=keep)
                expected = ((2, 701, 701), (9, 2815, 2815)) if keep == 1 else ((2, 701, 551), (9, 2815, 2140))
                assert (trace['image_count'], trace['original_length'], trace['retained_length']) == expected[sample]
                traces.append(trace)
            smokes[label] = {'passed': True, 'service': services[label], 'traces': traces}
            write(OUTPUT / (label + '_smoke_audit.json'), smokes[label], exclusive=True)
            if keep == 1:
                update(status='releasing_own_keep100')
                stop_owned(service_child)
                service_child = None
                time.sleep(3)
        single = OUTPUT / 'single_1250'
        single.mkdir()
        before = sorted(path.name for path in service.glob('prefill_*.json'))
        write(single / 'runtime_before.json', before, exclusive=True)
        command(['bash', 'scripts/run_habitat_task_python.sh', '-u', 'checks/run_simplevln_diagnostic.py',
                 '--policy', 'memory', '--manifest', 'setup/navigation-awq-dart-single-20261108.json',
                 '--output', str(single)], 'single_1250', 3600)
        audit = audit_batch(single, service, before)
        write(OUTPUT / 'single_audit.json', audit, exclusive=True)
        evidence = {'service_directory': services['keep75'], 'service_pid': service_child.pid,
                    'single_episode_evidence': single.relative_to(ROOT).as_posix(),
                    'single_episode_runtime_audit': (OUTPUT / 'single_audit.json').relative_to(ROOT).as_posix(),
                    'nine_image_trace': smokes['keep75']['traces'][1]['trace_id'],
                    'native_control_service': services['keep100'],
                    'validation_directory': OUTPUT.relative_to(ROOT).as_posix()}
        write(ROOT / 'setup/navigation-awq-dart75-evidence-20261109.json', evidence, exclusive=True)
        update(status='validated_ready', evidence=evidence, gpu_combination_verified=True)
        passed = True
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        stop_owned(task_child)
        if not passed:
            stop_owned(service_child)

def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

if __name__ == '__main__':
    main()
