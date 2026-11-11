"""Continue passed D75 GPU validation through exact150 and the ABCD report.

Exclusive, no automatic retries. Only the previously created and verified D75
service process and its recorded descendants may be released on exit.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from urllib.request import urlopen

from navigation_dart50_support import ROOT, read, write
from navigation_awq_dart_profiles import served_model_name
from run_navigation_dart50_night import processes, same_process, stop_owned, now

PYTHON = str(Path.home() / '.venvs/project2/bin/python')
OUTPUT = ROOT / 'logs/navigation_awq_dart75_sequence_20261109'
EVIDENCE = ROOT / 'setup/navigation-awq-dart75-evidence-20261109.json'
COMPARISON = ROOT / 'reports/navigation_abcd_same150_20261109.json'


def main():
    if os.name != 'posix':
        raise RuntimeError('Run in the existing WSL environment')
    state, evidence = read(OUTPUT / 'status.json'), read(EVIDENCE)
    if not (state['status'] == 'validated_ready' and state['gpu_combination_verified'] is True
            and state['evidence'] == evidence and not state['formal150_started']):
        raise RuntimeError('Expected passed unused GPU validation; inspect existing evidence instead of rerunning')
    inventory = processes()
    service_pid = evidence['service_pid']
    owner = inventory[service_pid]
    words = owner['words']
    if not (owner['cwd'] == str(ROOT) and 'checks/serve_navigation_awq_dart.py' in words
            and words[words.index('--keep-ratio') + 1] == '0.75'
            and words[words.index('--profile') + 1] == 'navigation'):
        raise RuntimeError('Original D75 service PID identity changed')
    with urlopen('http://127.0.0.1:23333/v1/models', timeout=15) as response:
        if [item['id'] for item in json.load(response)['data']] != [served_model_name(.75)]:
            raise RuntimeError('Original D75 endpoint identity changed')
    descendants = {service_pid}
    while True:
        more = {pid for pid, item in inventory.items() if item['ppid'] in descendants}
        if more <= descendants:
            break
        descendants |= more
    owned = {pid: inventory[pid] for pid in descendants}
    write(OUTPUT / 'formal_service_ownership.json', owned, exclusive=True)
    claim = OUTPUT / 'formal_sequence_claim.json'
    write(claim, {'pid': os.getpid(), 'started_utc': now(), 'automatic_retry': False}, exclusive=True)
    child = None
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LIBGL_ALWAYS_SOFTWARE='1')
    env.pop('DISPLAY', None)

    def update(**values):
        state.update(values, updated_utc=now())
        write(OUTPUT / 'status.json', state)
        print('AWQ_DART_SEQUENCE', state['status'], flush=True)

    def command(argv, label):
        nonlocal child
        update(status=label, command=argv)
        with (OUTPUT / (label + '.log')).open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        update(active_child_pid=child.pid)
        while child.poll() is None:
            if not same_process(service_pid, owner):
                raise RuntimeError('Validated service exited during ' + label)
            time.sleep(5)
        code, child = child.returncode, None
        if code:
            raise RuntimeError(f'{label} exited {code}; preserve logs and inspect without automatic rerun')

    try:
        command([PYTHON, '-B', 'checks/run_navigation_awq_dart150.py', '--prepare'], 'formal_prepare')
        command([PYTHON, '-B', 'checks/run_navigation_awq_dart150.py', '--check-only'], 'formal_check')
        before = set((ROOT / 'logs').glob('navigation_awq_dart150_keep75_*/status.json'))
        update(formal150_started=True, formal_started_utc=now(), formal_controller_pid=os.getpid())
        command([PYTHON, '-B', '-u', 'checks/run_navigation_awq_dart150.py', '--run'], 'formal150')
        added = set((ROOT / 'logs').glob('navigation_awq_dart150_keep75_*/status.json')) - before
        if len(added) != 1:
            raise RuntimeError('Expected exactly one new formal150 run')
        run = added.pop().parent
        update(formal_run=run.relative_to(ROOT).as_posix())
        command([PYTHON, '-B', 'checks/export_navigation_abcd150.py', '--d-run', str(run),
                 '--output', str(COMPARISON)], 'exporting_abcd')
        update(status='releasing_own_service', comparison=COMPARISON.relative_to(ROOT).as_posix())
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        stop_owned(child)
        # Match every PID's recorded start time; never kill by a process name.
        for sig, seconds in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            targets = [service_pid] if sig == signal.SIGINT else list(owned)
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
        remaining = [pid for pid, snap in owned.items() if same_process(pid, snap)]
        write(OUTPUT / 'formal_service_release.json', {'utc': now(), 'owned_pids': list(owned),
                                                      'remaining_owned_pids': remaining}, exclusive=True)
        if remaining:
            update(status='failed', cleanup_error='Owned service processes remain: ' + str(remaining))
        elif state['status'] == 'releasing_own_service':
            update(status='complete', finished_utc=now())
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
