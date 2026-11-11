"""Local overnight dependency watcher; no GPU/HTTP until audited C75 completes."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
import traceback

from navigation_dart50_support import ROOT, read, write, require_c75_complete
from run_navigation_dart50_night import processes

OUTPUT = ROOT / 'logs/navigation_dart50_night_20261108'
PYTHON = str(Path.home() / '.venvs/project2/bin/python')


def now():
    return datetime.now(timezone.utc).isoformat()


def candidates():
    paths = []
    for directory in (ROOT / 'logs').glob('*dart*keep75*'):
        if directory.is_dir():
            paths.extend(directory.glob('*report*.json'))
            paths.extend(directory.glob('recovery*/*report*.json'))
    paths.extend((ROOT / 'reports').glob('*dart*150*.json'))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def active_c75_work():
    result = []
    for pid, record in processes().items():
        script = next((Path(word).name for word in record['words'] if word.endswith('.py')), '')
        if record['cwd'] == str(ROOT) and (
                script in ('run_simplevln_diagnostic.py', 'resume_overnight_batch.py',
                           'resume_navigation_run.py', 'run_navigation_dart150.py', 'resume_navigation_dart150.py')
                or ('dart' in script and 'recover' in script)):
            result.append(pid)
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    lock = ROOT / 'logs/navigation_dart50_wait.active.lock'
    token = f'{os.getpid()}-{time.time_ns()}'
    write(lock, {'pid': os.getpid(), 'token': token, 'output': str(OUTPUT)}, exclusive=True)
    started = time.monotonic()
    try:
        previous = read(OUTPUT / 'status.json')
        if previous.get('c50_model_started') or previous.get('c50_navigation_started'):
            raise ValueError('C50 sequence already began; do not duplicate it')
        while True:
            selected, rejected = None, []
            for path in candidates():
                try:
                    require_c75_complete(path)
                    selected = path
                    break
                except (OSError, ValueError, KeyError) as error:
                    rejected.append({'path': str(path.relative_to(ROOT)), 'reason': str(error)})
            active = active_c75_work() if selected else []
            if selected and not active:
                break
            write(OUTPUT / 'status.json', {'status': 'waiting_c75', 'watcher_pid': os.getpid(),
                  'updated_utc': now(), 'c50_model_started': False, 'c50_navigation_started': False,
                  'candidate_report': str(selected) if selected else None,
                  'active_c75_pids': active, 'recent_candidates': rejected[:5],
                  'note': 'Wait for exact150 complete C75 report + all runtime audits; no GPU/HTTP calls yet'})
            if time.monotonic() - started > 12 * 3600:
                raise TimeoutError('C75 completion dependency remained unmet for12hours; retained all evidence')
            time.sleep(30)
        with (OUTPUT / 'sequence.log').open('x', encoding='utf-8') as stream:
            child = subprocess.Popen([PYTHON, '-B', '-u', 'checks/run_navigation_dart50_night.py',
                '--c75-report', str(selected), '--output', str(OUTPUT)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write(OUTPUT / 'watch_dispatch.json', {'utc': now(), 'sequence_pid': child.pid,
              'c75_report': str(selected.relative_to(ROOT))}, exclusive=True)
        code = child.wait()
        if code:
            print('DART50_SEQUENCE_NEEDS_REPAIR', code, flush=True)
        else:
            print('DART50_OVERNIGHT_COMPLETE', flush=True)
        return code
    except BaseException as error:
        write(OUTPUT / 'status.json', {'status': 'failed', 'updated_utc': now(),
              'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc(),
              'c50_model_started': False, 'c50_navigation_started': False})
        raise
    finally:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
