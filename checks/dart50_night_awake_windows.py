"""Temporary system sleep prevention for this authorized overnight sequence."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
run = args.run_dir.resolve()
if not run.is_relative_to(root / 'logs') or not run.is_dir():
    raise ValueError('Choose the current sequence directory inside project logs')
kernel = ctypes.windll.kernel32
kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
kernel.SetThreadExecutionState.restype = ctypes.c_uint
if not kernel.SetThreadExecutionState(0x80000001):
    raise RuntimeError('Temporary system wake request failed')
reason = 'bounded_16hours'
started = time.monotonic()
try:
    (run / 'wake_guard.json').write_text(json.dumps({
        'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Temporary system sleep prevention; display may sleep; maximum16hours',
    }, indent=2), encoding='utf-8')
    while time.monotonic() - started < 16 * 3600:
        try:
            state = json.loads((run / 'status.json').read_text(encoding='utf-8'))
            if state['status'] in ('complete', 'failed', 'cancelled'):
                reason = state['status']
                break
        except (OSError, ValueError):
            pass
        time.sleep(30)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    (run / 'wake_guard_finished.json').write_text(json.dumps({
        'reason': reason, 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'temporary_wake_request_released': True,
    }, indent=2), encoding='utf-8')
