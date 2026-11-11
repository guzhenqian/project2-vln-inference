"""Bounded temporary wake request for the authorized cumulative100 stage."""
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
run = root / 'logs/navigation_stage2_sequence_20261109'
start_file, end_file = run / 'rescreen_wake_started.json', run / 'rescreen_wake_finished.json'
assert run.is_dir() and not start_file.exists() and not end_file.exists()
kernel = ctypes.windll.kernel32
kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
kernel.SetThreadExecutionState.restype = ctypes.c_uint
if not kernel.SetThreadExecutionState(0x80000001):
    raise RuntimeError('Temporary system wake request failed')
reason, start = 'bounded_16hours', time.monotonic()
try:
    with start_file.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
            scope='Temporary system wake only; display may sleep; maximum16hours; no persistent power settings'), stream, indent=2)
    while time.monotonic() - start < 16 * 3600:
        try:
            state = json.loads((run / 'status.json').read_text(encoding='utf-8'))['status']
            if state in ('complete', 'failed', 'cancelled') or state.endswith('_failed'):
                reason = state
                break
        except (OSError, ValueError):
            pass
        time.sleep(30)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    with end_file.open('x', encoding='utf-8') as stream:
        json.dump(dict(reason=reason, finished_utc=datetime.now(timezone.utc).isoformat(),
            temporary_wake_request_released=True), stream, indent=2)
