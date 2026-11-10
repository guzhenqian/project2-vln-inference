"""Temporary wake request while the authorized precision study runs; no persistent settings."""
import ctypes
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

run=Path(__file__).resolve().parents[1]/'logs/navigation_precision_sequence_20261110'
assert run.is_dir()
kernel=ctypes.windll.kernel32
kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
kernel.SetThreadExecutionState.restype=ctypes.c_uint
assert kernel.SetThreadExecutionState(0x80000001)
started=time.monotonic()
receipt=run/f'wake_{os.getpid()}.json'
record=dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
            maximum_hours=24, scope='Temporary system wake; display can sleep; no persistent power policy changes')
receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
reason='24h_bound'
failure_since=None
try:
    while time.monotonic()-started < 24*3600:
        state=json.loads((run/'status.json').read_text(encoding='utf-8'))['status']
        if state in ('complete','cancelled'):
            reason=state
            break
        if state=='failed':
            failure_since=failure_since or time.monotonic()
            if time.monotonic()-failure_since>1800:
                reason='failed_unrecovered_30min'
                break
        else:
            failure_since=None
        time.sleep(30)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    record.update(released=True, reason=reason, finished_utc=datetime.now(timezone.utc).isoformat())
    receipt.write_text(json.dumps(record,indent=2),encoding='utf-8')
