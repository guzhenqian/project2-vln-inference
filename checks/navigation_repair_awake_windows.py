"""Temporary bounded system-awake request for the active repair experiment."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, required=True)
parser.add_argument('--hours', type=float, default=12)
args = parser.parse_args()
assert 0 < args.hours <= 24 and args.directory.is_dir()
kernel = ctypes.windll.kernel32
kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
kernel.SetThreadExecutionState.restype = ctypes.c_uint
start = time.monotonic()
assert kernel.SetThreadExecutionState(0x80000001)
path = args.directory/f'wake_{os.getpid()}.json'
record = dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
              maximum_hours=args.hours, permanent_setting_change=False, display_may_sleep=True)
path.write_text(json.dumps(record, indent=2), encoding='utf-8')
reason = 'maximum_duration'
try:
    while time.monotonic()-start < args.hours*3600:
        if (args.directory/'release_wake').exists():
            reason = 'released_by_controller'
            break
        time.sleep(30)
finally:
    kernel.SetThreadExecutionState(0x80000000)
    record.update(finished_utc=datetime.now(timezone.utc).isoformat(), reason=reason, released=True)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')
