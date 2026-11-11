"""Windows launcher: temporary awake request for the lifetime of the evaluator."""
import ctypes
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001):
        raise RuntimeError('Cannot request temporary system wake lock')
    try:
        subprocess.run(['wsl.exe', '-e', 'python3', 'checks/run_overnight_baseline.py'] + sys.argv[1:],
                       cwd=str(ROOT), check=True)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
