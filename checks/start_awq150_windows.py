"""Keep Windows awake only while the authorized overnight AWQ evaluation runs."""
import ctypes
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    kernel = ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001):
        raise RuntimeError('Temporary system wake request failed')
    try:
        result = subprocess.run(['wsl.exe', '--exec', 'python3', 'checks/run_overnight_awq150.py'], cwd=str(ROOT))
        raise SystemExit(result.returncode)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
