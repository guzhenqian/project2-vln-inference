"""Terminate only a named project navigation service started by this workflow."""
import argparse
import os
import signal
import time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
assert args.model in ['SimpleVLN-Qwen3VL-4B-RL','SimpleVLN-Qwen3VL-4B-RL-FP16','SimpleVLN-Qwen3VL-4B-RL-AWQ']
found=[]
for item in Path('/proc').iterdir():
    if not item.name.isdigit():continue
    try:words=(item/'cmdline').read_bytes().decode().split('\0')
    except (OSError,UnicodeDecodeError):continue
    if '--model-name' in words and words[words.index('--model-name')+1]==args.model and 'api_server' in words and any('SimpleVLN_Qwen3VL_4B_RL' in w for w in words):
        os.kill(int(item.name),signal.SIGTERM);found.append(int(item.name))
print('STOP_REQUESTED',args.model,found,flush=True)
for _ in range(50):
    if not any(Path('/proc',str(pid)).exists() for pid in found):break
    time.sleep(.2)
