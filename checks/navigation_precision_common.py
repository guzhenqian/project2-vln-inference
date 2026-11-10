"""Independent experiment identity; reuses the exact frozen case roster."""
import json
from pathlib import Path
from navigation_stage2_common import ROOT, PLAN, SOURCES as ORIGINAL_SOURCES, read, write, digest

RUN = ROOT/'logs/navigation_precision_sequence_20261110'
CONFIG = ROOT/'setup/navigation-precision-20261110.json'
GROUPS = read(CONFIG)['groups']
for key, value in GROUPS.items():
    value.update(method=value['algorithm'], keep=1.)
SOURCES = list(ORIGINAL_SOURCES) + [
    'navigation_precision_runtime.py', 'navigation_precision_module_map.py',
    'navigation_precision_common.py', 'navigation_precision_guard.py',
    'navigation_precision_kernels.py', 'serve_navigation_precision.py']


def model_name(group):
    return 'SimpleVLN-Qwen3VL-4B-RL-precision-'+group
