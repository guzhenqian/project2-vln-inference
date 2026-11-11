"""Snapshot core builds and verify task imports; no simulator is constructed."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--save-core', type=Path)
parser.add_argument('--core-before', type=Path)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 6), sys.version
core = {}
for record in (Path(sys.prefix) / 'conda-meta').glob('*.json'):
    data = json.loads(record.read_text())
    name = data.get('name')
    if name in ('python', 'numpy', 'habitat-sim', 'ffmpeg'):
        assert name not in core, 'Duplicate package records: ' + name
        core[name] = {'version': data['version'], 'build': data['build']}
assert set(core) == {'python', 'numpy', 'habitat-sim', 'ffmpeg'}, core
assert core['python']['version'] == '3.6.15'
assert core['numpy']['version'] == '1.19.5'
assert core['habitat-sim']['version'] == '0.1.7'
assert 'headless' in core['habitat-sim']['build']
if args.save_core:
    args.save_core.write_text(json.dumps(core, indent=2))
    print('Core builds recorded:', json.dumps(core))
    sys.exit(0)
assert args.core_before, '--core-before is required for import validation'
assert core == json.loads(args.core_before.read_text()), 'Core package build changed'

import cv2
import gym
import habitat
import habitat_sim
import imageio
import numpy as np
import pickle5
import yaml
import yacs
import torch
from habitat.tasks.nav.nav import DistanceToGoal, SPL, Success
from habitat.tasks.vln.vln import VLNEpisode, VLNTask

assert habitat.__version__ == '0.1.7', habitat.__version__
assert gym.__version__ == '0.17.3', gym.__version__
assert cv2.__version__ == '4.5.5', cv2.__version__
assert imageio.__version__ == '2.9.0', imageio.__version__
assert np.__version__ == '1.19.5', np.__version__
assert torch.__version__ == '1.10.2+cpu', torch.__version__
assert torch.version.cuda is None, 'Task runtime should use CPU-only torch'
assert torch.tensor([1.0, 2.0]).sum().item() == 3.0
source = Path(__file__).resolve().parents[1] / 'tmp/habitat_setup/sources/habitat-lab-0.1.7'
assert source.resolve() in Path(habitat.__file__).resolve().parents
assert cv2.resize(np.zeros((4, 4, 3), dtype=np.uint8), (2, 2)).shape == (2, 2, 3)
print('Habitat-Lab:', habitat.__version__, habitat.__file__)
print('OpenCV:', cv2.__version__, 'Gym:', gym.__version__, 'NumPy:', np.__version__)
print('Task PyTorch:', torch.__version__, '(CPU only)')
print('Metrics imported:', DistanceToGoal.cls_uuid, Success.cls_uuid, SPL.__name__)
print('VLN task imported:', VLNTask.__name__)
print('Core Python, NumPy, Habitat-Sim and FFmpeg builds unchanged.')
print('HABITAT_TASK_IMPORT_READY (imports only; navigation metrics not yet validated)')
