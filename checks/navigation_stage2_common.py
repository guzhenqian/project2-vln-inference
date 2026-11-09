"""Names and immutable recipe choices for the authorized stage2 run."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'logs/navigation_stage2_sequence_20261109'
PLAN = ROOT / 'data/baseline/navigation_stage2_20261109'
SOURCE = 'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
AWQ = 'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104'
GPTQ = 'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_stage2_20261109'
GROUPS = {
    'fp16': dict(method='fp16', keep=1.0, model=SOURCE, prune_layer=None),
    'awq': dict(method='awq', keep=1.0, model=AWQ, prune_layer=None),
    'gptq': dict(method='gptq', keep=1.0, model=GPTQ, prune_layer=None),
    'fastv75': dict(method='fastv', keep=.75, model=SOURCE, prune_layer=2),
    'fastv50': dict(method='fastv', keep=.5, model=SOURCE, prune_layer=2),
    'sparsevlm75': dict(method='sparsevlm', keep=.75, model=SOURCE, prune_layer=3),
    'sparsevlm50': dict(method='sparsevlm', keep=.5, model=SOURCE, prune_layer=3),
}
SOURCES = ['serve_navigation_stage2.py', 'navigation_stage2_common.py',
    'navigation_stage2_runtime.py', 'navigation_stage2_selection.py',
    'navigation_stage2_module_map.py', 'navigation_dart_kv_layout.py',
    'navigation_awq_dart_guard.py', 'navigation_awq_dart_profiles.py', 'navigation_dart_profiles.py']


def model_name(group):
    if group not in GROUPS:
        raise ValueError('Unknown stage2 group')
    return 'SimpleVLN-Qwen3VL-4B-RL-stage2-' + group + '-20261109'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, exclusive=True):
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()
