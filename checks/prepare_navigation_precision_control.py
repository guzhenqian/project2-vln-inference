"""Prespecify an FP16 control because LMDeploy0.11.1 forces AWQ activations to FP16."""
import json
from pathlib import Path
from prepare_navigation_quant_data import frozen_json
ROOT = Path(__file__).resolve().parents[1]
profile = json.loads((ROOT/'setup/navigation-compression-profile.json').read_text())
profile['version'] = 'navigation_compression_v2_fp16_control'
profile['precision_control_reason'] = 'Installed lmdeploy/pytorch/config.py:_update_torch_dtype unconditionally selects float16 for AWQ. Add unquantized FP16 control to separate activation precision from weight quantization.'
profile['first_experiment']['targets'] = 'language decoder Linear W4A16; visual/deepstack/embed/head/norms stay FP16 at runtime; original BF16 candidate remains historical reference'
profile['precision_control'] = {}
for key, folder in [('A_diagnostic','compare'),('B_confirmation','confirmation')]:
    m = json.loads((ROOT/('data/baseline/simplevln_4b_rl_'+folder+'/manifest.json')).read_text())
    control = dict(m,version='navigation_fp16_v1_'+key,model='SimpleVLN-Qwen3VL-4B-RL-FP16',dtype='FP16',
        scope='Unquantized FP16 matched control on frozen '+key,
        backend='LMDeploy0.11.1 PyTorch; float16; context4096; cache0.35')
    path = 'data/baseline/navigation_fp16_v1_'+key+'/manifest.json'
    frozen_json(ROOT/path,control);profile['precision_control'][key]=path
profile['evaluation_precision_note'] = 'v1 AWQ manifest dtype W4A16 is accurate; its descriptive BF16 exception text is superseded here: engine forces ALL nonquantized parameters/activations to FP16. No policy or task changes.'
frozen_json(ROOT/'setup/navigation-compression-v2-profile.json',profile)
print('FP16_CONTROL_FROZEN')
