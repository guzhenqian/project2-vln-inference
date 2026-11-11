"""CPU-only validation of equal image/token inputs and architecture across published variants."""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig,AutoProcessor

ROOT=Path(__file__).resolve().parents[1]
folders=[ROOT/('models/SimpleVLN_Qwen3VL_4B_'+v+'_lmdeploy') for v in ['SFT','RL']]
configs=[AutoConfig.from_pretrained(str(p),local_files_only=True) for p in folders]
processors=[AutoProcessor.from_pretrained(str(p),local_files_only=True) for p in folders]
for part in ['text_config','vision_config']:
    a=getattr(configs[0],part).to_dict();b=getattr(configs[1],part).to_dict()
    differences={k:(a.get(k),b.get(k)) for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    assert set(differences)<={'transformers_version','_name_or_path'},differences
instruction='Walk out of the bathroom and wait on the stairs that are on the right.'
messages=[{'role':'system','content':'You are a visual language navigation model.'},
          {'role':'user','content':[{'type':'image'},{'type':'text','text':instruction}]}]
texts=[processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True) for processor in processors]
assert texts[0]==texts[1]
image=Image.open(str(ROOT/'logs/simplevln_diagnostic_20261030T090743Z_47822/dev/episode_392/frame_0000.png')).convert('RGB')
features=[processor(text=[text],images=[image],return_tensors='pt') for processor,text in zip(processors,texts)]
assert set(features[0])==set(features[1])
for name in features[0]:assert torch.equal(features[0][name],features[1][name]),name
result={'status':'identical','architecture':'same713 tensor keys/shapes verified separately from safetensors headers',
        'runtime_precision':'BF16 for both; RL F32 source cast at load','input_tensors':{key:list(value.shape) for key,value in features[0].items()},
        'source_revisions':['2b2ae6e30f3e344fb9c6d2ed11ee9c117ac24566','4078dab77f3c93a0711b57549ac54a59c1c1e1ea']}
(ROOT/'tmp/habitat_setup/navigation_source_review/runtime_input_comparison.json').write_text(json.dumps(result,indent=2))
print('SFT_RL_RUNTIME_INPUTS_IDENTICAL',json.dumps(result))
