"""Create a separate backward-compatible LMDeploy config view; preserve original checkpoint."""
import hashlib
import json
import os
import argparse
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant',choices=['SFT','RL'],default='SFT')
args=parser.parse_args()
source=root/('models/SimpleVLN_Qwen3VL_4B_'+args.variant)
assert (source/'source_manifest.json').exists(), 'Download and checksum verification must finish first'
target=root/('models/SimpleVLN_Qwen3VL_4B_'+args.variant+'_lmdeploy')
target.mkdir(exist_ok=True)
for item in source.iterdir():
    if not item.is_file() or item.name.endswith(('.log','.part')) or item.name in ['config.json','tokenizer_config.json']:continue
    dest=target/item.name
    if not dest.exists():os.link(str(item),str(dest))
    assert dest.stat().st_size==item.stat().st_size
config=json.loads((source/'config.json').read_text(encoding='utf8'))
rope=config['text_config'].pop('rope_parameters')
config['text_config']['rope_theta']=rope.pop('rope_theta')
config['text_config']['rope_scaling']=rope
config['vision_config']['model_type']='qwen3_vl'
if args.variant=='RL':
    config['dtype']='bfloat16'
    config['text_config']['dtype']='bfloat16'
    config['vision_config']['dtype']='bfloat16'
(target/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
tokenizer=json.loads((source/'tokenizer_config.json').read_text(encoding='utf8'))
tokenizer['additional_special_tokens']=tokenizer.pop('extra_special_tokens')
tokenizer.pop('backend',None)
tokenizer['tokenizer_class']='Qwen2Tokenizer'
# Replace the serving hard link from an earlier adapter attempt without modifying source.
tokenizer_target=target/'tokenizer_config.json'
if tokenizer_target.exists() and os.path.samefile(str(tokenizer_target),str(source/'tokenizer_config.json')):
    tokenizer_target.unlink()
tokenizer_target.write_text(json.dumps(tokenizer,indent=2),encoding='utf8')
if not (target/'preprocessor_config.json').exists():
    processor=json.loads((source/'processor_config.json').read_text(encoding='utf8'))
    image_config=dict(processor['image_processor'],processor_class='Qwen3VLProcessor')
    (target/'preprocessor_config.json').write_text(json.dumps(image_config,indent=2),encoding='utf8')
(target/'serving_adapter.json').write_text(json.dumps({'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
    'changes':'Transformers5 rope_parameters -> Transformers4 rope_scaling + rope_theta; vision model_type alias; tokenizer extra_special_tokens list -> additional_special_tokens; no tensor/tokenizer vocabulary changes',
    'runtime_precision':'BF16; RL source file is F32 and engine casts at load; no4bit quantization',
    'processor_compatibility':'legacy preprocessor config extracted from nested source image_processor if absent',
    'weight_storage':'hard links to SHA-verified original files; no duplicated large weights'},indent=2),encoding='utf8')
print('SIMPLEVLN_SERVING_VIEW_READY')
