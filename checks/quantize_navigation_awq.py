"""Prespecified multimodal AWQ using llm-compressor in an isolated environment."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.awq.mappings import AWQMapping
from datasets import Dataset

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
CALIB=ROOT/'data/calibration/navigation_awq_v1'

def process_samples():
    processor=AutoProcessor.from_pretrained(str(SOURCE),local_files_only=True)
    samples=json.loads((CALIB/'samples.json').read_text())
    out=CALIB/'processed';out.mkdir(exist_ok=True)
    records=[]
    for index,sample in enumerate(samples):
        messages=copy.deepcopy(sample['messages']); images=[]
        for message in messages:
            for item in message['content']:
                if item['type']=='image':
                    path=ROOT/item['image'];images.append(Image.open(path).convert('RGB'))
                    del item['image']
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        features=processor(text=[text],images=images,return_tensors='pt')
        assert features['input_ids'].shape[1]+512<=4096, (sample['id'],features['input_ids'].shape)
        target=out/('%02d.pt'%index)
        torch.save(dict(features),str(target))
        records.append(dict(id=sample['id'],path=target.relative_to(ROOT).as_posix(),
            sequence_length=features['input_ids'].shape[1],images=len(images),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(records,indent=2))
    print('CALIBRATION_PROCESSED',len(records),'tokens',min(r['sequence_length'] for r in records),max(r['sequence_length'] for r in records),flush=True)
    return processor,records

def collate(batch):
    assert len(batch)==1
    return torch.load(str(ROOT/batch[0]['path']),weights_only=True,map_location='cpu')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output',default='models/SimpleVLN_Qwen3VL_4B_RL_AWQ_CT')
    args=parser.parse_args()
    torch.manual_seed(7);torch.set_num_threads(4)
    processor,records=process_samples()
    if args.prepare_only:return
    dest=ROOT/args.output
    assert not (dest/'config.json').exists(),'Do not overwrite a checkpoint'
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(SOURCE),local_files_only=True,
        torch_dtype=torch.float16,device_map='cpu',attn_implementation='sdpa')
    # Calibrate FP16-rounded source weights using FP32 math: late-layer AWQ trial
    # scales overflowed FP16. This is calibration arithmetic, not a model change.
    model.to(torch.float32)
    model.eval();model.config.use_cache=False
    # Qwen3-VL interpolation reads pos_embed.weight.device before its forward hook.
    # Keep the small, unquantized visual tower resident; decoder stays layer-offloaded.
    from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
    from compressed_tensors.utils import remove_dispatch
    original_dispatch=sequential_pipeline.dispatch_for_sequential
    def dispatch_with_resident_vision(value):
        original_dispatch(value)
        remove_dispatch(value.model.visual)
        value.model.visual.to('cuda')
        return value
    sequential_pipeline.dispatch_for_sequential=dispatch_with_resident_vision
    mappings=[
        AWQMapping('re:.*language_model.layers.*input_layernorm$', ['re:.*self_attn.q_proj$','re:.*self_attn.k_proj$','re:.*self_attn.v_proj$']),
        AWQMapping('re:.*language_model.layers.*post_attention_layernorm$', ['re:.*mlp.gate_proj$','re:.*mlp.up_proj$']),
        AWQMapping('re:.*language_model.layers.*mlp.up_proj$', ['re:.*mlp.down_proj$']),
    ]
    recipe=AWQModifier(targets=['re:.*language_model.layers.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'],
        scheme='W4A16_ASYM', ignore=['lm_head','re:.*visual.*'], mappings=mappings,offload_device=torch.device('cpu'))
    started=time.time()
    oneshot(model=model,processor=processor,recipe=recipe,dataset=Dataset.from_list([{'path':r['path'], 'input_ids':torch.load(str(ROOT/r['path']),weights_only=True)['input_ids'][0].tolist()} for r in records]),
        data_collator=collate,num_calibration_samples=32,batch_size=1,max_seq_length=4096,
        pipeline='sequential',sequential_targets=['Qwen3VLTextDecoderLayer'],
        sequential_offload_device='cpu',output_dir=str(dest),log_dir=str(ROOT/'logs/navigation_awq_calibration'))
    (dest/'project2_quantization.json').write_text(json.dumps(dict(algorithm='AWQ',source=str(SOURCE),
        calibration_sha256=hashlib.sha256((CALIB/'samples.json').read_bytes()).hexdigest(),
        samples=32,elapsed_s=time.time()-started,source_weights_unchanged=True,
        calibration_math='FP32 from FP16-rounded source weights; FP16 search failed at late MLP scale grid; deployment FP16',
        note='GQA v_proj-to-o_proj smoothing omitted due incompatible head dimensions; o_proj still weight-quantized; visual weights excluded'),indent=2))
    print('NAVIGATION_AWQ_CALIBRATED',dest,flush=True)

if __name__=='__main__':main()
