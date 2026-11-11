"""Lossless INT4 layout conversion: compressed-tensors -> LMDeploy AWQ GEMM."""
import hashlib
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_CT'
VIEW=ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy'
DEST=ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_AWQ'

def pack_awq(value):
    assert value.dtype==torch.int32 and value.ndim==2 and value.shape[1]%8==0
    assert value.min()>=0 and value.max()<=15
    order=[0,2,4,6,1,3,5,7]
    grouped=value.reshape(value.shape[0],-1,8)[:,:,order]
    shifts=torch.arange(8,dtype=torch.int32)*4
    packed=(grouped<<shifts).sum(-1,dtype=torch.int32)
    # Independent bit extraction must recover every original integer.
    recovered=torch.stack([(packed>>int(shift))&15 for shift in shifts],dim=-1)
    inverse=[order.index(i) for i in range(8)]
    assert torch.equal(recovered[:,:,inverse].reshape(value.shape),value)
    return packed.contiguous()

def main():
    torch.set_num_threads(4)
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'model.safetensors').exists(),'Do not overwrite quantized weights'
    state={}
    for file in sorted(SOURCE.glob('*.safetensors')):
        with safe_open(str(file),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                assert key not in state
                state[key]=reader.get_tensor(key)
    prefixes=sorted(k[:-len('.weight_packed')] for k in state if k.endswith('.weight_packed'))
    config=json.loads((VIEW/'config.json').read_text())
    assert len(prefixes)==config['text_config']['num_hidden_layers']*7
    output={};checks=[]
    for prefix in prefixes:
        assert '.language_model.layers.' in prefix and '.visual.' not in prefix
        shape=tuple(state[prefix+'.weight_shape'].tolist())
        scales=state[prefix+'.weight_scale']
        assert scales.shape==(shape[0],shape[1]//128)
        assert torch.isfinite(scales).all() and (scales>0).all()
        signed=unpack_from_int32(state[prefix+'.weight_packed'],4,shape).to(torch.int32)
        zeros=unpack_from_int32(state[prefix+'.weight_zero_point'],4,scales.shape,packed_dim=0).to(torch.int32)
        qweight=pack_awq((signed+8).t().contiguous())
        qzeros=pack_awq((zeros+8).t().contiguous())
        output[prefix+'.qweight']=qweight
        output[prefix+'.qzeros']=qzeros
        deployed_scales=scales.to(torch.float16)
        assert torch.isfinite(deployed_scales).all() and (deployed_scales>0).all()
        output[prefix+'.scales']=deployed_scales.t().contiguous()
        checks.append(dict(layer=prefix,shape=shape,integer_roundtrip=True,scale_dtype=str(scales.dtype),
            fp16_scale_max_relative_rounding=float(((deployed_scales.float()-scales.float()).abs()/scales.float()).max())))
    excluded={suffix for suffix in ['weight_packed','weight_shape','weight_scale','weight_zero_point','weight_g_idx']}
    for key,tensor in state.items():
        if key.rsplit('.',1)[-1] in excluded:continue
        assert not ('quantization' in key)
        output[key]=(tensor.to(torch.float16) if tensor.is_floating_point() else tensor).contiguous()
    unchanged=[]
    with safe_open(str(VIEW/'model.safetensors'),framework='pt',device='cpu') as original:
        original_keys=set(original.keys())
        for key,tensor in output.items():
            if '.visual.' in key or '.embed_tokens.' in key or key.startswith('lm_head.') or key.endswith(('.q_norm.weight','.k_norm.weight')):
                source_key=key
                if key=='lm_head.weight' and key not in original_keys:
                    assert config['text_config'].get('tie_word_embeddings',config.get('tie_word_embeddings',False))
                    source_key='model.language_model.embed_tokens.weight'
                assert source_key in original_keys, key
                assert torch.equal(tensor,original.get_tensor(source_key).to(torch.float16)),key
                unchanged.append(key)
    # Compressor materializes tied output weights; deployment retains source tying.
    if 'lm_head.weight' in output and 'lm_head.weight' not in original_keys:
        assert torch.equal(output['lm_head.weight'],output['model.language_model.embed_tokens.weight'])
        del output['lm_head.weight']
    save_file(output,str(DEST/'model.safetensors'),metadata={'format':'pt'})
    for file in VIEW.iterdir():
        if file.is_file() and file.suffix in ['.json','.jinja'] and file.name not in ['config.json','serving_adapter.json','model.safetensors.index.json']:
            shutil.copyfile(str(file),str(DEST/file.name))
    quant=dict(quant_method='awq',version='gemm',bits=4,group_size=128,zero_point=True)
    config['quantization_config']=quant;config['text_config']['quantization_config']=quant
    for cfg in [config,config['text_config'],config['vision_config']]:cfg['dtype']='float16';cfg['torch_dtype']='float16'
    (DEST/'config.json').write_text(json.dumps(config,indent=2))
    digest=hashlib.sha256()
    with (DEST/'model.safetensors').open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    report=dict(status='all quantized integers and zero-points roundtrip exactly; FP32 scales rounded to deployment FP16 with error reported',
        conversion='integer layout only, no second integer quantization; unquantized weights/scales cast to runtime FP16',layers=checks,quantized_layer_count=len(prefixes),
        bytes=(DEST/'model.safetensors').stat().st_size,sha256=digest.hexdigest(),
        source='models/SimpleVLN_Qwen3VL_4B_RL_AWQ_CT',unchanged_fp16_tensors=unchanged)
    (DEST/'conversion_audit.json').write_text(json.dumps(report,indent=2))
    print('AWQ_LAYOUT_VERIFIED',len(prefixes),report['bytes'],report['sha256'],flush=True)

if __name__=='__main__':main()
