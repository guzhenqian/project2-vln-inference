"""Post-run CPU preprocessing and real-weight projection diagnostics, sequential."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

from navigation_stage2_common import ROOT, GROUPS, read, write, digest
from check_navigation_awq_dart_request import prepare_input


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    run=args.run.resolve()
    assert run.is_relative_to(ROOT/'logs') and read(run/'status.json')['status']=='complete'
    assert not (run/'bottleneck_probes.json').exists()
    import torch
    from lmdeploy.vl.model.qwen3 import Qwen3VLModel
    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
    from lmdeploy.pytorch.kernels.cuda.awq_kernels import awq_linear
    from safetensors import safe_open
    processor=Qwen3VLModel(str(ROOT/GROUPS['fp16']['model']),backend='pytorch')
    processor.build_preprocessor()
    cpu=[]
    original_threads=torch.get_num_threads()
    # Baseline default and a diagnostic lower-thread-count ablation; no serving setting changed.
    for threads in dict.fromkeys([original_threads,4]):
        torch.set_num_threads(threads)
        for sample in (0,1):
            _,_,_,_,encoded=prepare_input(1.,sample,'navigation')
            messages=json.loads(encoded)['messages']
            rows=[]
            for i in range(8):
                start=time.perf_counter()
                converted=asyncio.run(VLAsyncEngine.async_convert_to_pil_images(copy.deepcopy(messages)))
                converted_at=time.perf_counter()
                processed=processor.preprocess(converted)
                end=time.perf_counter()
                tensors=[x['pixel_values'] for x in processed[-1]['content']]
                actual=torch.cat(tensors,dim=0).contiguous()
                expected=read(run/'00_audited_fp16/service'/f'input_{sample}.json')['pixel_values']
                assert list(actual.shape)==expected['shape'] and str(actual.dtype)==expected['dtype']
                assert hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest()==expected['sha256']
                if i>=3:
                    rows.append(dict(decode_images_ms=(converted_at-start)*1000,image_preprocess_ms=(end-converted_at)*1000))
            cpu.append(dict(threads=threads,sample=sample,n=len(rows),measurements=rows,
                medians={k:statistics.median(r[k] for r in rows) for k in rows[0]},pixels_match_runtime=True))
            print('CPU_PREPROCESS',threads,sample,cpu[-1]['medians'],flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    def unpack(packed):
        return torch.stack([(packed>>shift)&15 for shift in [0,16,4,20,8,24,12,28]],-1).reshape(packed.shape[0],-1)
    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        events=[]
        begin_wall=time.perf_counter()
        for _ in range(30):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();fn();b.record()
            events.append((a,b))
        torch.cuda.synchronize()
        return dict(cuda_interval_median_ms=statistics.median(a.elapsed_time(b) for a,b in events),
            mean_wall_ms=(time.perf_counter()-begin_wall)*1000/30)
    kernels=[]
    fused={'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
        'o':['self_attn.o_proj'],'gate_up':['mlp.gate_proj','mlp.up_proj'],'down':['mlp.down_proj']}
    with torch.inference_mode():
        for group in ('awq','gptq'):
            source=ROOT/GROUPS[group]['model']/'model.safetensors'
            with safe_open(str(source),framework='pt',device='cpu') as reader:
                for label,parts in fused.items():
                    tensors=[]
                    for suffix in ('qweight','qzeros','scales'):
                        tensors.append(torch.cat([reader.get_tensor('model.language_model.layers.0.'+part+'.'+suffix) for part in parts],dim=1).cuda())
                    qw,qz,scales=tensors
                    dense=((unpack(qw)-unpack(qz).repeat_interleave(128,0)).half()*scales.repeat_interleave(128,0)).contiguous()
                    dense_weight=dense.T.contiguous()
                    for rows in (1,701,2815):
                        x=torch.randn(rows,dense.shape[0],dtype=torch.float16,device='cuda')*.1
                        quant=lambda:awq_linear(x,qw,scales,qz)
                        fp16=lambda:torch.nn.functional.linear(x,dense_weight)
                        q,ref=quant(),fp16()
                        err=float((q.float()-ref.float()).norm()/ref.float().norm())
                        assert err<.01 and torch.isfinite(q).all()
                        times={}
                        for pass_index in range(2):
                            for name,fn in ([('fp16',fp16),('int4',quant)] if pass_index==0 else [('int4',quant),('fp16',fp16)]):
                                times.setdefault(name,[]).append(bench(fn))
                        kernels.append(dict(group=group,layer=0,projection=label,rows=rows,
                            in_features=dense.shape[0],out_features=dense.shape[1],relative_l2=err,measurements=times,
                            median_ms={k:statistics.median(t['cuda_interval_median_ms'] for t in v) for k,v in times.items()}))
                        print('PROJECTION',group,label,rows,kernels[-1]['median_ms'],flush=True)
                    del tensors,qw,qz,scales,dense,dense_weight,x,q,ref
    result=dict(status='complete',cpu_preprocessing=cpu,projection_microbenchmarks=kernels,
        sources_sha256={Path(__file__).name:digest(Path(__file__))},default_cpu_threads=original_threads,
        scope='Separate attribution probes. Real layer0 fused projection weights versus the same dequantized FP16 matrix; hot isolated matrices do not represent full-model cache/memory traffic. No model weights or serving configs changed.',
        profiler_limit='CUPTI initialization failed in WSL; CUDA kernel activity trace unavailable. CUDA events and HTTP timers remain measured; do not infer activation bandwidth saturation from these probes.')
    write(run/'bottleneck_probes.json',result)


if __name__=='__main__':
    main()
