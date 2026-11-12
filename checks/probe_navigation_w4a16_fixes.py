"""Screen independent W4A16 implementation fixes on real frozen projections."""
import argparse
import gc
import json
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from navigation_stage2_common import ROOT, GROUPS, read, write, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', nargs='+', choices=['awq', 'gptq'], default=['awq', 'gptq'])
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ROOT/'logs') and not out.exists()
    out.mkdir(parents=True)
    import torch
    from safetensors import safe_open
    from navigation_w4a16_candidate_kernels import candidate_linear, VARIANTS
    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False

    def unpack(packed):
        return torch.stack([(packed >> shift) & 15 for shift in
                            [0,16,4,20,8,24,12,28]], -1).reshape(packed.shape[0], -1)

    def timed(fn, repeats=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        events = []
        start = time.perf_counter()
        for _ in range(repeats):
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record()
            events.append((a,b))
        torch.cuda.synchronize()
        return dict(cuda_interval_ms=statistics.median(a.elapsed_time(b) for a,b in events),
                    wall_mean_ms=1000*(time.perf_counter()-start)/repeats)

    def replay_timed(fn):
        # Compilation and allocation warmup precede capture. No persistent FP16 model cache.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = fn()
        graph.replay()
        torch.cuda.synchronize()
        result = timed(graph.replay, repeats=40)
        del captured, graph
        return result

    rows, failures = [], []
    meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), groups=args.groups,
        variants=list(VARIANTS), source_sha256={name:digest(ROOT/'checks'/name) for name in
            ('probe_navigation_w4a16_fixes.py','navigation_w4a16_candidate_kernels.py')},
        scope='Real layer0 fused weights; same quantized matrix represented densely for FP16 reference. Hot microbench only, not full model speed or navigation quality.',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,temperature.gpu',
                                      '--format=csv,noheader'],text=True).strip())
    write(out/'protocol.json',meta)
    fused={'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
           'o':['self_attn.o_proj'], 'gate_up':['mlp.gate_proj','mlp.up_proj'], 'down':['mlp.down_proj']}
    try:
        with torch.inference_mode():
            for group in args.groups:
                with safe_open(str(ROOT/GROUPS[group]['model']/'model.safetensors'),framework='pt',device='cpu') as reader:
                    for label,parts in fused.items():
                        qw,qz,s=[torch.cat([reader.get_tensor('model.language_model.layers.0.'+part+'.'+suffix)
                                           for part in parts],dim=1).cuda() for suffix in ('qweight','qzeros','scales')]
                        dense=((unpack(qw)-unpack(qz).repeat_interleave(128,0)).half()
                               * s.repeat_interleave(128,0)).T.contiguous()
                        for m in (1,701,2815):
                            x=torch.randn(m,qw.shape[0],device='cuda',dtype=torch.float16)*.1
                            reference=torch.nn.functional.linear(x,dense)
                            functions={'fp16_reference':lambda:torch.nn.functional.linear(x,dense)}
                            functions.update({v:(lambda v=v:candidate_linear(x,qw,s,qz,v)) for v in VARIANTS})
                            item=dict(group=group,layer=0,projection=label,rows=m,k=qw.shape[0],n=s.shape[1],variants={})
                            for variant,fn in functions.items():
                                actual=fn()
                                torch.cuda.synchronize()
                                error=float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1.e-8))
                                assert torch.isfinite(actual).all() and error<.005,(variant,label,m,error)
                                eager=timed(fn)
                                replay=replay_timed(fn)
                                item['variants'][variant]=dict(relative_l2=error,eager=eager,cuda_graph_replay=replay)
                                del actual
                            rows.append(item)
                            write(out/'measurements.json',rows,exclusive=False)
                            print(json.dumps(dict(group=group,projection=label,m=m,
                                timings={k:round(v['eager']['cuda_interval_ms'],4) for k,v in item['variants'].items()})),flush=True)
                            del x,reference,functions
                        del qw,qz,s,dense
                        gc.collect();torch.cuda.empty_cache()
        write(out/'result.json',dict(status='complete',protocol=meta,measurements=rows))
        print('W4A16_CANDIDATE_PROBES_COMPLETE',len(rows),flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed_rows=len(rows)))
        raise


if __name__=='__main__':
    main()
