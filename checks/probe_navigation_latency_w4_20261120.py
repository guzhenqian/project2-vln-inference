"""Bounded wider W4A16 launch search; root alone schedules GPU execution.

Outputs retain all attempted candidates, numerical checks, baseline timings and
exact selected configurations. Descriptive hot timings are not full-model gains.
"""
import argparse
import gc
import hashlib
import json
import random
import shutil
import statistics
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from navigation_stage2_common import ROOT, GROUPS, digest, write


PROJECTIONS = {'qkv': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
    'o': ['self_attn.o_proj'], 'gate_up': ['mlp.gate_proj', 'mlp.up_proj'],
    'down': ['mlp.down_proj']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group', choices=['awq', 'gptq'], default='awq')
    p.add_argument('--baseline-table', type=Path, required=True)
    p.add_argument('--rows', nargs='+', type=int, default=[1, 701, 2815])
    p.add_argument('--projections', nargs='+', choices=list(PROJECTIONS), default=list(PROJECTIONS))
    p.add_argument('--max-candidates', type=int, default=32)
    p.add_argument('--repeats', type=int, default=12)
    p.add_argument('--confirm-repeats', type=int, default=40)
    p.add_argument('--cold-repeats', type=int, default=20)
    p.add_argument('--cold-cache-mib', type=int, default=96,
                   help='GPU read/write eviction buffer, strictly larger than this RTX4070Ti 48MiB L2; excluded from kernel CUDA timing.')
    p.add_argument('--search-budget-seconds', type=float, default=900)
    p.add_argument('--boundary-rows', nargs='+', type=int, default=[2, 16, 17, 33, 65, 713, 1737, 3001, 3072])
    p.add_argument('--validation-layers', nargs='+', type=int, default=[0, 18, 35])
    args = p.parse_args()
    if args.repeats < 2 or args.confirm_repeats < 2 or args.cold_repeats < 2 or args.max_candidates < 2:
        p.error('At least two candidates/repetitions required')
    if args.cold_cache_mib < 64:
        p.error('Cold-cache buffer must be at least 64MiB, larger than the target GPU 48MiB L2')
    if len(set(args.rows)) != len(args.rows) or any(m < 1 or m > 3072 for m in args.rows + args.boundary_rows):
        p.error('Unique search M in 1..3072 required')
    if len(set(args.validation_layers)) != len(args.validation_layers) or any(v < 0 or v > 35 for v in args.validation_layers):
        p.error('Unique validation layers in 0..35 required')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'logs') or out.exists():
        p.error('Require a fresh output directory under logs')
    out.mkdir(parents=True)
    import torch
    from safetensors import safe_open
    import navigation_w4a16_repair_kernels as baseline
    import navigation_latency_w4_kernels_20261120 as wide
    from run_navigation_w4a16_fixes import FROZEN
    torch.set_num_threads(4)
    torch.manual_seed(20261120)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = ROOT/GROUPS[args.group]['model']
    model_sha = digest(model/'model.safetensors')
    assert model_sha == FROZEN[args.group], 'Checkpoint differs from frozen model'
    meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), algorithm=args.group,
        model_weights_sha256=model_sha, baseline_table=str(args.baseline_table.resolve()),
        baseline_table_sha256=digest(args.baseline_table), layers=[0], rows=args.rows,
        projections=args.projections, max_candidates=args.max_candidates,
        repeats=args.repeats, confirm_repeats=args.confirm_repeats,
        cold_repeats=args.cold_repeats, cold_cache_mib=args.cold_cache_mib,
        target_l2_mib=48, cold_cache_scope='Separate evidence; not mixed with hot samples or used to silently replace hot winner',
        search_budget_seconds=args.search_budget_seconds, boundary_rows=args.boundary_rows,
        validation_layers=args.validation_layers, validation_input_magnitudes=[.1,1.0],
        relative_l2_limit=.005, seed=20261120,
        scope='Same frozen quantized matrices, wider native launch choices. Exact-M hot projection timing; not navigation quality or service latency.',
        sources_sha256={name:digest(ROOT/'checks'/name) for name in
            ['navigation_latency_w4_kernels_20261120.py','probe_navigation_latency_w4_20261120.py',
             'navigation_w4a16_repair_kernels.py','navigation_w4a16_candidate_kernels.py']},
        gpu=subprocess.check_output([shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,memory.used,temperature.gpu',
                                      '--format=csv,noheader'],text=True).strip())
    write(out/'protocol.json', meta)
    attempts, rows, shapes, boundaries, numerical_validation = [], [], [], [], []
    cache_flush = torch.zeros(args.cold_cache_mib*1024*1024, device='cuda', dtype=torch.uint8)
    cache_flush_count = 0

    def load(label, layer=0, with_sha=False):
        with safe_open(str(model/'model.safetensors'),framework='pt',device='cpu') as reader:
            cpu_tensors = tuple(torch.cat([reader.get_tensor(f'model.language_model.layers.{layer}.'+part+'.'+suffix)
                 for part in PROJECTIONS[label]],dim=1).contiguous() for suffix in ['qweight','scales','qzeros'])
        sha = {name:dict(sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                        dtype=str(value.dtype),shape=list(value.shape))
               for name,value in zip(['qweight','scales','qzeros'],cpu_tensors)} if with_sha else None
        gpu_tensors=tuple(value.cuda() for value in cpu_tensors)
        return (gpu_tensors,sha) if with_sha else gpu_tensors

    def unpack(t):
        return torch.stack([(t >> shift) & 15 for shift in [0,16,4,20,8,24,12,28]],-1).reshape(t.shape[0],-1)

    def check(actual, ref):
        torch.cuda.synchronize()
        error=float((actual.float()-ref.float()).norm()/ref.float().norm().clamp_min(1.e-8))
        ok=bool(torch.isfinite(actual).all()) and error < .005
        return dict(relative_l2=error, passed=ok)

    def timed(fn, count, graph_mode=False, cold_cache=False):
        nonlocal cache_flush_count
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        captured=None
        graph=None
        if graph_mode:
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                captured=fn()
            fn=graph.replay
            fn()
            torch.cuda.synchronize()
        events=[]
        started=time.perf_counter()
        for _ in range(count):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            if cold_cache:
                # A real full-buffer GPU read-modify-write, not an unobserved
                # allocation or host-only cache manipulation. The begin event
                # is ordered AFTER this kernel on the current CUDA stream.
                cache_flush.add_(1)
                cache_flush_count += 1
            a.record(); fn(); b.record()
            events.append((a,b))
        torch.cuda.synchronize()
        samples=[a.elapsed_time(b) for a,b in events]
        result=dict(cuda_p50_ms=statistics.median(samples),cuda_min_ms=min(samples),
                    samples_ms=samples,
                    mode='cuda_graph_replay' if graph_mode else 'eager',
                    cache_condition='eviction_before_each_sample' if cold_cache else 'hot')
        if cold_cache:
            result.update(eviction_buffer_bytes=cache_flush.numel(),
                          flush_gpu_ms_in_kernel_timing=False,
                          wall_mean_including_flush_ms=(time.perf_counter()-started)*1000/count)
        else:
            result['wall_mean_ms']=(time.perf_counter()-started)*1000/count
        del graph,captured
        return result

    def save_attempt(attempt):
        attempts.append(attempt)
        with (out/'attempts.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(attempt,ensure_ascii=False)+'\n')

    try:
        with torch.inference_mode():
            baseline.load_launch_table(args.baseline_table,args.group,model_sha)
            # Old table validation requires all four layouts, even if this run
            # deliberately screens only a subset of them.
            for label in PROJECTIONS:
                qw,s,qz=load(label)
                baseline.prepare_projection(qw,s,qz)
                del qw,s,qz
                gc.collect(); torch.cuda.empty_cache()
            receipt=baseline.freeze(args.group,model_sha)
            write(out/'baseline_replayed_table.json',receipt)
            budget_start=time.perf_counter()
            shape_count=len(args.projections)*len(args.rows)
            shape_index=0
            for label in args.projections:
                qw,s,qz=load(label)
                dense=((unpack(qw)-unpack(qz).repeat_interleave(128,0)).half()*s.repeat_interleave(128,0)).T.contiguous()
                selected={}
                for m in args.rows:
                    remaining=max(0,args.search_budget_seconds-(time.perf_counter()-budget_start))
                    slice_seconds=remaining/max(1,shape_count-shape_index)
                    shape_index+=1
                    started=time.perf_counter()
                    x=torch.randn((m,qw.shape[0]),device='cuda',dtype=torch.float16)*.1
                    ref=torch.nn.functional.linear(x,dense)
                    basefn=lambda:baseline.precision_linear(x,qw,s,qz)
                    basenum=check(basefn(),ref)
                    assert basenum['passed'], ('baseline',label,m,basenum)
                    base_before=timed(basefn,args.repeats,m==1)
                    options=wide.candidates(m,qw.shape[0],limit=args.max_candidates)
                    accepted=[]
                    for ci,config in enumerate(options):
                        # Guarantee both native-style choices are examined.
                        if ci>=2 and time.perf_counter()-started > slice_seconds:
                            break
                        attempt=dict(projection=label,m=m,k=qw.shape[0],n=s.shape[1],config=config)
                        fn=lambda config=config:wide.launch(x,qw,s,qz,config)
                        try:
                            numerical=check(fn(),ref)
                            attempt['numerical']=numerical
                            if numerical['passed']:
                                attempt['timing']=timed(fn,args.repeats,m==1)
                                accepted.append(attempt)
                            else:
                                attempt['status']='rejected_numerically'
                        except Exception as exc:
                            attempt['status']='candidate_error'
                            attempt['error']=repr(exc)
                            # A device fault poisons the process; never silently
                            # continue through it as a mere unsupported config.
                            if any(v in str(exc).lower() for v in ['illegal memory','device-side assert','unspecified launch']):
                                save_attempt(attempt)
                                raise
                        save_attempt(attempt)
                    if not accepted:
                        raise RuntimeError(f'No numerical candidate for {label} M{m}')
                    winner=min(accepted,key=lambda v:v['timing']['cuda_p50_ms'])
                    config=winner['config']
                    candidatefn=lambda:wide.launch(x,qw,s,qz,config)
                    # Independent longer measurements after selecting, with
                    # randomized baseline/candidate order in two rounds.
                    confirms={'baseline':[],'candidate':[]}
                    rng=random.Random(20261120+shape_index)
                    for _ in range(2):
                        order=['baseline','candidate']; rng.shuffle(order)
                        for name in order:
                            confirms[name].append(timed(basefn if name=='baseline' else candidatefn,
                                                       args.confirm_repeats,m==1))
                    combined={name:statistics.median(v for block in blocks for v in block['samples_ms'])
                              for name,blocks in confirms.items()}
                    cold_confirms={'baseline':[],'candidate':[]}
                    for _ in range(2):
                        order=['baseline','candidate']; rng.shuffle(order)
                        for name in order:
                            cold_confirms[name].append(timed(basefn if name=='baseline' else candidatefn,
                                                            args.cold_repeats,m==1,cold_cache=True))
                    cold_combined={name:statistics.median(v for block in blocks for v in block['samples_ms'])
                                   for name,blocks in cold_confirms.items()}
                    row=dict(projection=label,m=m,k=qw.shape[0],n=s.shape[1],
                             numerical=winner['numerical'],baseline_numerical=basenum,
                             baseline_before=base_before,selected_config=config,
                             attempted_candidates=sum(a['m']==m and a['projection']==label for a in attempts),
                             confirmation=confirms,confirmation_p50_ms=combined,
                             candidate_to_baseline_ratio=combined['candidate']/combined['baseline'],
                             cold_confirmation=cold_confirms,cold_confirmation_p50_ms=cold_combined,
                             cold_candidate_to_baseline_ratio=cold_combined['candidate']/cold_combined['baseline'],
                             selection_basis='hot search winner; cold confirmation separately retained')
                    rows.append(row)
                    shapes.append(dict(m=m,k=qw.shape[0],n=s.shape[1],config=config))
                    selected[m]=config
                    write(out/'measurements.json',rows,exclusive=False)
                    print(json.dumps({k:row[k] for k in ['projection','m','attempted_candidates',
                          'confirmation_p50_ms','candidate_to_baseline_ratio',
                          'cold_confirmation_p50_ms','cold_candidate_to_baseline_ratio']}),flush=True)
                    del x,ref
                # Validate selected configurations on nearby and bucket-edge
                # actual M. They are NOT added to the runtime exact-M table.
                for m in args.boundary_rows:
                    source_m=min(args.rows,key=lambda measured:abs(measured-m))
                    config=selected[source_m]
                    x=torch.randn((m,qw.shape[0]),device='cuda',dtype=torch.float16)*.1
                    ref=torch.nn.functional.linear(x,dense)
                    numerical=check(wide.launch(x,qw,s,qz,config),ref)
                    b=dict(projection=label,m=m,selected_at_m=source_m,config=config,**numerical)
                    boundaries.append(b)
                    assert numerical['passed'],b
                    del x,ref
                del qw,s,qz,dense
                gc.collect();torch.cuda.empty_cache()
            # The search uses layer0 only; validate its selected exact-M table
            # on independent real layers and two activation magnitudes before
            # publishing a table eligible for full-model service experiments.
            selected_by_shape={(v['m'],v['k'],v['n']):v['config'] for v in shapes}
            for layer in args.validation_layers:
                for label in args.projections:
                    (qw,s,qz),tensor_sha=load(label,layer=layer,with_sha=True)
                    dense=((unpack(qw)-unpack(qz).repeat_interleave(128,0)).half()*s.repeat_interleave(128,0)).T.contiguous()
                    for m in args.rows:
                        config=selected_by_shape[(m,qw.shape[0],s.shape[1])]
                        for magnitude in (.1,1.0):
                            x=torch.randn((m,qw.shape[0]),device='cuda',dtype=torch.float16)*magnitude
                            ref=torch.nn.functional.linear(x,dense)
                            actual=wide.launch(x,qw,s,qz,config)
                            numerical=check(actual,ref)
                            row=dict(layer=layer,projection=label,m=m,k=qw.shape[0],n=s.shape[1],
                                     input_magnitude=magnitude,tensors=tensor_sha,config=config,**numerical)
                            numerical_validation.append(row)
                            write(out/'numerical_validation.json',numerical_validation,exclusive=False)
                            assert numerical['passed'],row
                            del x,ref,actual
                    del qw,s,qz,dense
                    gc.collect();torch.cuda.empty_cache()
            # Check the complete buffer, outside every benchmark interval, so
            # the eviction operations have an observed device-side result.
            flush_checksum=int(cache_flush.sum(dtype=torch.int64).item())
            expected_flush_checksum=(cache_flush_count % 256)*cache_flush.numel()
            assert flush_checksum == expected_flush_checksum, 'Eviction buffer mutation verification failed'
            table=dict(scope='exact_m_only',algorithm=args.group,model_weights_sha256=model_sha,
                       numerical_gate_passed=True,shapes=shapes,
                       validation_layers=args.validation_layers,validation_input_magnitudes=[.1,1.0],
                       validation_cases=len(numerical_validation),validation_all_passed=True)
            write(out/'launch_table.json',table)
            write(out/'result.json',dict(status='passed',numerical_gate_passed=True,protocol=meta,
                 search_elapsed_seconds=time.perf_counter()-budget_start,
                 measurements=rows,boundary_checks=boundaries,numerical_validation=numerical_validation,launch_table=table,
                 cache_flush_evidence=dict(buffer_bytes=cache_flush.numel(),flush_count=cache_flush_count,
                    full_buffer_checksum=flush_checksum,expected_checksum=expected_flush_checksum,
                    verified=True,timing_scope='Eviction kernel precedes begin CUDA event; only operation under test is timed'),
                 limitation='Search timings use layer0; selected table has explicit real-layer numerical validation. Hot and cache-evicted kernel timings are distinct and neither proves full-model gains. Full model correctness, latency and restart replay remain required.'))
            print('W4_WIDE_SEARCH_COMPLETE',len(rows),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
                                     completed_rows=len(rows),attempts=len(attempts)))
        raise


if __name__=='__main__':
    main()
