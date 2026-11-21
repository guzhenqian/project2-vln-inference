"""Isolated M=1 GEMV screening; root schedules GPU execution explicitly.

Preserves every attempted configuration. No service/adapter/table is installed.
Tests existing native, saved wide GEMM, and this independent GEMV branch against
FP16-rounded dequantized weights with FP32 products/accumulation.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from navigation_stage2_common import ROOT, GROUPS, digest, write

PROJECTIONS = {'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
               'o':['self_attn.o_proj'], 'gate_up':['mlp.gate_proj','mlp.up_proj'],
               'down':['mlp.down_proj']}
FROZEN = {'awq':'31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
          'gptq':'2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=tuple(FROZEN), default='awq')
    parser.add_argument('--baseline-table', type=Path, required=True)
    parser.add_argument('--wide-table', type=Path, required=True)
    parser.add_argument('--projections', nargs='+', choices=tuple(PROJECTIONS), default=list(PROJECTIONS))
    parser.add_argument('--max-candidates', type=int, default=72)
    parser.add_argument('--search-budget-seconds', type=float, default=600.)
    parser.add_argument('--repeats', type=int, default=12)
    parser.add_argument('--confirm-repeats', type=int, default=40)
    parser.add_argument('--cold-repeats', type=int, default=20)
    parser.add_argument('--cold-cache-mib', type=int, default=96)
    parser.add_argument('--cold-min-gain', type=float, default=.03)
    args = parser.parse_args()
    if not 1 <= args.max_candidates <= 72 or min(args.repeats,args.confirm_repeats,args.cold_repeats) < 2:
        parser.error('Require 1..72 candidates and at least two repetitions')
    if args.search_budget_seconds <= 0 or args.cold_cache_mib < 64 or not 0 <= args.cold_min_gain < 1:
        parser.error('Invalid budget/cache/advance threshold')
    if len(set(args.projections)) != len(args.projections):
        parser.error('Duplicate projections')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'logs') or out.exists():
        parser.error('Require a fresh output directory below logs')
    out.mkdir(parents=True)
    attempts, measurements, validations, boundaries = [], [], [], []
    selected, flush_count, cache_flush = {}, 0, None
    locks = []
    started = time.perf_counter()
    try:
        import fcntl
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock',ROOT/'tmp/navigation_repair_gpu.lock'):
            lock=path.open('a+')
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException: lock.close();raise
            locks.append(lock)
        import torch
        from safetensors import safe_open
        import navigation_w4a16_repair_kernels as native
        import navigation_latency_w4_kernels_20261120 as wide
        import navigation_latency_w4_gemv_20261120 as gemv
        torch.set_num_threads(4)
        torch.manual_seed(20261120)
        torch.backends.cuda.matmul.allow_tf32 = False
        model = ROOT/GROUPS[args.group]['model']/'model.safetensors'
        model_sha = digest(model)  # streaming hash, before all timed intervals
        assert model_sha == FROZEN[args.group], 'Checkpoint differs from frozen identity'
        wide_table = json.loads(args.wide_table.read_text(encoding='utf-8'))
        assert wide_table['algorithm'] == args.group and wide_table['model_weights_sha256'] == model_sha
        assert wide_table['scope']=='exact_m_only' and wide_table['numerical_gate_passed']
        wide_configs={(v['k'],v['n']):v['config'] for v in wide_table['shapes'] if v['m']==1}
        assert len(wide_configs)==4, 'Wide reference needs all four M1 layouts'
        meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),algorithm=args.group,
            model_weights_sha256=model_sha,checkpoint_hash='Streaming SHA256 before GPU timing',
            baseline_table=str(args.baseline_table.resolve()),baseline_table_sha256=digest(args.baseline_table),
            wide_table=str(args.wide_table.resolve()),wide_table_sha256=digest(args.wide_table),
            search_layer=0,m=1,projections=args.projections,max_candidates=args.max_candidates,
            search_budget_seconds=args.search_budget_seconds,repeats=args.repeats,
            confirm_repeats=args.confirm_repeats,cold_repeats=args.cold_repeats,
            cold_cache_mib=args.cold_cache_mib,target_l2_mib=48,cold_min_gain=args.cold_min_gain,
            validation_layers=[0,18,35],input_magnitudes=[.1,1.],relative_l2_limit=.005,seed=20261120,
            reference='round_fp16((uint4_weight-uint4_zero)*FP16_scale), then FP32 x@weight with TF32 disabled, finally FP16 output',
            split='Independent FP32 partial[split,N] followed by FP32 reduction and FP16 cast; no atomic accumulation',
            sources_sha256={name:digest(ROOT/'checks'/name) for name in (
                'probe_navigation_latency_w4_gemv_20261120.py','navigation_latency_w4_gemv_20261120.py',
                'navigation_w4a16_repair_kernels.py','navigation_latency_w4_kernels_20261120.py',
                'navigation_w4a16_candidate_kernels.py')},
            native_awq_helper_source=dict(path=gemv._unpack_weight.fn.__code__.co_filename,
                sha256=digest(gemv._unpack_weight.fn.__code__.co_filename)),
            gpu=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                     sm_count=torch.cuda.get_device_properties(0).multi_processor_count),
            scope='Independent M1 projection experiment only; no runtime candidate table or service integration')
        write(out/'protocol.json',meta)
        cache_flush=torch.zeros(args.cold_cache_mib*1048576,device='cuda',dtype=torch.uint8)

        def load(label,layer):
            with safe_open(str(model),framework='pt',device='cpu') as reader:
                tensors=tuple(torch.cat([reader.get_tensor(f'model.language_model.layers.{layer}.{part}.{suffix}')
                    for part in PROJECTIONS[label]],dim=1).contiguous() for suffix in ('qweight','scales','qzeros'))
            sha={name:dict(sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                          dtype=str(value.dtype),shape=list(value.shape))
                 for name,value in zip(('qweight','scales','qzeros'),tensors)}
            return tuple(t.cuda() for t in tensors),sha

        def unpack(value):
            return torch.stack([(value>>shift)&15 for shift in (0,16,4,20,8,24,12,28)],-1).reshape(value.shape[0],-1)

        def dense_reference(qw,s,qz):
            # One explicit FP16 rounding before returning a FP32 reference.
            return ((unpack(qw)-unpack(qz).repeat_interleave(128,0)).float()*
                    s.repeat_interleave(128,0).float()).half().float().T.contiguous()

        def check(actual,reference):
            torch.cuda.synchronize()
            delta=actual.float()-reference.float()
            relative=float(delta.norm()/reference.float().norm().clamp_min(1.e-8))
            maximum=float(delta.abs().max())
            finite=bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
            return dict(relative_l2=relative if math.isfinite(relative) else None,
                        max_absolute_error=maximum if math.isfinite(maximum) else None,
                        finite=finite,passed=finite and math.isfinite(relative) and relative<.005)

        def timed(fn,count,cold=False):
            nonlocal flush_count
            for _ in range(3): fn()
            torch.cuda.synchronize()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): fn()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream): output=fn()
            graph.replay();torch.cuda.synchronize()
            events=[]
            for _ in range(count):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                if cold: cache_flush.add_(1);flush_count+=1
                begin.record();graph.replay();end.record();events.append((begin,end))
            torch.cuda.synchronize()
            samples=[a.elapsed_time(b) for a,b in events]
            assert all(math.isfinite(v) and v>0 for v in samples),'Invalid CUDA timing samples'
            result=dict(samples_ms=samples,cuda_p50_ms=statistics.median(samples),mode='cuda_graph_replay',
                        cache_condition='eviction_before_each_sample' if cold else 'hot',
                        eviction_buffer_bytes=cache_flush.numel() if cold else 0,
                        flush_gpu_ms_in_kernel_timing=False)
            del output,graph
            return result

        def save_attempt(value):
            attempts.append(value)
            with (out/'attempts.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

        def fatal(error):
            return any(text in str(error).lower() for text in
                       ('illegal memory','device-side assert','unspecified launch','misaligned address'))

        with torch.inference_mode():
            native.load_launch_table(args.baseline_table,args.group,model_sha)
            for label in PROJECTIONS:
                (qw,s,qz),_=load(label,0);native.prepare_projection(qw,s,qz)
                del qw,s,qz;gc.collect();torch.cuda.empty_cache()
            write(out/'baseline_replayed_table.json',native.freeze(args.group,model_sha))
            search_start=time.perf_counter()
            for index,label in enumerate(args.projections):
                (qw,s,qz),tensor_sha=load(label,0);k,n=qw.shape[0],s.shape[1]
                dense=dense_reference(qw,s,qz)
                inputs=[torch.randn((1,k),device='cuda',dtype=torch.float16)*a for a in (.1,1.)]
                refs=[torch.nn.functional.linear(x.float(),dense).half() for x in inputs]
                baselines=dict(native=lambda x:native.precision_linear(x,qw,s,qz),
                               wide=lambda x:wide.launch(x,qw,s,qz,wide_configs[(k,n)]))
                baseline_numeric={name:[check(fn(x),ref) for x,ref in zip(inputs,refs)] for name,fn in baselines.items()}
                assert all(v['passed'] for values in baseline_numeric.values() for v in values),baseline_numeric
                remaining=max(0.,args.search_budget_seconds-(time.perf_counter()-search_start))
                deadline=time.perf_counter()+remaining/max(1,len(args.projections)-index)
                accepted=[]
                for ci,config in enumerate(gemv.candidates(args.max_candidates)):
                    if ci and time.perf_counter()>deadline: break
                    attempt=dict(projection=label,m=1,k=k,n=n,config=config,
                                 schedule=gemv.schedule_description(k,n,config))
                    try:
                        actual,compiled=gemv.launch(inputs[0],qw,s,qz,config,return_metadata=True)
                        numeric=[check(actual,refs[0]),check(gemv.launch(inputs[1],qw,s,qz,config),refs[1])]
                        attempt.update(numerical=numeric,compiled=compiled)
                        if all(v['passed'] for v in numeric):
                            attempt['timing']=timed(lambda:gemv.launch(inputs[0],qw,s,qz,config),args.repeats)
                            accepted.append(attempt)
                        else: attempt['status']='rejected_numerically'
                    except Exception as error:
                        attempt.update(status='candidate_error',error=repr(error))
                        if fatal(error): save_attempt(attempt);raise
                    save_attempt(attempt)
                if not accepted:
                    measurements.append(dict(projection=label,m=1,k=k,n=n,status='no_usable_candidate',
                                             baseline_numerical=baseline_numeric))
                    write(out/'measurements.json',measurements,exclusive=False)
                    del qw,s,qz,dense,inputs,refs;gc.collect();torch.cuda.empty_cache()
                    continue
                winner=min(accepted,key=lambda v:v['timing']['cuda_p50_ms']);config=winner['config'];selected[label]=config
                fns={name:(lambda fn=fn:fn(inputs[0])) for name,fn in baselines.items()}
                fns['gemv']=lambda:gemv.launch(inputs[0],qw,s,qz,config)
                confirm={condition:{name:[] for name in fns} for condition in ('hot','cold')}
                rng=random.Random(20261120+index)
                for condition in confirm:
                    for _ in range(2):
                        order=list(fns);rng.shuffle(order)
                        for name in order:
                            confirm[condition][name].append(timed(fns[name],args.cold_repeats if condition=='cold'
                                                                 else args.confirm_repeats,condition=='cold'))
                medians={condition:{name:statistics.median(x for block in blocks for x in block['samples_ms'])
                                    for name,blocks in implementations.items()} for condition,implementations in confirm.items()}
                reductions={condition:{name:1-values['gemv']/values[name] for name in ('native','wide')}
                            for condition,values in medians.items()}
                # Predeclared screening rule only, never statistical proof.
                promising=all(v>=args.cold_min_gain for v in reductions['cold'].values()) and all(v>0 for v in reductions['hot'].values())
                measurement=dict(projection=label,m=1,k=k,n=n,status='confirmed',tensors=tensor_sha,
                    selected_config=config,selection_basis='Hot search winner; cold is independent confirmation',
                    baseline_numerical=baseline_numeric,compiled=winner['compiled'],confirmation=confirm,
                    confirmation_p50_ms=medians,reduction_fraction=reductions,
                    promising_for_whole_model=promising,advance_rule='Beats both native and saved wide in hot; cold gain >= protocol threshold against both')
                measurements.append(measurement);write(out/'measurements.json',measurements,exclusive=False)
                print(json.dumps({key:measurement[key] for key in ('projection','confirmation_p50_ms','promising_for_whole_model')}),flush=True)
                del qw,s,qz,dense,inputs,refs;gc.collect();torch.cuda.empty_cache()
            # Validate every selected shape even when it was slower, preserving
            # numerical evidence separately from the performance stop rule.
            for layer in (0,18,35):
                for label,config in selected.items():
                    (qw,s,qz),tensor_sha=load(label,layer);k,n=qw.shape[0],s.shape[1]
                    dense=dense_reference(qw,s,qz)
                    for amplitude in (.1,1.):
                        x=torch.randn((1,k),device='cuda',dtype=torch.float16)*amplitude
                        ref=torch.nn.functional.linear(x.float(),dense).half()
                        functions=dict(native=lambda:native.precision_linear(x,qw,s,qz),
                            wide=lambda:wide.launch(x,qw,s,qz,wide_configs[(k,n)]),
                            gemv=lambda:gemv.launch(x,qw,s,qz,config))
                        for implementation,fn in functions.items():
                            value=dict(layer=layer,projection=label,m=1,k=k,n=n,input_magnitude=amplitude,
                                implementation=implementation,tensors=tensor_sha,config=config if implementation=='gemv'
                                else wide_configs[(k,n)] if implementation=='wide' else None,**check(fn(),ref))
                            validations.append(value);write(out/'numerical_validation.json',validations,exclusive=False)
                            assert value['passed'],value
                    del qw,s,qz,dense,x,ref;gc.collect();torch.cuda.empty_cache()
            # Synthetic tails: K=384/BK256 and split8 leave masked groups and
            # empty split CTAs; N=40 exercises packed/output tail columns.
            for k,n in (((128,8),(384,40),(9728,40)) if selected else ()):
                qw=torch.randint(-(2**31),2**31-1,(k,n//8),device='cuda',dtype=torch.int32)
                qz=torch.randint(-(2**31),2**31-1,(k//128,n//8),device='cuda',dtype=torch.int32)
                s=(torch.rand((k//128,n),device='cuda',dtype=torch.float16)*.1).clamp_min(.001)
                x=torch.randn((1,k),device='cuda',dtype=torch.float16)
                dense=dense_reference(qw,s,qz);ref=torch.nn.functional.linear(x.float(),dense).half()
                configs=list({json.dumps(c,sort_keys=True):c for c in list(selected.values())+
                    [dict(block_n=32,block_k=256,split_k=8,num_warps=4,compiler_stages=1)]}.values())
                for config in configs:
                    value=dict(k=k,n=n,m=1,config=config,**check(gemv.launch(x,qw,s,qz,config),ref))
                    boundaries.append(value);write(out/'boundary_checks.json',boundaries,exclusive=False)
                    assert value['passed'],value
                del qw,s,qz,x,dense,ref;gc.collect();torch.cuda.empty_cache()
            checksum=int(cache_flush.sum(dtype=torch.int64).item());expected=(flush_count%256)*cache_flush.numel()
            assert checksum==expected,'Cold eviction checksum differs'
            assert len(validations)==len(selected)*3*2*3,'Incomplete real-layer numerical validation'
            for name,sha in meta['sources_sha256'].items():
                assert digest(ROOT/'checks'/name)==sha,'Source changed during probe: '+name
            helper=meta['native_awq_helper_source']
            assert digest(helper['path'])==helper['sha256'],'Native unpack source changed during probe'
            promising=[v['projection'] for v in measurements if v.get('promising_for_whole_model')]
            complete=len(selected)==len(args.projections)
            status='candidate_for_full_model' if promising and complete else 'no_latency_advantage' if complete else 'incomplete_no_usable_candidate'
            write(out/'selection.json',dict(scope='diagnostic_only_not_a_runtime_table',selected=selected,
                numerical_all_selected_passed=True,promising_projections=promising,service_integration_authorized=False))
            write(out/'result.json',dict(status=status,protocol=meta,elapsed_seconds=time.perf_counter()-started,
                attempts=len(attempts),attempt_status_counts=dict(Counter(a.get('status','numerical_and_timing_passed') for a in attempts)),
                measurements=measurements,numerical_validation=validations,boundary_checks=boundaries,
                cache_flush_evidence=dict(buffer_bytes=cache_flush.numel(),flush_count=flush_count,
                    full_buffer_checksum=checksum,expected_checksum=expected,verified=True,
                    timing_scope='96MiB-or-configured eviction finishes before begin event; both GEMV and partial reduction are timed'),
                selected_all_projections=complete,promising_projections=promising,
                next_action='Root may consider a separately audited hybrid whole-model experiment' if promising and complete else 'Stop this branch; retain failed/slower evidence',
                limitation='Different floating-point reduction order. Isolated projections and synthetic tails do not prove full-model tokens, navigation quality or HTTP gains. No service adapter or launch table was installed.'))
    except BaseException as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            attempts=len(attempts),completed_measurements=len(measurements),elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        for lock in locks:
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    main()
