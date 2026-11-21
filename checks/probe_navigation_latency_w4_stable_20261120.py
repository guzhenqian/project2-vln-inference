"""New split-restricted W4 probe with independent repeat-bytewise gates.

Old probe/source/tables stay frozen. This file does not install a service.
GPU execution is explicit and takes the existing exclusive study locks.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback

from navigation_stage2_common import ROOT,GROUPS,digest,write
from navigation_latency_w4_stable_selector_20261120 import SHAPES,shortlist,load_attempts

PROJECTIONS={'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
             'o':['self_attn.o_proj'],'gate_up':['mlp.gate_proj','mlp.up_proj'],'down':['mlp.down_proj']}
FROZEN={'awq':'31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
        'gptq':'2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--group',choices=tuple(FROZEN),default='awq')
    p.add_argument('--baseline-table',required=True,type=Path)
    p.add_argument('--source-wide-table',required=True,type=Path)
    p.add_argument('--decode-policy',choices=('native_split','split_le_2'),default='native_split')
    p.add_argument('--max-candidates',type=int,default=8)
    p.add_argument('--search-budget-seconds',type=float,default=600.)
    p.add_argument('--repeats',type=int,default=12)
    p.add_argument('--cold-search-repeats',type=int,default=8)
    p.add_argument('--confirm-repeats',type=int,default=40)
    p.add_argument('--cold-repeats',type=int,default=20)
    p.add_argument('--stability-repeats',type=int,default=16)
    p.add_argument('--cold-cache-mib',type=int,default=96)
    p.add_argument('--require-native-bytewise',action='store_true',
                   help='Reject M1 outputs that differ bytewise from the native reference, in addition to requiring repeatability.')
    args=p.parse_args()
    if not 1<=args.max_candidates<=64 or min(args.repeats,args.cold_search_repeats,args.confirm_repeats,args.cold_repeats)<2:
        p.error('Invalid candidate/repetition count')
    if args.stability_repeats<8 or args.cold_cache_mib<64 or args.search_budget_seconds<=0:
        p.error('Require at least8 stability repeats, >48MiB cache buffer and positive budget')
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'logs') or out.exists():p.error('Use a fresh output under project logs')
    out.mkdir(parents=True)
    locks=[];attempts=[];measurements=[];numeric=[];validation_stability=[];search_controls=[];selected={}
    flush_counts=Counter();cache=None;started=time.perf_counter()
    try:
        import fcntl
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock',ROOT/'tmp/navigation_repair_gpu.lock'):
            lock=path.open('a+')
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:lock.close();raise
            locks.append(lock)
        import torch
        from safetensors import safe_open
        import navigation_w4a16_repair_kernels as native
        import navigation_latency_w4_kernels_20261120 as wide
        torch.set_num_threads(4);torch.manual_seed(20261120)
        torch.backends.cuda.matmul.allow_tf32=False
        assert torch.cuda.get_device_capability()==(8,9),'This saved native pipeline config is scoped to SM89'
        model=ROOT/GROUPS[args.group]['model']/'model.safetensors'
        model_sha=digest(model);assert model_sha==FROZEN[args.group]
        source_table_path=args.source_wide_table.resolve();source_dir=source_table_path.parent
        source_table=json.loads(source_table_path.read_text(encoding='utf-8'))
        source_result=json.loads((source_dir/'result.json').read_text(encoding='utf-8'))
        source_protocol=json.loads((source_dir/'protocol.json').read_text(encoding='utf-8'))
        assert source_result['status']=='passed' and source_result['launch_table']==source_table and source_table['numerical_gate_passed']
        assert source_result['protocol']==source_protocol
        assert source_protocol['baseline_table_sha256']==digest(args.baseline_table)
        for name,sha in source_protocol['sources_sha256'].items():
            assert digest(ROOT/'checks'/name)==sha,'Source wide probe code changed: '+name
        assert source_table['algorithm']==args.group and source_table['model_weights_sha256']==model_sha
        assert source_table['scope']=='exact_m_only' and not (source_dir/'failure.json').exists()
        source_configs={(v['m'],v['k'],v['n']):v['config'] for v in source_table['shapes']}
        assert set(source_configs)=={(m,k,n) for k,n in SHAPES.values() for m in (1,701,2815)} and len(source_table['shapes'])==12
        baseline_table=json.loads(args.baseline_table.read_text(encoding='utf-8'))
        saved_attempts=load_attempts(source_dir/'attempts.jsonl')
        pools={label:shortlist(saved_attempts,baseline_table,label,args.decode_policy,args.max_candidates) for label in SHAPES}
        names=['probe_navigation_latency_w4_stable_20261120.py','navigation_latency_w4_stable_selector_20261120.py',
               'navigation_latency_w4_kernels_20261120.py','navigation_w4a16_repair_kernels.py','navigation_w4a16_candidate_kernels.py']
        helper_path=wide._jit.fn.__code__.co_filename
        meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),algorithm=args.group,model_weights_sha256=model_sha,
            baseline_table=str(args.baseline_table.resolve()),baseline_table_sha256=digest(args.baseline_table),
            source_wide_table=str(source_table_path),source_wide_table_sha256=digest(source_table_path),
            source_wide_result_sha256=digest(source_dir/'result.json'),source_wide_attempts_sha256=digest(source_dir/'attempts.jsonl'),
            source_wide_protocol_sha256=digest(source_dir/'protocol.json'),source_wide_sources_sha256=source_protocol['sources_sha256'],
            decode_policy=args.decode_policy,require_native_bytewise=args.require_native_bytewise,
            candidate_shortlists=pools,search_layer=0,rows=[1,701,2815],projections=list(SHAPES),
            repeats=args.repeats,cold_search_repeats=args.cold_search_repeats,confirm_repeats=args.confirm_repeats,
            cold_repeats=args.cold_repeats,stability_repeats=args.stability_repeats,cold_cache_mib=args.cold_cache_mib,
            target_l2_mib=48,search_budget_seconds=args.search_budget_seconds,validation_layers=[0,18,35],
            validation_input_magnitudes=[.1,1.],relative_l2_limit=.005,seed=20261120,
            sources_sha256={name:digest(ROOT/'checks'/name) for name in names},
            native_awq_source=dict(path=helper_path,sha256=digest(helper_path)),
            reference='FP16-round dequantized weights, then FP32 x@weight with TF32 disabled, final FP16 output',
            selection='Lowest fresh cold graph-replay P50 among numerically accepted repeat-stable candidates; optional native-bytewise filter. Historical hot timings rank search order only.',
            stability='Fixed x/weights; eager and CUDA graph replay, each hot and96MiB-evicted, repeated full output byte SHA. Global equality required across all four conditions.',
            scope='M1 configurations only are replaced; M701/2815 remain the source wide prefill configs. Old wide M1 is explicitly an unapproved diagnostic reference. Unknown M still requires native fallback in any later service.')
        write(out/'protocol.json',meta)
        cache=torch.zeros(args.cold_cache_mib*1048576,device='cuda',dtype=torch.uint8)

        def load(label,layer):
            with safe_open(str(model),framework='pt',device='cpu') as f:
                values=tuple(torch.cat([f.get_tensor(f'model.language_model.layers.{layer}.{part}.{suffix}')
                    for part in PROJECTIONS[label]],dim=1).contiguous() for suffix in ('qweight','scales','qzeros'))
            tensors={name:dict(sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest(),dtype=str(v.dtype),shape=list(v.shape))
                     for name,v in zip(('qweight','scales','qzeros'),values)}
            return tuple(v.cuda() for v in values),tensors

        def unpack(v):
            return torch.stack([(v>>shift)&15 for shift in (0,16,4,20,8,24,12,28)],-1).reshape(v.shape[0],-1)

        def dense_reference(qw,s,qz):
            return ((unpack(qw)-unpack(qz).repeat_interleave(128,0)).float()*s.repeat_interleave(128,0).float()).half().float().T.contiguous()

        def check(actual,reference):
            torch.cuda.synchronize();d=actual.float()-reference.float()
            error=float(d.norm()/reference.float().norm().clamp_min(1.e-8));maximum=float(d.abs().max())
            finite=bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
            return dict(relative_l2=error if math.isfinite(error) else None,max_absolute_error=maximum if math.isfinite(maximum) else None,
                        finite=finite,passed=finite and math.isfinite(error) and error<.005)

        def evict(purpose):
            cache.add_(1);flush_counts[purpose]+=1

        def capture(fn):
            for _ in range(3):fn()
            torch.cuda.synchronize();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):fn()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=fn()
            graph.replay();torch.cuda.synchronize()
            return graph,output

        def timed(fn,repeats,cold=False):
            graph,output=capture(fn);events=[]
            for _ in range(repeats):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                if cold:evict('timing')
                begin.record();graph.replay();end.record();events.append((begin,end))
            torch.cuda.synchronize();samples=[a.elapsed_time(b) for a,b in events]
            assert all(math.isfinite(v) and v>0 for v in samples)
            value=dict(samples_ms=samples,cuda_p50_ms=statistics.median(samples),mode='cuda_graph_replay',
                cache_condition='eviction_before_each_sample' if cold else 'hot',
                eviction_buffer_bytes=cache.numel() if cold else 0,flush_gpu_ms_in_kernel_timing=False)
            del graph,output
            return value

        def stability(fn):
            conditions=[];all_hashes=[]
            for mode in ('eager','cuda_graph_replay'):
                graph,output=capture(fn) if mode=='cuda_graph_replay' else (None,None)
                for cold in (False,True):
                    hashes=[]
                    for _ in range(args.stability_repeats):
                        if cold:evict('stability')
                        if graph is None:output=fn()
                        else:graph.replay()
                        torch.cuda.synchronize()
                        cpu=output.detach().contiguous().cpu()
                        assert cpu.dtype==torch.float16 and cpu.ndim==2
                        assert bool(torch.isfinite(cpu).all()),'Nonfinite repeat output'
                        hashes.append(hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
                    all_hashes.extend(hashes)
                    conditions.append(dict(mode=mode,cache_condition='eviction_before_each_sample' if cold else 'hot',
                        repeats=args.stability_repeats,output_shape=list(cpu.shape),output_dtype=str(cpu.dtype),
                        output_sha256=hashes,unique_sha256_counts=dict(Counter(hashes)),all_identical=len(set(hashes))==1,
                        eviction_buffer_bytes=cache.numel() if cold else 0))
                del graph,output
            return dict(conditions=conditions,global_unique_sha256_counts=dict(Counter(all_hashes)),
                        all_identical=len(set(all_hashes))==1,first_sha256=all_hashes[0],
                        scope='Repeated bytes on these fixed inputs only; not a proof for all activations or whole-model output.')

        def compare_stability(candidate,baseline):
            return dict(candidate_repeat_stable=candidate['all_identical'],native_repeat_stable=baseline['all_identical'],
                exact_native_bytes=candidate['all_identical'] and baseline['all_identical'] and candidate['first_sha256']==baseline['first_sha256'])

        def save_attempt(value):
            attempts.append(value)
            with (out/'attempts.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')

        def fatal(error):
            return any(v in str(error).lower() for v in ('illegal memory','device-side assert','unspecified launch','misaligned address'))

        with torch.inference_mode():
            native.load_launch_table(args.baseline_table,args.group,model_sha)
            for label in SHAPES:
                (qw,s,qz),_=load(label,0);native.prepare_projection(qw,s,qz)
                del qw,s,qz;gc.collect();torch.cuda.empty_cache()
            write(out/'baseline_replayed_table.json',native.freeze(args.group,model_sha))
            search_start=time.perf_counter()
            for index,label in enumerate(SHAPES):
                (qw,s,qz),tensors=load(label,0);k,n=SHAPES[label];dense=dense_reference(qw,s,qz)
                inputs=[torch.randn((1,k),device='cuda',dtype=torch.float16)*a for a in (.1,1.)]
                references=[torch.nn.functional.linear(x.float(),dense).half() for x in inputs]
                native_suites=[];old_suites=[]
                for amplitude,x,ref in zip((.1,1.),inputs,references):
                    input_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                    fn=lambda:native.precision_linear(x,qw,s,qz)
                    numerical=check(fn(),ref)
                    control=dict(projection=label,input_magnitude=amplitude,input_sha256=input_sha,native_numerical=numerical)
                    search_controls.append(control);write(out/'search_controls.json',search_controls,exclusive=False)
                    assert numerical['passed'],numerical
                    suite=stability(fn);control['native_stability']=suite
                    write(out/'search_controls.json',search_controls,exclusive=False)
                    assert suite['all_identical'],'Native itself is not bytewise stable on the control input'
                    native_suites.append(dict(input_magnitude=amplitude,input_sha256=input_sha,numerical=numerical,stability=suite))
                    old=lambda:wide.launch(x,qw,s,qz,source_configs[(1,k,n)])
                    old_suites.append(dict(input_magnitude=amplitude,input_sha256=input_sha,numerical=check(old(),ref),stability=stability(old),
                                           scope='Unapproved original wide M1 diagnostic reference; never an acceptance baseline'))
                    control['old_wide_diagnostic']=old_suites[-1]
                    write(out/'search_controls.json',search_controls,exclusive=False)
                remaining=max(0.,args.search_budget_seconds-(time.perf_counter()-search_start))
                deadline=time.perf_counter()+remaining/max(1,len(SHAPES)-index);accepted=[]
                for ci,option in enumerate(pools[label]):
                    if ci and time.perf_counter()>deadline:break
                    config=option['config'];attempt=dict(projection=label,m=1,k=k,n=n,**option)
                    try:
                        tests=[]
                        for amplitude,x,ref,base in zip((.1,1.),inputs,references,native_suites):
                            fn=lambda:wide.launch(x,qw,s,qz,config)
                            numerical=check(fn(),ref)
                            if not numerical['passed']:
                                tests.append(dict(input_magnitude=amplitude,input_sha256=base['input_sha256'],numerical=numerical));break
                            suite=stability(fn);relation=compare_stability(suite,base['stability'])
                            tests.append(dict(input_magnitude=amplitude,input_sha256=base['input_sha256'],numerical=numerical,stability=suite,**relation))
                        attempt['checks']=tests
                        reasons=[]
                        if len(tests)!=2 or any(not t['numerical']['passed'] for t in tests):reasons.append('numerical_failed')
                        if any('stability' in t and not t['candidate_repeat_stable'] for t in tests):reasons.append('candidate_repeat_bytes_differ')
                        if args.require_native_bytewise and any('stability' in t and not t['exact_native_bytes'] for t in tests):reasons.append('candidate_bytes_differ_from_native')
                        attempt['rejection_reasons']=reasons
                        if reasons:attempt['status']='rejected'
                        else:
                            fn=lambda:wide.launch(inputs[0],qw,s,qz,config)
                            attempt['fresh_hot_timing']=timed(fn,args.repeats)
                            attempt['fresh_cold_timing']=timed(fn,args.cold_search_repeats,True)
                            attempt['status']='accepted_for_selection';accepted.append(attempt)
                    except Exception as error:
                        attempt.update(status='candidate_error',error=repr(error))
                        if fatal(error):save_attempt(attempt);raise
                    save_attempt(attempt)
                if not accepted:
                    measurements.append(dict(projection=label,m=1,k=k,n=n,status='no_eligible_candidate',
                        native_controls=native_suites,old_wide_diagnostic=old_suites))
                    write(out/'measurements.json',measurements,exclusive=False)
                    del qw,s,qz,dense,inputs,references;gc.collect();torch.cuda.empty_cache();continue
                winner=min(accepted,key=lambda v:v['fresh_cold_timing']['cuda_p50_ms']);config=winner['config'];selected[label]=config
                functions=dict(native=lambda:native.precision_linear(inputs[0],qw,s,qz),
                    candidate=lambda:wide.launch(inputs[0],qw,s,qz,config),
                    old_wide_unapproved=lambda:wide.launch(inputs[0],qw,s,qz,source_configs[(1,k,n)]))
                confirmations={condition:{name:[] for name in functions} for condition in ('hot','cold')}
                rng=random.Random(20261120+index)
                for condition in confirmations:
                    for _ in range(2):
                        order=list(functions);rng.shuffle(order)
                        for name in order:
                            confirmations[condition][name].append(timed(functions[name],args.cold_repeats if condition=='cold'
                                                                        else args.confirm_repeats,condition=='cold'))
                medians={condition:{name:statistics.median(x for block in blocks for x in block['samples_ms'])
                                    for name,blocks in values.items()} for condition,values in confirmations.items()}
                row=dict(projection=label,m=1,k=k,n=n,status='confirmed',tensors=tensors,selected_config=config,
                    winner_checks=winner['checks'],native_controls=native_suites,old_wide_diagnostic=old_suites,
                    confirmation=confirmations,confirmation_p50_ms=medians,
                    confirmed_candidate_to_native_ratio={condition:v['candidate']/v['native'] for condition,v in medians.items()})
                measurements.append(row);write(out/'measurements.json',measurements,exclusive=False)
                print(json.dumps(dict(projection=label,selected_config=config,p50_ms=medians)),flush=True)
                del qw,s,qz,dense,inputs,references;gc.collect();torch.cuda.empty_cache()
            complete=set(selected)==set(SHAPES)
            shapes=[]
            if complete:
                for label,(k,n) in SHAPES.items():
                    for m in (1,701,2815):
                        shapes.append(dict(m=m,k=k,n=n,config=selected[label] if m==1 else source_configs[(m,k,n)]))
                configs={(v['m'],v['k'],v['n']):v['config'] for v in shapes}
                for layer in (0,18,35):
                    for label,(k,n) in SHAPES.items():
                        (qw,s,qz),tensors=load(label,layer);dense=dense_reference(qw,s,qz)
                        for m in (1,701,2815):
                            config=configs[(m,k,n)]
                            for amplitude in (.1,1.):
                                x=torch.randn((m,k),device='cuda',dtype=torch.float16)*amplitude
                                ref=torch.nn.functional.linear(x.float(),dense).half()
                                fn=lambda:wide.launch(x,qw,s,qz,config)
                                value=dict(layer=layer,projection=label,m=m,k=k,n=n,input_magnitude=amplitude,
                                           tensors=tensors,config=config,**check(fn(),ref))
                                numeric.append(value);write(out/'numerical_validation.json',numeric,exclusive=False)
                                assert value['passed'],value
                                if m==1:
                                    native_fn=lambda:native.precision_linear(x,qw,s,qz)
                                    base_num=check(native_fn(),ref);assert base_num['passed'],base_num
                                    base_suite=stability(native_fn);candidate_suite=stability(fn)
                                    relation=compare_stability(candidate_suite,base_suite)
                                    record=dict(layer=layer,projection=label,m=1,k=k,n=n,input_magnitude=amplitude,
                                        input_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
                                        tensors=tensors,config=config,native_numerical=base_num,
                                        native_stability=base_suite,candidate_stability=candidate_suite,**relation)
                                    validation_stability.append(record);write(out/'stability_validation.json',validation_stability,exclusive=False)
                                    assert relation['native_repeat_stable'] and relation['candidate_repeat_stable'],record
                                    assert not args.require_native_bytewise or relation['exact_native_bytes'],record
                        del qw,s,qz,dense,x,ref;gc.collect();torch.cuda.empty_cache()
                assert len(numeric)==72 and len(validation_stability)==24
            checksum=int(cache.sum(dtype=torch.int64).item());flush_count=sum(flush_counts.values())
            expected=(flush_count%256)*cache.numel();assert checksum==expected
            for name,sha in meta['sources_sha256'].items():assert digest(ROOT/'checks'/name)==sha,'Source changed: '+name
            for name,sha in meta['source_wide_sources_sha256'].items():assert digest(ROOT/'checks'/name)==sha,'Original probe source changed: '+name
            assert digest(helper_path)==meta['native_awq_source']['sha256']
            table=None
            if complete:
                table=dict(scope='exact_m_only',algorithm=args.group,model_weights_sha256=model_sha,shapes=shapes,
                    numerical_gate_passed=True,validation_layers=[0,18,35],validation_input_magnitudes=[.1,1.],
                    validation_cases=72,validation_all_passed=True,stability_gate_passed=True,stability_cases=24,
                    decode_policy=args.decode_policy,native_bytewise_required=args.require_native_bytewise,
                    native_bytewise_all_m1_validation=all(v['exact_native_bytes'] for v in validation_stability),
                    prefill_source_wide_table_sha256=meta['source_wide_table_sha256'],
                    whole_model_output_gate='pending_new_four_cell_test')
                write(out/'launch_table.json',table)
            write(out/'result.json',dict(status='passed_projection_and_stability_gates' if complete else 'no_eligible_candidate',
                protocol=meta,elapsed_seconds=time.perf_counter()-started,attempts=len(attempts),
                attempt_status_counts=dict(Counter(v['status'] for v in attempts)),measurements=measurements,
                numerical_validation=numeric,stability_validation=validation_stability,launch_table=table,
                cache_flush_evidence=dict(buffer_bytes=cache.numel(),flush_count=flush_count,
                    flush_counts_by_purpose=dict(flush_counts),full_buffer_checksum=checksum,expected_checksum=expected,verified=True),
                limitation='Bytewise stability on fixed random inputs is a finite test. Native equality is optional and explicitly recorded. This does not prove all activations, whole-model tokens, navigation quality or HTTP gains; failed original wide64 gate remains failed.'))
    except BaseException as error:
        write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            attempts=len(attempts),completed_measurements=len(measurements),numeric_cases=len(numeric),
            stability_cases=len(validation_stability),elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        for lock in locks:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
