"""NEW independent864-case fixed per-module INT8-mask exploration.

Run only in a root-scheduled GPU slot. The original microprobe is immutable;
its failed projection classes remain failed. This new experiment independently
screens every module, retaining original W4 for any failure or missing proof.
No service is installed. Prior layer0 shape timings are bound, not rerun or
multiplied by36. New case-local seed inputs never claim to reproduce old inputs.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time
import traceback

from navigation_stage2_common import ROOT,GROUPS,digest,write
from navigation_latency_expanded_int8_module_contract_20261120 import (
    SCHEMA,CHECKPOINT,SOURCES,SHAPES,ROWS,AMPLITUDES,LIMIT,LAYOUT,BOUNDARIES,
    require,case_plan,case_reasons,decisions,publish_mask,bind_prior,verify_sources,
    object_sha,selected_indices as sample_indices)

PROJECTIONS = {'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
               'o':['self_attn.o_proj'],'gate_up':['mlp.gate_proj','mlp.up_proj'],'down':['mlp.down_proj']}
DESIGN = 'reports/navigation_latency_expanded_int8_modules_protocol_20261120.json'
DESIGN_SHA = '859028dad2a0401544670bca54a120ff48fce2ceb454c4b0351d25bb6c1a9f68'
HELPERS = ('tensor_sha','load','unpack','dense_dequant','error','measured_setup','evict','capture',
           'stability','cpu_dequant_columns','cpu_implementation_check','fatal','boundary_case')


def verify_copied_helpers():
    old=ast.parse((ROOT/'checks/probe_navigation_latency_expanded_int8_prefill_20261120.py').read_text(encoding='utf-8'))
    new=ast.parse(Path(__file__).read_text(encoding='utf-8'))
    receipts={}
    for name in HELPERS:
        before=[n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name==name]
        after=[n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name==name]
        require(len(before)==len(after)==1,'Copied helper missing or ambiguous: '+name)
        a,b=(ast.dump(nodes[0],include_attributes=False) for nodes in (before,after))
        require(a==b,'Frozen numerical helper changed: '+name)
        receipts[name]=hashlib.sha256(a.encode()).hexdigest()
    return receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--group',choices=('awq',),default='awq')
    parser.add_argument('--baseline-table',type=Path,required=True)
    parser.add_argument('--timing-study',type=Path,required=True)
    parser.add_argument('--stability-repeats',type=int,choices=(4,),default=4)
    args=parser.parse_args();args.pad_m_multiple=0;args.cold_cache_mib=96
    out=args.output.resolve()
    require(out.is_relative_to(ROOT/'logs') and not out.exists(),'Require fresh project logs output')
    locks=[];attempts=[];conversions=[];numerical=[];boundaries=[];flush_counts=Counter();prior=None;cache=None
    started=time.perf_counter()
    try:
        import fcntl
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock',ROOT/'tmp/navigation_repair_gpu.lock'):
            lock=path.open('a+')
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException: lock.close();raise
            locks.append(lock)
        out.mkdir(parents=True)
        verify_sources();helper_identity=verify_copied_helpers()
        require(digest(ROOT/DESIGN)==DESIGN_SHA,'Prewritten independent module protocol changed')
        design=json.loads((ROOT/DESIGN).read_text(encoding='utf-8'));plan=case_plan()
        require(design['case_plan']==plan and design['case_plan_sha256']==object_sha(plan),'Independent case seed plan differs')
        prior=bind_prior(args.timing_study,args.baseline_table)
        write(out/'prior_evidence_binding.json',prior)
        import numpy as np
        import torch
        from safetensors import safe_open
        import navigation_w4a16_repair_kernels as native
        import navigation_latency_expanded_int8_prefill_20261120 as candidate
        from lmdeploy.pytorch.kernels.cuda import awq_kernels
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        require(torch.cuda.get_device_capability()==(8,9),'Only original SM89 device scope supported')
        versions={name:version(name) for name in ('torch','triton','lmdeploy','numpy')}
        old_protocol=json.loads((args.timing_study.resolve()/'protocol.json').read_text(encoding='utf-8'))
        require(versions==old_protocol['versions'],'Installed versions differ from bound shape timing evidence')
        require(digest(awq_kernels.__file__)==old_protocol['native_awq_source']['sha256'],'Native AWQ source changed')
        model=ROOT/GROUPS[args.group]['model']/'model.safetensors';model_sha=digest(model)
        require(model_sha==CHECKPOINT,'Actual whole checkpoint changed')
        sources=dict(SOURCES)
        for name in ('probe_navigation_latency_expanded_int8_modules_20261120.py','navigation_latency_expanded_int8_module_contract_20261120.py'):
            sources[name]=digest(ROOT/'checks'/name)
        meta=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),group=args.group,
            model_weights_sha256=model_sha,actual_checkpoint_verified=True,versions=versions,
            gpu=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability())),
            sources_sha256=sources,copied_frozen_helper_ast_sha256=helper_identity,
            design_sha256=DESIGN_SHA,case_plan_sha256=object_sha(plan),cases=864,modules=144,cases_per_module=6,
            layers=list(range(36)),projections=list(PROJECTIONS),rows=list(ROWS),amplitudes=list(AMPLITUDES),
            layout=LAYOUT,pad_m_multiple=0,approximation_relative_l2_limit=LIMIT,native_reference_relative_l2_limit=.005,
            stability_repeats=4,stability_modes=['eager','cuda_graph_replay'],stability_cache_conditions=['hot','cold'],cold_cache_mib=96,
            baseline_table=str(args.baseline_table.resolve()),baseline_table_sha256=digest(args.baseline_table),
            prior_binding_sha256=digest(out/'prior_evidence_binding.json'),
            seed_scope='Independent deterministic case-local CUDA generator; no old input/repeat identity inherited',
            selected_cpu_scope='Selected4 rows x8 columns int64 dot/ordered FP32 epilogue; quantizer rows/columns and FP16 dequant exact. Not full CPU matrix multiplication.',
            mask_scope='Whole module fixed at startup. All6 new cases plus prior shape timing gate required; old layer35 FFN failures forced originalW4.',
            approximation_gate='Reconstructed weight, activation and combined output finite/global relativeL2<=.02. Component-only output errors diagnostic.',
            timing_scope='No new performance measurements. Bind8 prior NK shape-M records/16 hotcold summaries from layer0; not144 module speed evidence.',
            old_projection_class_gate_changed=False,original_checkpoint_modified=False,full_model_adapter_created=False,
            whole_model_approval=False,actual_activation_quality='not_evaluated',navigation_quality='not_evaluated')
        write(out/'protocol.json',meta);write(out/'case_plan.json',plan)
        cache=torch.zeros(96*1048576,device='cuda',dtype=torch.uint8)

        def tensor_sha(t):
            return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

        def load(label,layer):
            begin = time.perf_counter()
            with safe_open(str(model),framework='pt',device='cpu') as reader:
                cpu = tuple(torch.cat([reader.get_tensor(f'model.language_model.layers.{layer}.{part}.{suffix}')
                    for part in PROJECTIONS[label]],dim=1).contiguous() for suffix in ('qweight','scales','qzeros'))
            receipt = {name:dict(sha256=tensor_sha(t),shape=list(t.shape),dtype=str(t.dtype))
                       for name,t in zip(('qweight','scales','qzeros'),cpu)}
            gpu = tuple(t.cuda() for t in cpu)
            torch.cuda.synchronize()
            return gpu,cpu,receipt,time.perf_counter()-begin

        def unpack(t):
            return torch.stack([(t>>shift)&15 for shift in (0,16,4,20,8,24,12,28)],-1).reshape(t.shape[0],-1)

        def dense_dequant(qw,scales,qz):
            # FP16 rounding is mandatory before later W8 requantization.
            return ((unpack(qw)-unpack(qz).repeat_interleave(128,0)).float()*
                    scales.repeat_interleave(128,0).float()).half().contiguous()

        def error(actual,reference,limit):
            delta = actual.float()-reference.float()
            relative = float(delta.norm()/reference.float().norm().clamp_min(1.e-30))
            maximum = float(delta.abs().max())
            finite = bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
            return dict(relative_l2=relative if math.isfinite(relative) else None,
                max_absolute_error=maximum if math.isfinite(maximum) else None,finite=finite,
                threshold=limit,passed=finite and math.isfinite(relative) and relative<=limit)

        def measured_setup(fn):
            torch.cuda.synchronize()
            before = dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start = time.perf_counter();begin.record();value=fn();end.record();torch.cuda.synchronize()
            receipt = dict(cuda_ms=begin.elapsed_time(end),host_wall_s=time.perf_counter()-start,before=before,
                after=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()),
                peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()),
                excluded_from_steady_state_time=True)
            return value,receipt

        def evict(purpose):
            cache.add_(1)
            flush_counts[purpose] += 1

        def capture(fn):
            for _ in range(3): fn()
            torch.cuda.synchronize()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): fn()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream): result=fn()
            graph.replay();torch.cuda.synchronize()
            return graph,result

        def stability(pipe,x):
            pipe.run(x);torch.cuda.synchronize()
            refs={name:getattr(pipe,name).clone() for name in ('qx','row_scale','integer','output')}
            output_sha=tensor_sha(refs['output'])
            conditions=[]
            for mode in ('eager','cuda_graph_replay'):
                graph,result=capture(lambda:pipe.run(x)) if mode=='cuda_graph_replay' else (None,None)
                for cold in (False,True):
                    trials=[]
                    for _ in range(args.stability_repeats):
                        if cold: evict('stability')
                        if graph is None: pipe.run(x)
                        else: graph.replay()
                        equal={name:bool(torch.equal(value.contiguous().view(torch.uint8),
                                      getattr(pipe,name).contiguous().view(torch.uint8))) for name,value in refs.items()}
                        trials.append(equal)
                    conditions.append(dict(mode=mode,cache_condition='cold' if cold else 'hot',
                        repeats=args.stability_repeats,full_tensor_bytewise_equal=trials,
                        passed=all(all(t.values()) for t in trials),
                        eviction_buffer_bytes=cache.numel() if cold else 0))
                del graph,result
            return dict(reference_output_sha256=output_sha,conditions=conditions,
                passed=all(c['passed'] for c in conditions),
                comparison_scope='All bytes of qx,row_scale,INT32 accumulator,FP16 output, compared on GPU outside timing. Hash is the first reference, not invented per-repeat hashes.')

        def cpu_dequant_columns(cpu,columns):
            qw,scales,qz=(t.numpy() for t in cpu)
            k=qw.shape[0];cols=np.asarray(columns,dtype=np.int64)
            shift=np.asarray((0,16,4,20,8,24,12,28),dtype=np.int32)[cols%8]
            w=(qw[:,cols//8]>>shift[None,:])&15
            z=((qz[:,cols//8]>>shift[None,:])&15)[np.arange(k)//128,:]
            # Select the eight columns before expanding groups; never materialize
            # a full dense CPU weight merely to validate a few columns.
            return ((w-z).astype(np.float32)*scales[:,cols][np.arange(k)//128,:].astype(np.float32)).astype(np.float16)

        def cpu_implementation_check(pipe,x,dense,cpu):
            rows=sample_indices(pipe.m,4);columns=sample_indices(pipe.n,8)
            row_idx=torch.tensor(rows,device=x.device,dtype=torch.long)
            col_idx=torch.tensor(columns,device=x.device,dtype=torch.long)
            cpu_x=x[row_idx].cpu().numpy().astype(np.float32)
            max_x=np.max(np.abs(cpu_x),axis=1)
            sx=np.where(max_x>0,max_x/np.float32(127),np.float32(1)).astype(np.float32)
            qx=np.clip(np.rint(cpu_x/sx[:,None]),-127,127).astype(np.int8)
            actual_qx=pipe.qx[row_idx].cpu().numpy();actual_sx=pipe.row_scale[row_idx].cpu().numpy()
            w16=cpu_dequant_columns(cpu,columns)
            actual_w16=dense[:,col_idx].cpu().numpy()
            max_w=np.max(np.abs(w16.astype(np.float32)),axis=0)
            sw=np.where(max_w>0,max_w/np.float32(127),np.float32(1)).astype(np.float32)
            qw=np.clip(np.rint(w16.astype(np.float32)/sw[None,:]),-127,127).astype(np.int8)
            actual_qw=pipe.weight[:,col_idx].cpu().numpy();actual_sw=pipe.channel_scale[col_idx].cpu().numpy()
            # Independent CPU int64 dot, no float GEMM or shared Triton implementation.
            expected_integer=actual_qx.astype(np.int64)@actual_qw.astype(np.int64)
            actual_integer=pipe.integer[row_idx[:,None],col_idx[None,:]].cpu().numpy()
            expected_half=((expected_integer.astype(np.float32)*actual_sx[:,None])*actual_sw[None,:]).astype(np.float16)
            actual_half=pipe.output[row_idx[:,None],col_idx[None,:]].cpu().numpy()
            flags=dict(packed_fp16_dequant_exact=np.array_equal(w16.view(np.uint16),actual_w16.copy().view(np.uint16)),
                activation_scale_exact=np.array_equal(sx.view(np.uint32),actual_sx.view(np.uint32)),
                activation_quant_exact=np.array_equal(qx,actual_qx),
                weight_scale_exact=np.array_equal(sw.view(np.uint32),actual_sw.view(np.uint32)),
                weight_quant_exact=np.array_equal(qw,actual_qw),
                int32_accumulator_exact=np.array_equal(expected_integer,actual_integer),
                ordered_fp32_epilogue_half_exact=np.array_equal(expected_half.view(np.uint16),actual_half.copy().view(np.uint16)))
            return dict(rows=rows,columns=columns,k=pipe.k,flags={k:bool(v) for k,v in flags.items()},
                passed=all(flags.values()),int64_expected=expected_integer.tolist(),int32_actual=actual_integer.tolist(),
                cpu_expected_half_bits=expected_half.view(np.uint16).tolist(),actual_half_bits=actual_half.copy().view(np.uint16).tolist(),
                theoretical_max_abs_integer=pipe.k*127*127,int32_safe=pipe.k*127*127<2**31,
                scope='Selected entries/quantizer rows/columns only; all-output approximation and repeat gates are separate.')

        def fatal(error):
            return any(v in str(error).lower() for v in ('illegal memory','device-side assert','unspecified launch','misaligned address'))

        def boundary_case(case,layout):
            if case=='zero_weight':
                zero_dense=torch.zeros((128,8),device='cuda',dtype=torch.float16)
                zero_q,zero_scale=candidate.prepare_weight(zero_dense,layout)
                return dict(case=case,layout=layout,
                    passed=bool((zero_q==0).all()) and bool((zero_scale==1).all()))
            k=9728 if case in ('positive_bound','negative_bound') else 128
            n,m=8,17
            host_x=np.full((m,k),np.float16(127/512),dtype=np.float16)
            if case=='negative_bound': host_x *= np.float16(-1)
            if case=='zero_row': host_x.fill(0)
            if case=='ties_to_even':
                host_x.fill(0)
                host_x[:,:8]=np.asarray([127,.5,1.5,2.5,-.5,-1.5,-2.5,-127],dtype=np.float16)
            x=torch.from_numpy(host_x).cuda()
            w8=torch.full((k,n),127,device='cuda',dtype=torch.int8)
            if layout=='nk_transposed': w8=w8.T.contiguous().T
            sw=torch.full((n,),1/512,device='cuda',dtype=torch.float32)
            pipe=candidate.Pipeline(w8,sw,m,args.pad_m_multiple)
            pipe.run(x);torch.cuda.synchronize()
            xf=host_x.astype(np.float32);maximum=np.abs(xf).max(axis=1)
            expected_scale=np.where(maximum>0,maximum/np.float32(127),np.float32(1)).astype(np.float32)
            expected_q=np.clip(np.rint(xf/expected_scale[:,None]),-127,127).astype(np.int8)
            expected_int=expected_q.astype(np.int64)@np.full((k,n),127,dtype=np.int64)
            expected_half=((expected_int.astype(np.float32)*expected_scale[:,None])*np.float32(1/512)).astype(np.float16)
            actual_q=pipe.qx[:m].cpu().numpy();actual_scale=pipe.row_scale[:m].cpu().numpy()
            actual_int=pipe.integer[:m].cpu().numpy();actual_half=pipe.output.cpu().numpy()
            flags=dict(quantized_exact=np.array_equal(expected_q,actual_q),
                scale_exact=np.array_equal(expected_scale.view(np.uint32),actual_scale.view(np.uint32)),
                integer_exact=np.array_equal(expected_int,actual_int),
                half_exact=np.array_equal(expected_half.view(np.uint16),actual_half.view(np.uint16)),
                finite=bool(np.isfinite(actual_half).all()))
            if pipe.padded_m>m:
                flags['padded_q_zero']=bool((pipe.qx[m:]==0).all())
                flags['padded_scale_one']=bool((pipe.row_scale[m:]==1).all())
                flags['padded_integer_zero']=bool((pipe.integer[m:]==0).all())
            boundary=dict(case=case,layout=layout,m=m,k=k,n=n,pad_m_multiple=args.pad_m_multiple,
                padded_m=pipe.padded_m,flags={key:bool(v) for key,v in flags.items()},passed=all(flags.values()),
                max_abs_expected_int=int(np.abs(expected_int).max()),int32_limit=2**31-1)
            return boundary

        def save_case(record):
            attempts.append(record)
            with (out/'attempts.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')

        with torch.inference_mode():
            for name in BOUNDARIES:
                boundary=boundary_case(name,LAYOUT);boundaries.append(boundary)
                write(out/'boundaries.json',boundaries,exclusive=False)
                require(boundary['passed'] is True,'Independent NK semantic boundary failed: '+name)
            gc.collect();torch.cuda.empty_cache()
            native.load_launch_table(args.baseline_table,args.group,model_sha)
            for label in PROJECTIONS:
                (qw,s,qz),cpu,_,_=load(label,0);native.prepare_projection(qw,s,qz)
                del qw,s,qz,cpu;gc.collect();torch.cuda.empty_cache()
            write(out/'baseline_replayed_table.json',native.freeze(args.group,model_sha))
            for layer in range(36):
                for label in PROJECTIONS:
                    (qw,s,qz),cpu,tensors,load_s=load(label,layer);k,n=SHAPES[label]
                    require(qw.shape==(k,n//8) and s.shape==(k//128,n) and qz.shape==(k//128,n//8),'Native packed module layout differs')
                    dense,dequant_setup=measured_setup(lambda:dense_dequant(qw,s,qz));dense32=dense.float()
                    (w8,sw),setup=measured_setup(lambda:candidate.prepare_weight(dense,LAYOUT))
                    reconstructed=w8.float()*sw;weight_error=error(reconstructed,dense,LIMIT)
                    if layer==35 and label in ('gate_up','down'):
                        require(weight_error['passed'] is False and weight_error['relative_l2'] is not None and
                                weight_error['relative_l2']>LIMIT,'Original layer35 weight-error failure did not reproduce under unchanged conversion')
                    conversion=dict(layer=layer,projection=label,layout=LAYOUT,k=k,n=n,tensors=tensors,
                        checkpoint_load_and_transfer_s=load_s,dequant_setup=dequant_setup,requant_and_layout_setup=setup,
                        weight_error=weight_error,expanded_weight_logical_sha256=tensor_sha(w8),channel_scale_sha256=tensor_sha(sw),
                        stride=list(w8.stride()),resident_weight_bytes=w8.untyped_storage().nbytes(),resident_scale_bytes=sw.untyped_storage().nbytes())
                    conversions.append(conversion);write(out/'conversions.json',conversions,exclusive=False)
                    module_plan=[v for v in plan if v['layer']==layer and v['projection']==label]
                    for expected in module_plan:
                        m,amplitude=expected['m'],expected['amplitude']
                        generator=torch.Generator(device='cuda').manual_seed(expected['case_seed'])
                        x=torch.randn((m,k),device='cuda',dtype=torch.float16,generator=generator)*amplitude
                        x_sha=tensor_sha(x);reference=torch.mm(x.float(),dense32).half()
                        native_result=native.precision_linear(x,qw,s,qz);native_error=error(native_result,reference,.005)
                        fp16_result=torch.mm(x,dense);fp16_error=error(fp16_result,reference,.005)
                        base=dict(expected,input_sha256=x_sha,tensors=tensors,native_error=native_error,dequant_fp16_gemm_error=fp16_error)
                        numerical.append(base)
                        with (out/'numerical_baselines.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(base,ensure_ascii=False,allow_nan=False)+'\n')
                        require(native_error['passed'] and fp16_error['passed'],'A required same-weight reference failed')
                        record=dict(base,weight_error=weight_error,conversion_index=len(conversions)-1,
                            semantic_boundaries_passed=True)
                        pipe=actual=reconstructed_x=component_weight=component_activation=None
                        try:
                            pipe=candidate.Pipeline(w8,sw,m,0);actual=pipe.run(x)
                            implementation=cpu_implementation_check(pipe,x,dense,cpu)
                            reconstructed_x=pipe.qx[:m].float()*pipe.row_scale[:m,None]
                            activation_error=error(reconstructed_x,x,LIMIT)
                            component_weight=torch.mm(x.float(),reconstructed).half()
                            component_activation=torch.mm(reconstructed_x,dense32).half()
                            approximation=dict(weight_only=error(component_weight,reference,LIMIT),
                                activation_only=error(component_activation,reference,LIMIT),combined=error(actual,reference,LIMIT),
                                activation=activation_error,scope='Independent random module inputs only; not actual activations or model quality')
                            record.update(implementation=implementation,approximation=approximation,
                                approximation_screen_passed=weight_error['passed'] and activation_error['passed'] and approximation['combined']['passed'],
                                pipeline_storage=pipe.storage_receipt(),padded_m=pipe.padded_m)
                            record['stability']=stability(pipe,x) if implementation['passed'] else dict(passed=False,not_run_reason='implementation_gate_failed')
                            reasons=case_reasons(record,expected,4)
                            record.update(status='accepted_new_module_case' if not reasons else 'rejected_new_module_case',case_rejection_reasons=reasons)
                        except Exception as error_value:
                            record.update(status='candidate_error',error=repr(error_value),traceback=traceback.format_exc())
                            if fatal(error_value):save_case(record);raise
                        save_case(record)
                        del pipe,actual,reconstructed_x,component_weight,component_activation,x,reference,native_result,fp16_result,generator
                        gc.collect();torch.cuda.empty_cache()
                    # Tiny progress receipts; all case evidence remains in the
                    # append-only ledger. No all-mask publication until864cases.
                    decision=next(v for v in decisions(attempts,prior,4) if v['layer']==layer and v['projection']==label)
                    print(json.dumps(dict(module_id=decision['module_id'],route=decision['route'],reasons=decision['reasons'],completed_cases=len(attempts))),flush=True)
                    del qw,s,qz,cpu,dense,dense32,w8,sw,reconstructed
                    gc.collect();torch.cuda.empty_cache()

        verify_sources()
        final_sources={name:digest(ROOT/'checks'/name)==sha for name,sha in sources.items()}
        final_sources['protocol_design']=digest(ROOT/DESIGN)==DESIGN_SHA
        final_sources['baseline_table']=digest(args.baseline_table)==meta['baseline_table_sha256']
        final_sources['native_awq_source']=digest(awq_kernels.__file__)==old_protocol['native_awq_source']['sha256']
        require(all(final_sources.values()),'Source/dependency changed during module screen')
        require(bind_prior(args.timing_study,args.baseline_table)==prior,'Prior timing/negative evidence changed during module screen')
        require(len(attempts)==len(numerical)==864 and len(conversions)==144,'Incomplete module/case coverage')
        count=sum(flush_counts.values());expected_checksum=(count%256)*cache.numel();checksum=int(cache.sum(dtype=torch.int64))
        require(checksum==expected_checksum,'Actual cold eviction work differs')
        evidence=publish_mask(attempts,prior,4)
        write(out/'module_decisions.json',evidence['module_decisions']);write(out/'module_mask.json',evidence['mask'])
        write(out/'module_mask.sha256',dict(semantic_mask_sha256=evidence['mask_sha256'],file_sha256=digest(out/'module_mask.json')))
        result=dict(status='completed_independent_module_micro_screen',protocol=meta,elapsed_s=time.perf_counter()-started,
            attempts=864,modules=144,accepted_modules=evidence['accepted_modules'],retained_w4_modules=evidence['retained_w4_modules'],
            attempt_status_counts=dict(Counter(v['status'] for v in attempts)),mask_sha256=evidence['mask_sha256'],
            module_decisions_sha256=digest(out/'module_decisions.json'),attempt_ledger_file_sha256=digest(out/'attempts.jsonl'),
            conversions_sha256=digest(out/'conversions.json'),selected_weight_and_scale_bytes=evidence['selected_weight_and_scale_bytes'],
            memory_scope=evidence['memory_scope'],original_negative_modules_excluded=prior['negative_modules'],
            prior_shape_timing_scope=prior['timing_scope'],new_performance_measurements=0,
            source_checks=final_sources,cold_cache=dict(bytes=cache.numel(),flush_counts_by_purpose=dict(flush_counts),
                flush_count=count,checksum=checksum,expected_checksum=expected_checksum,passed=True),
            full_model_approval=False,actual_activation_quality='not_evaluated',navigation_quality='not_evaluated',
            old_projection_class_gate_changed=False,per_input_fallback=False,full_model_adapter_created=False)
        write(out/'result.json',result)
        print(json.dumps(dict(status=result['status'],accepted_modules=result['accepted_modules'],retained_w4_modules=result['retained_w4_modules'],mask_sha256=result['mask_sha256'])),flush=True)
    except BaseException as error_value:
        if out.exists():
            write(out/'failure.json',dict(error=repr(error_value),traceback=traceback.format_exc(),completed_cases=len(attempts),
                elapsed_s=time.perf_counter()-started,mask_file_exists=(out/'module_mask.json').exists(),
                mask_validity='No completed result means no valid published mask, even if intermediate files exist'),exclusive=False)
            if prior is not None: write(out/'partial_module_decisions.json',decisions(attempts,prior,4),exclusive=False)
        raise
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':main()
