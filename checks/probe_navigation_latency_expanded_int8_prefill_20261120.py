"""Isolated expanded-INT8 prefill microprobe; never installs a model adapter.

New W8 weights and A8 activations introduce approximation beyond the unchanged
W4 checkpoint. Strict implementation gates and a predeclared2% random-input
screen are separate from full-model/actual-activation/navigation quality.
Only an explicit root-scheduled invocation may acquire the study GPU locks.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback

from navigation_stage2_common import ROOT, GROUPS, digest, write

PROJECTIONS = {'qkv':['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
               'o':['self_attn.o_proj'], 'gate_up':['mlp.gate_proj','mlp.up_proj'], 'down':['mlp.down_proj']}
SHAPES = {'qkv':(2560,6144), 'o':(4096,2560), 'gate_up':(2560,19456), 'down':(9728,2560)}
FROZEN = {'awq':'31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
          'gptq':'2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f'}
LAYERS, ROWS, AMPLITUDES = (0,18,35), (17,701,2815), (.1,1.)
APPROX_LIMIT, NATIVE_LIMIT = .02, .005  # Frozen before any GPU observation.


def sample_indices(size, count):
    """Deterministic boundary/interior coverage, no GPU work."""
    if size < count:
        raise ValueError('Not enough entries')
    values = [0, size-1, size//2]
    for i in range(1, count*2):
        value = (i*size)//count
        if value not in values:
            values.append(value)
        if len(values) == count:
            break
    return sorted(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=tuple(FROZEN), default='awq')
    parser.add_argument('--baseline-table', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--cold-repeats', type=int, default=10)
    parser.add_argument('--stability-repeats', type=int, default=4)
    parser.add_argument('--cold-cache-mib', type=int, default=96)
    parser.add_argument('--pad-m-multiple', type=int, choices=(0,8,16), default=0,
                        help='Separate experiment;0 means actual M with no padding.')
    args = parser.parse_args()
    if min(args.repeats,args.cold_repeats) < 4 or args.stability_repeats < 4 or args.cold_cache_mib < 64:
        parser.error('Require >=4 timing/stability repeats and a buffer larger than48MiB L2')
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'logs') or out.exists():
        parser.error('Require a fresh output directory below logs')
    out.mkdir(parents=True)
    locks, numerical, attempts, measurements, conversions, boundaries = [], [], [], [], [], []
    flush_counts, cache = Counter(), None
    started = time.perf_counter()
    try:
        import fcntl
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock',ROOT/'tmp/navigation_repair_gpu.lock'):
            lock = path.open('a+')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:
                lock.close()
                raise
            locks.append(lock)
        import numpy as np
        import torch
        from safetensors import safe_open
        import navigation_w4a16_repair_kernels as native
        import navigation_latency_expanded_int8_prefill_20261120 as candidate
        from lmdeploy.pytorch.kernels.cuda import awq_kernels
        torch.set_num_threads(4)
        torch.manual_seed(20261120)
        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.cuda.get_device_capability() == (8,9), 'This experiment is scoped to SM89'
        versions = {name:version(name) for name in ('torch','triton','lmdeploy','numpy')}
        assert versions['torch']=='2.5.1+cu121' and versions['triton']=='3.1.0' and versions['lmdeploy']=='0.11.1'
        model = ROOT/GROUPS[args.group]['model']/'model.safetensors'
        model_sha = digest(model)
        assert model_sha == FROZEN[args.group], 'Actual checkpoint identity changed'
        sources = ('probe_navigation_latency_expanded_int8_prefill_20261120.py',
                   'navigation_latency_expanded_int8_prefill_20261120.py','navigation_stage2_common.py',
                   'navigation_w4a16_repair_kernels.py','navigation_w4a16_candidate_kernels.py')
        meta = dict(schema='navigation_expanded_int8_prefill_probe/v1',
            created_utc=datetime.now(timezone.utc).isoformat(),group=args.group,model_weights_sha256=model_sha,
            versions=versions,gpu=dict(name=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability())),
            baseline_table=str(args.baseline_table.resolve()),baseline_table_sha256=digest(args.baseline_table),
            sources_sha256={name:digest(ROOT/'checks'/name) for name in sources},
            native_awq_source=dict(path=awq_kernels.__file__,sha256=digest(awq_kernels.__file__)),
            design_sha256=digest(ROOT/'reports/navigation_latency_expanded_int8_prefill_design_20261120.json'),
            layers=list(LAYERS),rows=list(ROWS),amplitudes=list(AMPLITUDES),projections=list(PROJECTIONS),
            layouts=list(candidate.LAYOUTS),logical_cases=72,layout_execution_cases=144,
            semantic_boundaries_per_layout=['positive_bound','negative_bound','zero_row','ties_to_even','zero_weight'],
            pad_m_multiple=args.pad_m_multiple,stability_repeats=args.stability_repeats,
            timing_repeats=args.repeats,cold_repeats=args.cold_repeats,cold_cache_mib=args.cold_cache_mib,
            approximation_relative_l2_limit=APPROX_LIMIT,native_reference_relative_l2_limit=NATIVE_LIMIT,
            seed=20261120,
            arithmetic='Original W4 FP16-round dequant -> symmetric per-channel W8; per-token A8; INT32 whole-K dot; FP32 (integer*row_scale)*channel_scale -> FP16',
            quantization='maxabs/127 in FP32; zero-vector scale1; round-nearest-even, clamp[-127,127]; no clipping/calibration/smoothing',
            implementation_gate='Independent selected CPU int64 dot exact; CPU ordered FP32 epilogue final half bytes exact; CPU quantizer/dequant selected rows/columns exact; full integer/output repeat bytes',
            approximation_gate='Finite and <=2% global relativeL2 for reconstructed weights, reconstructed activations and combined output on every random case; component output errors retained',
            timing='Full activation quant + whole-K torch._int_mm + epilogue; graph replay. Two randomized rounds, hot and96MiB cold. GEMM-only timings are diagnostic.',
            advancement_rule='All cases at a projection/layout pass and both hot/cold full-path gain>=3% versus frozen W4 and same-W4 dequant FP16 at each measured M. Still only a micro candidate.',
            memory_scope='Single-projection isolated micro; both layouts may coexist here. Full model must retain one layout only and unchanged KV64x64; fit unproven.',
            quality_status='not_evaluated; random projection screen is not actual activation or navigation accuracy',
            full_model_adapter_created=False,original_checkpoint_modified=False)
        write(out/'protocol.json',meta)
        cache = torch.zeros(args.cold_cache_mib*1048576,device='cuda',dtype=torch.uint8)

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

        def timed(fn,count,cold):
            graph,result=capture(fn);events=[]
            for _ in range(count):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                if cold: evict('timing')
                begin.record();graph.replay();end.record();events.append((begin,end))
            torch.cuda.synchronize();samples=[a.elapsed_time(b) for a,b in events]
            assert all(math.isfinite(v) and v>0 for v in samples)
            record=dict(samples_ms=samples,cuda_p50_ms=statistics.median(samples),mode='cuda_graph_replay',
                cache_condition='eviction_before_each_sample' if cold else 'hot',
                eviction_buffer_bytes=cache.numel() if cold else 0,flush_in_timed_interval=False)
            del graph,result
            return record

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

        def save_attempt(record):
            attempts.append(record)
            with (out/'attempts.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')

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

        with torch.inference_mode():
            # Keep unsupported/error layouts visible; one layout may fail a
            # boundary without hiding all observations for the other layout.
            boundary_passed={}
            for layout in candidate.LAYOUTS:
                for case in ('positive_bound','negative_bound','zero_row','ties_to_even','zero_weight'):
                    try:
                        boundary=boundary_case(case,layout)
                    except Exception as exc:
                        boundary=dict(case=case,layout=layout,passed=False,error=repr(exc),traceback=traceback.format_exc())
                        if fatal(exc):
                            boundaries.append(boundary);write(out/'boundaries.json',boundaries,exclusive=False)
                            raise
                    boundaries.append(boundary);write(out/'boundaries.json',boundaries,exclusive=False)
                boundary_passed[layout]=all(v['passed'] for v in boundaries if v['layout']==layout)
            gc.collect();torch.cuda.empty_cache()
            native.load_launch_table(args.baseline_table,args.group,model_sha)
            for label in PROJECTIONS:
                (qw,s,qz),cpu,_,_=load(label,0)
                native.prepare_projection(qw,s,qz)
                del qw,s,qz,cpu;gc.collect();torch.cuda.empty_cache()
            write(out/'baseline_replayed_table.json',native.freeze(args.group,model_sha))

            for layer in LAYERS:
                for label in PROJECTIONS:
                    (qw,s,qz),cpu,tensors,load_s=load(label,layer)
                    k,n=SHAPES[label]
                    assert qw.shape==(k,n//8) and s.shape==(k//128,n) and qz.shape==(k//128,n//8)
                    dense,dequant_setup=measured_setup(lambda:dense_dequant(qw,s,qz))
                    dense32=dense.float()
                    weights={};setup_records={}
                    for layout in candidate.LAYOUTS:
                        (w8,sw),setup=measured_setup(lambda layout=layout:candidate.prepare_weight(dense,layout))
                        reconstructed=w8.float()*sw
                        weight_error=error(reconstructed,dense,APPROX_LIMIT)
                        receipt=dict(layer=layer,projection=label,layout=layout,k=k,n=n,tensors=tensors,
                            checkpoint_load_and_transfer_s=load_s,dequant_setup=dequant_setup,requant_and_layout_setup=setup,
                            weight_error=weight_error,expanded_weight_logical_sha256=tensor_sha(w8),
                            channel_scale_sha256=tensor_sha(sw),stride=list(w8.stride()),
                            resident_weight_bytes=w8.untyped_storage().nbytes(),resident_scale_bytes=sw.untyped_storage().nbytes())
                        conversions.append(receipt);setup_records[layout]=receipt
                        weights[layout]=(w8,sw,reconstructed)
                    assert setup_records['kn_contiguous']['expanded_weight_logical_sha256']==setup_records['nk_transposed']['expanded_weight_logical_sha256']
                    write(out/'conversions.json',conversions,exclusive=False)

                    for m in ROWS:
                        for amplitude in AMPLITUDES:
                            x=torch.randn((m,k),device='cuda',dtype=torch.float16)*amplitude
                            x_sha=tensor_sha(x)
                            reference=torch.mm(x.float(),dense32).half()
                            native_result=native.precision_linear(x,qw,s,qz)
                            native_error=error(native_result,reference,NATIVE_LIMIT)
                            fp16_result=torch.mm(x,dense)
                            fp16_error=error(fp16_result,reference,NATIVE_LIMIT)
                            base=dict(layer=layer,projection=label,m=m,k=k,n=n,amplitude=amplitude,
                                input_sha256=x_sha,tensors=tensors,native_error=native_error,dequant_fp16_gemm_error=fp16_error)
                            numerical.append(base);write(out/'numerical_baselines.json',numerical,exclusive=False)
                            assert native_error['passed'] and fp16_error['passed'], 'A required same-weight reference failed'
                            for layout in candidate.LAYOUTS:
                                w8,sw,reconstructed=weights[layout]
                                record=dict(layer=layer,projection=label,m=m,k=k,n=n,amplitude=amplitude,
                                    input_sha256=x_sha,tensors=tensors,layout=layout,pad_m_multiple=args.pad_m_multiple,
                                    weight_error=setup_records[layout]['weight_error'])
                                pipe=actual=reconstructed_x=component_weight=component_activation=None
                                try:
                                    allocation_before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                                    pipe=candidate.Pipeline(w8,sw,m,args.pad_m_multiple)
                                    allocation_after=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                                    actual=pipe.run(x)
                                    implementation=cpu_implementation_check(pipe,x,dense,cpu)
                                    reconstructed_x=pipe.qx[:m].float()*pipe.row_scale[:m,None]
                                    activation_error=error(reconstructed_x,x,APPROX_LIMIT)
                                    component_weight=torch.mm(x.float(),reconstructed).half()
                                    component_activation=torch.mm(reconstructed_x,dense32).half()
                                    approximation=dict(weight_only=error(component_weight,reference,APPROX_LIMIT),
                                        activation_only=error(component_activation,reference,APPROX_LIMIT),
                                        combined=error(actual,reference,APPROX_LIMIT),activation=activation_error,
                                        scope='Random-projection approximation only; not full-model accuracy')
                                    approximation_passed=record['weight_error']['passed'] and activation_error['passed'] and approximation['combined']['passed']
                                    record.update(implementation=implementation,approximation=approximation,
                                        approximation_screen_passed=approximation_passed,pipeline_storage=pipe.storage_receipt(),
                                        padded_m=pipe.padded_m,pipeline_allocation=dict(before=allocation_before,after=allocation_after,
                                            allocated_delta=allocation_after['allocated']-allocation_before['allocated'],
                                            scope='Isolated projection buffers; no claim of fullmodel resident/peak VRAM'))
                                    if implementation['passed']:
                                        record['stability']=stability(pipe,x)
                                    else:
                                        record['stability']=dict(passed=False,not_run_reason='implementation_gate_failed')
                                    record['semantic_boundaries_passed']=boundary_passed[layout]
                                    accepted=implementation['passed'] and approximation_passed and record['stability']['passed'] and boundary_passed[layout]
                                    record['status']='accepted_random_micro_case' if accepted else 'rejected_random_micro_case'
                                    if accepted and layer==0 and amplitude==1. and m in (701,2815):
                                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                                        timing_memory_before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
                                        fns=dict(native_w4a16=lambda:native.precision_linear(x,qw,s,qz),
                                                 same_w4_dense_fp16=lambda:torch.mm(x,dense),expanded_int8_full=lambda:pipe.run(x))
                                        confirm={condition:{name:[] for name in fns} for condition in ('hot','cold')}
                                        rng=random.Random(20261120+m+list(PROJECTIONS).index(label)*7+list(candidate.LAYOUTS).index(layout))
                                        for condition in confirm:
                                            for round_index in range(2):
                                                order=list(fns);rng.shuffle(order)
                                                for name in order:
                                                    block=timed(fns[name],args.cold_repeats if condition=='cold' else args.repeats,condition=='cold')
                                                    block.update(round=round_index,execution_order=order)
                                                    confirm[condition][name].append(block)
                                        medians={condition:{name:statistics.median(v for block in blocks for v in block['samples_ms'])
                                            for name,blocks in values.items()} for condition,values in confirm.items()}
                                        gain={condition:{name:1-values['expanded_int8_full']/values[name]
                                            for name in ('native_w4a16','same_w4_dense_fp16')} for condition,values in medians.items()}
                                        diagnostic={}
                                        # Full-path confirmation above is authoritative. Pieces are never summed.
                                        for name,fn in (('activation_quant',lambda:pipe.quantize(x)),('int_mm_only',pipe.gemm),('epilogue_only',pipe.epilogue)):
                                            pipe.run(x)
                                            diagnostic[name]=timed(fn,args.repeats,False)
                                        measurement=dict(layer=layer,projection=label,m=m,k=k,n=n,layout=layout,
                                            pad_m_multiple=args.pad_m_multiple,padded_m=pipe.padded_m,tensors=tensors,input_sha256=x_sha,
                                            confirmation=confirm,p50_ms=medians,gain_fraction=gain,diagnostic_hot_components=diagnostic,
                                            timing_screen_passed=all(v>=.03 for values in gain.values() for v in values.values()),
                                            timing_memory=dict(before=timing_memory_before,
                                                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                                                scope='This micro retains reference/validation tensors and two layouts; not deployment memory'),
                                            scope='Full path micro timing only; not language stage or HTTP latency')
                                        measurements.append(measurement);write(out/'measurements.json',measurements,exclusive=False)
                                        record['timing_measurement_index']=len(measurements)-1
                                        print(json.dumps({key:measurement[key] for key in ('projection','m','layout','p50_ms','timing_screen_passed')}),flush=True)
                                        del fns,fn
                                except Exception as exc:
                                    record.update(status='candidate_error',error=repr(exc),traceback=traceback.format_exc())
                                    if fatal(exc): save_attempt(record);raise
                                save_attempt(record)
                                del pipe,actual,reconstructed_x,component_weight,component_activation
                                gc.collect();torch.cuda.empty_cache()
                            del x,reference,native_result,fp16_result
                    del qw,s,qz,cpu,dense,dense32,weights,w8,sw,reconstructed
                    gc.collect();torch.cuda.empty_cache()

        final_source_checks={name:digest(ROOT/'checks'/name)==sha for name,sha in meta['sources_sha256'].items()}
        final_source_checks['native_awq_source']=digest(meta['native_awq_source']['path'])==meta['native_awq_source']['sha256']
        final_source_checks['baseline_table']=digest(args.baseline_table)==meta['baseline_table_sha256']
        final_source_checks['design']=digest(ROOT/'reports/navigation_latency_expanded_int8_prefill_design_20261120.json')==meta['design_sha256']
        assert all(final_source_checks.values()), 'Source or protocol dependency changed during probe'
        count=sum(flush_counts.values());expected_checksum=(count%256)*cache.numel()
        checksum=int(cache.sum(dtype=torch.int64))
        assert checksum==expected_checksum, 'Cold eviction work was not observed as expected'
        assert len(numerical)==72 and len(attempts)==144
        groups=[]
        for label in PROJECTIONS:
            for layout in candidate.LAYOUTS:
                cases=[v for v in attempts if v['projection']==label and v['layout']==layout]
                times=[v for v in measurements if v['projection']==label and v['layout']==layout]
                gates=boundary_passed[layout] and len(cases)==18 and all(v['status']=='accepted_random_micro_case' for v in cases)
                promising=gates and {v['m'] for v in times}=={701,2815} and all(v['timing_screen_passed'] for v in times)
                groups.append(dict(projection=label,layout=layout,cases=len(cases),all_random_micro_gates_passed=gates,
                    promising_for_next_investigation=promising,scope='Not full-model approval; future actual activations/navigation quality and VRAM required'))
        result=dict(status='completed_isolated_micro_screen',protocol=meta,elapsed_s=time.perf_counter()-started,
            attempt_status_counts=dict(Counter(v['status'] for v in attempts)),numerical_baseline_cases=len(numerical),
            per_projection_layout=groups,measurements=measurements,conversions=conversions,boundaries=boundaries,
            final_source_checks=final_source_checks,
            cold_cache=dict(bytes=cache.numel(),flush_counts_by_purpose=dict(flush_counts),flush_count=count,
                expected_final_uint8_value=count%256,checksum=checksum,expected_checksum=expected_checksum,passed=True,
                error_case_ledger_scope='Counts include any completed eviction before an interrupted/rejected case; no invented per-case attribution'),
            all144_weight_expansion_estimate=dict(int8_bytes=3633315840,fp32_scale_bytes=4423680,
                total_gib=(3633315840+4423680)/2**30,measured_fullmodel_peak=False),
            whole_model_quality='not_evaluated',actual_activation_quality='not_evaluated',
            service_table=None,limitations=['Random inputs only;2% threshold is not model accuracy.',
                'Native decode remains outside this experiment; this is new prefill quantization arithmetic.',
                'Cold and hot micro times cannot be multiplied by36 to manufacture HTTP savings.',
                'Keep original checkpoint and KV64x64; fullmodel VRAM fit is unproven.'])
        write(out/'result.json',result)
        print(json.dumps(dict(status=result['status'],attempt_status_counts=result['attempt_status_counts'],
                              promising=[v for v in groups if v['promising_for_next_investigation']])),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-started,
            completed_attempts=len(attempts),completed_baselines=len(numerical),
            flush_counts_by_purpose=dict(flush_counts),service_table=None),exclusive=False)
        raise
    finally:
        for lock in locks:
            lock.close()


if __name__=='__main__':
    main()
