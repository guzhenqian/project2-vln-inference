"""Experimental qkv/o-only expanded INT8 prefill; original W4 decode/MLP.

This changes precision semantics. It is not a same-W4 implementation speedup
and has no navigation-quality approval. Only the dedicated service installs it.
Every enabled module must pass its startup weight screen and both real-input
smokes. Failed gates stop; they never silently disable a module.
"""
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
import triton

from navigation_repair_runtime import NavigationRepairModel
from navigation_stage2_common import ROOT,digest,read


LIMIT=.02
MAX_M=3072
SHAPES={'qkv':(2560,6144),'o':(4096,2560),'gate_up':(2560,19456),'down':(9728,2560)}
SUFFIX={'qkv':'self_attn.qkv_proj','o':'self_attn.o_proj','gate_up':'mlp.gate_up_proj','down':'mlp.down_proj'}
ENABLED=('qkv','o')
WARM_M=(17,701,2815,3072)
PACKAGE_GUARDS={
    'pytorch/backends/cuda/awq_modules.py':'a8715b0c22df0a36501c56f59f0f4fb53d8fb7b5061cd0b5a7ff2b73244c9da7',
    'pytorch/backends/cuda/graph_runner.py':'f777a15c13f04f25fe6199c459ae09fa21f337ca406241106ae04fbc3e14b8f9',
    'pytorch/models/utils/cudagraph.py':'e41a02fd3767da693b724ba670d11a06198fb58bd7ba8d55fd9f51a1dfe134b9',
    'pytorch/models/qwen3_vl.py':'377b26921dfe42b7d91c2306a908713741fa461a792863b18da0182435127e25',
    'pytorch/engine/model_agent.py':'7844f9e2a6f05dc7b1356860e6f669ca3b57268632066837b37f9f85df404a27',
    'pytorch/engine/cache_engine.py':'e10e0cd870587ce0a44f74e309990323e1a285f3b7727d888cde8caef6c5bfdc',
    'pytorch/backends/cuda/op_backend.py':'6220ce492e60b3052b78bcc25783cc0f1f483ac273358a20f505ddc82845b963',
    'pytorch/model_inputs.py':'7bbbb91cd2a4497a95b3751003d5c4d03ece99d35f148acb7bfd71e6c0fa8adf',
}


def module_inventory():
    return {f'language_model.layers.{layer}.{suffix}':(layer,kind,*SHAPES[kind])
            for layer in range(36) for kind,suffix in SUFFIX.items()}


def require(condition,message):
    if not condition: raise RuntimeError(message)


def tensor_receipt(t):
    raw=t.detach().cpu().contiguous()
    return dict(shape=list(t.shape),dtype=str(t.dtype),
        sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())


def dequant(qw,scales,qz):
    def unpack(t):
        return torch.stack([(t>>shift)&15 for shift in (0,16,4,20,8,24,12,28)],-1).reshape(t.shape[0],-1)
    return ((unpack(qw)-unpack(qz).repeat_interleave(128,0)).float()*
            scales.repeat_interleave(128,0).float()).half().contiguous()


def local_error(actual,reference):
    require(actual.shape==reference.shape and actual.dtype==reference.dtype==torch.float16,
            'Local comparison shape/dtype differs')
    delta=actual.float()-reference.float()
    relative=float(delta.norm()/reference.float().norm().clamp_min(1.e-30))
    maximum=float(delta.abs().max())
    finite=bool(torch.isfinite(actual).all()) and bool(torch.isfinite(reference).all())
    return dict(relative_l2=relative if math.isfinite(relative) else None,
        max_absolute_error=maximum if math.isfinite(maximum) else None,finite=finite,
        threshold=LIMIT,passed=finite and math.isfinite(relative) and relative<=LIMIT)


def memory_receipt():
    free,total=torch.cuda.mem_get_info()
    return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        device_free=free,device_total=total)


class SharedScratch:
    """One compute-stream owner. Output is deliberately not shared."""
    def __init__(self,device):
        self.stream=torch.cuda.current_stream(device).cuda_stream
        self.device=device
        self.qx=torch.empty(MAX_M*4096,device=device,dtype=torch.int8)
        self.integer=torch.empty(MAX_M*6144,device=device,dtype=torch.int32)
        self.scale=torch.empty(MAX_M,device=device,dtype=torch.float32)
        self.busy=False

    def run(self,x,weight,channel_scale):
        import navigation_latency_expanded_int8_prefill_20261120 as math_kernel
        m,k=x.shape;n=weight.shape[1]
        require(16<m<=MAX_M and (k,n) in (SHAPES['qkv'],SHAPES['o']), 'Unsupported enabled prefill shape')
        require(x.dtype==torch.float16 and x.is_contiguous() and x.device==self.device and x.data_ptr()%16==0,
                'Unexpected input dtype/stride/device/alignment')
        require(tuple(weight.stride())==(1,k) and weight.dtype==torch.int8 and channel_scale.dtype==torch.float32,
                'Expected frozen NK-transposed INT8 weights')
        require(weight.device==self.device and channel_scale.device==self.device and channel_scale.shape==(n,)
                and channel_scale.is_contiguous() and weight.data_ptr()%16==channel_scale.data_ptr()%16==0,
                'Unexpected expanded weight/scale device or alignment')
        require(torch.cuda.current_stream(self.device).cuda_stream==self.stream and not self.busy,
                'Shared scratch requires serial execution on its owner stream')
        require(not torch.cuda.is_current_stream_capturing(),'INT8 prefill must not enter decode capture')
        self.busy=True
        try:
            # Flat prefixes preserve contiguous [M,K] and [M,N] views. Slicing
            # a max-width2D allocation would retain the wrong leading stride.
            qx=self.qx[:m*k].view(m,k)
            integer=self.integer[:m*n].view(m,n)
            row_scale=self.scale[:m]
            output=torch.empty((m,n),device=self.device,dtype=torch.float16)
            math_kernel._quantize_rows[(m,)](x,qx,row_scale,m,k,triton.next_power_of_2(k),
                num_warps=8,enable_fp_fusion=False)
            torch._int_mm(qx,weight,out=integer)
            math_kernel._scale_output[(triton.cdiv(m*n,1024),)](integer,row_scale,channel_scale,output,m,n,1024,
                num_warps=4,enable_fp_fusion=False)
            return output
        finally:
            self.busy=False

    def receipt(self):
        return dict(max_m=MAX_M,owner_stream=self.stream,
            storage={name:dict(bytes=t.untyped_storage().nbytes(),pointer=t.data_ptr(),dtype=str(t.dtype))
                for name,t in (('qx',self.qx),('integer',self.integer),('row_scale',self.scale))},
            total_bytes=sum(t.untyped_storage().nbytes() for t in (self.qx,self.integer,self.scale)),
            output='Fresh independent FP16[M,N] allocation each invocation',tight_flat_prefix_views=True,
            scope='Single request and single model compute stream; no cross-stream scratch reuse')


class NavigationLatencyInt8AttentionModel(NavigationRepairModel):
    def __init__(self,*args,**kwargs):
        self._int8_installed=False
        self._int8_calls=Counter();self._int8_capture=Counter();self._int8_request_start=Counter()
        self._int8_smoke_passed=set();self._int8_smoke_rows={};self._int8_smoke_input={}
        super().__init__(*args,**kwargs)
        require(self._is_quant and self._method=='awq','This first experimental mask is AWQ only')
        require(os.environ.get('PROJECT2_LATENCY_INT8_ATTENTION')=='qkv_o','Explicit qkv/o INT8 mode required')
        require(os.environ.get('PROJECT2_REPAIR_WBITS')=='4' and os.environ.get('PROJECT2_REPAIR_ABITS')=='16',
                'Original packed W4/native fallback protocol changed')

    def forward(self,*args,**kwargs):
        first=not self._fix_execution_verified
        if first or (self._fix_index in (0,1) and not self.ctx_mgr.current_context().is_decoding):
            caches=kwargs.get('past_key_values',args[2] if len(args)>2 else None)
            actual_kv=self._actual_kv_receipt(caches)
            if hasattr(self,'_int8_actual_kv'):
                require(actual_kv==self._int8_actual_kv,'Actual KV allocation/layout changed')
            else:
                self._int8_actual_kv=actual_kv
        result=super().forward(*args,**kwargs)
        if first:
            self._install_int8_attention()
        elif self._fix_index in (0,1) and not self.ctx_mgr.current_context().is_decoding:
            index=self._fix_index
            records=self._int8_smoke_rows[index]
            require(len(records)==72 and {v['module'] for v in records}==self._int8_enabled,
                    'Real-input smoke did not cover all72 enabled modules exactly once')
            require(all(v['comparison']['passed'] for v in records),'A smoke module exceeded the fixed error gate')
            self._fix_write(f'int8_attention_smoke_{index:03d}.json',dict(index=index,status='passed_local_error_gate',
                input_binding=self._int8_smoke_input[index],modules=records,enabled_count=72,relative_l2_limit=LIMIT,
                actual_kv=self._int8_actual_kv,
                reference='Original frozen W4 linear on the same candidate-trajectory activation',
                extra_reference_linear_calls=72,extra_model_forwards=0,
                whole_model_quality='not_evaluated',timing_scope='Excluded input warmup only'))
            self._int8_smoke_passed.add(index)
        return result

    def _actual_kv_receipt(self,caches):
        context=self.ctx_mgr.current_context();config=context.cache_config
        model_config=context.model_config
        require(config.num_gpu_blocks==64 and config.block_size==64 and config.quant_policy==0,
                'Actual KV config must stay64 blocks x64 FP16 tokens')
        require(model_config.num_layers==36 and model_config.dtype==torch.float16 and
                not model_config.use_mla_fp8_cache and not model_config.cache_shapes,
                'Only the original FP16 KV layout without custom/MLA caches is supported')
        # Guarded CUDA op_backend get_k/v_block_shape returns
        # (block_size,num_heads,head_size). CacheEngine prefixes num_blocks;
        # quant_policy0 and empty cache_shapes produce exactly K/V, no None tail.
        head_count=model_config.num_key_value_heads
        expected_shapes={'key':(64,64,head_count,model_config.k_head_dim or model_config.head_dim),
                         'value':(64,64,head_count,model_config.v_head_dim or model_config.head_dim)}
        require(isinstance(caches,(list,tuple)) and len(caches)==36,'Expected36 actual KV layer pairs')
        layers=[];storages={}
        for index,pair in enumerate(caches):
            require(len(pair)==2,'Quantized/custom KV cache entries are not supported')
            values=[]
            for label,tensor in zip(('key','value'),pair):
                require(torch.is_tensor(tensor) and tuple(tensor.shape)==expected_shapes[label]
                        and tensor.dtype==torch.float16 and tensor.device.type=='cuda','Actual KV shape/dtype differs')
                storage=tensor.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
                values.append(dict(kind=label,shape=list(tensor.shape),stride=list(tensor.stride()),dtype=str(tensor.dtype),
                    device=str(tensor.device),data_pointer=tensor.data_ptr(),storage_pointer=storage.data_ptr(),
                    allocated_blocks=tensor.shape[0],block_size=tensor.shape[1]))
            layers.append(dict(layer=index,tuple_length=len(pair),metadata_tail=[],tensors=values))
        return dict(layer_count=36,allocated_blocks=64,block_size=64,quant_policy=0,layers=layers,
            expected_shapes={k:list(v) for k,v in expected_shapes.items()},custom_cache_shapes=[],
            unique_storage_bytes=sum(storages.values()),scope='Actual tensor metadata only; no KV values read')

    def _install_int8_attention(self):
        import navigation_latency_expanded_int8_prefill_20261120 as math_kernel
        from navigation_latency_int8_attention_contract_20261120 import load_contract
        import lmdeploy.pytorch.backends.cuda.awq_modules as backend
        from importlib.metadata import distribution
        require(not self._int8_installed and not torch.cuda.is_current_stream_capturing(),'Invalid installation lifecycle')
        package=distribution('lmdeploy')
        for name,sha in PACKAGE_GUARDS.items():
            require(digest(package.locate_file('lmdeploy/'+name))==sha,'LMDeploy source guard failed: '+name)
        launch=read(self._fix_dir/'launch.json')
        probe_dir=Path(os.environ['PROJECT2_LATENCY_INT8_PROBE']).resolve()
        proof=load_contract(probe_dir,self._method,launch['model_weights_sha256'],verify_sources=True)
        require(proof==launch['int8_attention_contract']['micro_proof'],'Micro evidence changed since launch')
        require(proof['status']=='verified_micro_contract' and proof['allowed_projections']==list(ENABLED)
                and proof['layout']=='nk_transposed' and proof['full_model_approval'] is False,
                'Micro contract does not authorize exactly qkv/o NK experiment')
        for name,sha in launch['sources_sha256'].items():
            require(digest(ROOT/'checks'/name)==sha,'Source changed before installation: '+name)
        fallback_receipt=read(self._fix_dir/'repair_execution_audit.json')['frozen_launch_table']
        require(fallback_receipt['frozen'] and fallback_receipt['actual_m_preserved'] and fallback_receipt['generic_m_specialization_disabled'] and
                not fallback_receipt['runtime_cold_compilation_allowed'] and not fallback_receipt['runtime_autotuning_allowed'],
                'Original W4 fallback is not frozen')
        expected=module_inventory();objects=self._awq_objects
        require(set(objects)==set(expected) and self._awq_loaded_weights==self._awq_expected_weights,
                'Require all144 canonical original projections and loaded tensors')
        self._int8_enabled={name for name,value in expected.items() if value[1] in ENABLED}
        self._int8_names=set(expected);self._int8_expanded={}
        pointers={module.qweight.data_ptr():name for name,module in objects.items()}
        require(len(pointers)==144,'Original packed weight pointers must be unique')
        prior={ (v['layer'],v['projection']):v for v in read(probe_dir/'conversions.json') if v['layout']=='nk_transposed'}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        startup_before=memory_receipt();started=time.perf_counter();inventory=[]
        self._int8_scratch=SharedScratch(next(iter(objects.values())).scales.device)
        for name,(layer,kind,k,n) in expected.items():
            module=objects[name]
            require(tuple(module.qweight.shape)==(k,n//8) and tuple(module.scales.shape)==(k//128,n)
                    and tuple(module.qzeros.shape)==(k//128,n//8),'Actual projection layout differs: '+name)
            row=dict(module=name,layer=layer,projection=kind,k=k,n=n,
                original_tensors={suffix:tensor_receipt(getattr(module,suffix)) for suffix in ('qweight','scales','qzeros')},
                prefill_route='expanded_int8' if name in self._int8_enabled else 'native_w4',decode_route='native_w4')
            if (layer,kind) in prior:
                require(row['original_tensors']==prior[(layer,kind)]['tensors'],'Runtime/probe packed fusion differs: '+name)
            if name in self._int8_enabled:
                begin=time.perf_counter();before=memory_receipt()
                dense=dequant(module.qweight,module.scales,module.qzeros)
                w8,sw=math_kernel.prepare_weight(dense,'nk_transposed')
                # Compare the actual unrounded reconstructed values, exactly
                # as the frozen probe's weight screen.
                restored32=w8.float()*sw
                delta=restored32-dense.float()
                rel=float(delta.norm()/dense.float().norm().clamp_min(1.e-30))
                finite=bool(torch.isfinite(restored32).all()) and bool(torch.isfinite(dense).all())
                maximum=float(delta.abs().max())
                we=dict(relative_l2=rel if math.isfinite(rel) else None,
                        max_absolute_error=maximum if math.isfinite(maximum) else None,finite=finite,threshold=LIMIT,
                        passed=finite and math.isfinite(rel) and rel<=LIMIT)
                row.update(weight_error=we,expanded_weight=tensor_receipt(w8),channel_scale=tensor_receipt(sw),
                    expanded_stride=list(w8.stride()),resident_bytes=w8.untyped_storage().nbytes()+sw.untyped_storage().nbytes())
                if (layer,kind) in prior:
                    require(row['expanded_weight']['sha256']==prior[(layer,kind)]['expanded_weight_logical_sha256']
                            and row['channel_scale']['sha256']==prior[(layer,kind)]['channel_scale_sha256'],
                            'Expanded runtime bytes differ from tested converter: '+name)
                if not we['passed']:
                    self._fix_write('int8_attention_startup_failed.json',dict(reason='weight_error_gate',failed=row,completed_modules=inventory))
                    raise RuntimeError('Enabled module exceeds predeclared2% weight error: '+name)
                self._int8_expanded[name]=(w8,sw)
                x=torch.zeros((17,k),device=w8.device,dtype=torch.float16)
                warm=self._int8_scratch.run(x,w8,sw)
                require(bool(torch.isfinite(warm).all()),'Nonfinite actual-module warmup')
                torch.cuda.synchronize()
                row.update(warm_m17=True,conversion_and_hash_wall_s=time.perf_counter()-begin,
                           memory_before=before,memory_after=memory_receipt())
                del dense,restored32,delta,x,warm
            else:
                row.update(weight_error=None,weight_error_scope='Not requantized; original W4 tensors unchanged')
            inventory.append(row)
        # Revalidate new shared invocation against the exact frozen Pipeline,
        # including maximum M and nonzero data. Never infer safety from zeros.
        shared_checks=[]
        for kind in ENABLED:
            name=f'language_model.layers.0.{SUFFIX[kind]}'
            w8,sw=self._int8_expanded[name];k,n=SHAPES[kind]
            for m in WARM_M:
                x=torch.randn((m,k),device=w8.device,dtype=torch.float16)*.1
                independent=math_kernel.Pipeline(w8,sw,m)
                ref=independent.run(x);actual=self._int8_scratch.run(x,w8,sw)
                same=bool(torch.equal(ref.contiguous().view(torch.uint8),actual.contiguous().view(torch.uint8)))
                independent_storage=(actual.data_ptr()!=ref.data_ptr() and actual.data_ptr() not in
                    (self._int8_scratch.qx.data_ptr(),self._int8_scratch.integer.data_ptr(),self._int8_scratch.scale.data_ptr()))
                shared_checks.append(dict(module=name,m=m,k=k,n=n,bitwise_equal_to_frozen_pipeline=same,
                    independent_fp16_output_storage=independent_storage,input_sha256=tensor_receipt(x)['sha256']))
                require(same and independent_storage,'Shared scratch invocation differs or aliases output storage')
                del x,ref,actual,independent
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # Startup only: release diagnostic temporary reservations.
        def reject_compile(*args,**kwargs):
            raise RuntimeError('INT8 attention rejected cold compilation after actual-module warmup')
        jits=(math_kernel._quantize_rows,math_kernel._scale_output,math_kernel._weight_scales,math_kernel._quantize_weight)
        for jit in jits: jit.compile=reject_compile
        fallback=backend.wq_gemm_forward

        def dispatch(x,qweight,qzeros,scales,w_bit=4,group_size=128,bias=None,out_features=0):
            require(w_bit==4 and group_size==128 and out_features==scales.shape[1] and x.dtype==torch.float16,
                    'Native AWQ signature/dtype changed')
            name=pointers.get(qweight.data_ptr());require(name in expected,'Unregistered original weight pointer')
            module=objects[name]
            require(scales.data_ptr()==module.scales.data_ptr() and qzeros.data_ptr()==module.qzeros.data_ptr(),
                    'Original scale/zero pointers differ from the registered module')
            layer,kind,k,n=expected[name];flat=x.flatten(0,-2).contiguous();m=flat.shape[0]
            require(flat.shape[1]==k and scales.shape[1]==n,'Runtime shape differs from registry')
            decoding=bool(self.ctx_mgr.current_context().is_decoding)
            require(decoding or 1<=m<=MAX_M,'Prefill exceeds registered maximum')
            route='native_decode' if decoding else 'int8_prefill' if m>16 and name in self._int8_enabled else 'native_prefill'
            self._int8_calls[(route,m,name)]+=1
            if torch.cuda.is_current_stream_capturing():
                require(decoding and route=='native_decode','Only original decode may enter capture')
                self._int8_capture[(route,name)]+=1
            if route!='int8_prefill':
                return fallback(x,qweight,qzeros,scales,w_bit=w_bit,group_size=group_size,bias=bias,out_features=out_features)
            output=self._int8_scratch.run(flat,*self._int8_expanded[name])
            if bias is not None: output=output+bias
            output=output.reshape(x.shape[:-1]+(out_features,))
            if output.ndim==2: output=output.unsqueeze(0)
            index=self._fix_index
            if index in (0,1):
                reference=fallback(x,qweight,qzeros,scales,w_bit=w_bit,group_size=group_size,bias=bias,out_features=out_features)
                comparison=local_error(output,reference)
                record=dict(module=name,layer=layer,projection=kind,m=m,k=k,n=n,input_dtype=str(flat.dtype),
                    input_stride=list(flat.stride()),comparison=comparison,
                    output_metadata={label:dict(shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()),
                        device=str(value.device)) for label,value in (('candidate',output),('reference',reference))},
                    execution=dict(route='eager_prefill',cuda_stream_capturing=torch.cuda.is_current_stream_capturing(),
                        owner_stream=self._int8_scratch.stream,current_stream=torch.cuda.current_stream(flat.device).cuda_stream),
                    activation_selected_rows_sha256=tensor_receipt(flat[[0,m//2,m-1]])['sha256'],
                    sampling_rows=[0,m//2,m-1],reference_extra_linear_calls=1)
                self._int8_smoke_rows[index].append(record)
                if not comparison['passed']:
                    self._fix_write(f'int8_attention_smoke_failed_{index:03d}.json',dict(index=index,failed_module=name,
                        input_binding=self._int8_smoke_input[index],partial_modules=self._int8_smoke_rows[index],
                        threshold=LIMIT,stopped_without_fallback=True))
                    raise RuntimeError('Actual-input local error exceeds predeclared2%: '+name)
            return output

        backend.wq_gemm_forward=dispatch
        self._int8_installed=True
        require(load_contract(probe_dir,self._method,launch['model_weights_sha256'],verify_sources=True)==proof,
                'Micro proof changed during startup')
        for source,sha in launch['sources_sha256'].items():
            require(digest(ROOT/'checks'/source)==sha,'Source changed during startup: '+source)
        self._fix_write('int8_attention_install.json',dict(schema='navigation_int8_attention_install/v1',
            method=self._method,checkpoint_sha256=launch['model_weights_sha256'],micro_proof=proof,
            enabled_projections=list(ENABLED),layout='nk_transposed',enabled_modules=sorted(self._int8_enabled),
            native_prefill_modules=sorted(self._int8_names-self._int8_enabled),all_decode_modules=sorted(self._int8_names),
            module_inventory=inventory,enabled_count=72,native_prefill_count=72,native_decode_count=144,
            prefill_rows=dict(candidate_min=17,candidate_max=MAX_M,small_rows_route='original_native'),
            scratch=self._int8_scratch.receipt(),shared_invocation_checks=shared_checks,
            actual_kv=self._int8_actual_kv,
            startup_before=startup_before,startup_after=memory_receipt(),startup_wall_s=time.perf_counter()-started,
            guarded_package_sources=PACKAGE_GUARDS,
            cold_compile_guard=True,native_fallback_receipt=fallback_receipt,
            original_first_synthetic_forward='Native W4 before install; KV already allocated; decode capture follows',
            numerical_semantics='New qkv/o W8A8 prefill; original W4 MLP/decode; not same-W4 equivalence',
            actual_smoke_gate='pending_two_real_inputs',whole_model_quality='not_evaluated'))

    def _fix_before_step(self,kwargs):
        value=super()._fix_before_step(kwargs)
        if value and self._fix_current_step['phase']=='prefill':
            require(self._int8_installed,'Client prefill before INT8 installation')
            index=self._fix_index;m=kwargs['input_ids'].numel()
            self._int8_request_start=self._int8_calls.copy()
            if index in (0,1):
                require(self._fix_plan[index]['phase']=='input_warmup' and self._fix_plan[index]['sample']==index,
                        'Required diagnostic warmup plan differs')
                require(m==(701 if index==0 else 2815),'Unexpected true-input smoke shape')
                binding=read(self._fix_dir/f'input_{m}.json')
                require(binding['request_index']==index,'Input SHA receipt is not this smoke')
                self._int8_smoke_input[index]=binding;self._int8_smoke_rows[index]=[]
            else:
                require(self._int8_smoke_passed=={0,1},'Both actual-input local-error smokes must pass before formal requests')
        return value

    def _fix_flush(self):
        if not self._fix_steps: return super()._fix_flush()
        index=self._fix_index;steps=list(self._fix_steps)
        prefill=[v['rows'] for v in steps if v['phase']=='prefill']
        require(len(prefill)==1,'Require one complete prefill per request')
        m=prefill[0];delta=self._int8_calls-self._int8_request_start
        expected={name:('int8_prefill' if m>16 and name in self._int8_enabled else 'native_prefill') for name in self._int8_names}
        actual={}
        for (route,rows,name),count in delta.items():
            require(name not in actual and rows==m and count==1 and expected.get(name)==route,
                    'Unexpected per-request route/call/M count')
            actual[name]=route
        require(actual==expected,'Request did not cover all144 projection routes')
        captures={name:count for (route,name),count in self._int8_capture.items() if route=='native_decode'}
        bad={f'{route}:{name}':count for (route,name),count in self._int8_capture.items() if route!='native_decode'}
        modes=Counter(v.get('execution_mode') for v in steps)
        if modes['replay']:
            require(set(captures)==self._int8_names and not bad,'Decode replay lacks all144 original-native capture identities')
        super()._fix_flush()
        self._fix_write(f'int8_attention_execution_{index:03d}.json',dict(index=index,prefill_m=prefill,
            python_projection_calls=[dict(route=r,m=rows,module=n,calls=c) for (r,rows,n),c in sorted(delta.items())],
            prefill_route_counts=dict(Counter(expected.values())),native_decode_capture_projection_calls=captures,
            unexpected_capture_projection_calls=bad,outside_graph_execution_modes=dict(modes),
            graph_replay_python_counters_increment=False,real_smoke_passed_indices=sorted(self._int8_smoke_passed),
            extra_reference_calls=72 if index in (0,1) else 0,
            candidate_trajectory_local_error_only=True,whole_model_quality='not_evaluated'))
