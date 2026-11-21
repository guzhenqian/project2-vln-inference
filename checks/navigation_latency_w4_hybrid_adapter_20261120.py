"""Independent down-only GEMV decode candidate over the saved wide GEMM.

M1: exactly 36 down projections use GEMV, the other108 use wide GEMM.
M701/2815: all144 projections use the existing saved wide table.
Other M: original frozen native fallback. No existing adapter is modified.
Capture coverage and outside-graph replay observations are separate evidence.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import torch

from navigation_repair_runtime import NavigationRepairModel
from navigation_latency_w4_hybrid_contract_20261120 import load_hybrid_recipe

MANDATORY_ROWS = (1,701,2815)
PROJECTIONS = ('self_attn.qkv_proj','self_attn.o_proj','mlp.gate_up_proj','mlp.down_proj')


class NavigationLatencyW4HybridModel(NavigationRepairModel):
    def __init__(self,*args,**kwargs):
        self._hybrid_installed=False
        self._hybrid_calls=Counter()
        self._hybrid_capture_calls=Counter()
        self._hybrid_request_start=Counter()
        super().__init__(*args,**kwargs)
        if not self._is_quant or self._method not in ('awq','gptq'):
            raise ValueError('Hybrid candidate requires existing AWQ/GPTQ W4A16 weights')
        if (int(os.environ.get('PROJECT2_REPAIR_WBITS','0')),
                int(os.environ.get('PROJECT2_REPAIR_ABITS','0'))) != (4,16):
            raise ValueError('Hybrid candidate does not quantize activations')
        if os.environ.get('PROJECT2_LATENCY_W4_TABLE') or os.environ.get('PROJECT2_LATENCY_W4_PHASE'):
            raise ValueError('Use only the hybrid recipe, not the old wide opt-in variables')

    def forward(self,*args,**kwargs):
        first=not self._fix_execution_verified
        value=super().forward(*args,**kwargs)
        if first:
            if self._hybrid_installed:
                raise RuntimeError('Hybrid installed twice')
            self._install_hybrid()
        return value

    def _install_hybrid(self):
        import navigation_latency_w4_kernels_20261120 as wide
        import navigation_latency_w4_gemv_20261120 as gemv
        import lmdeploy.pytorch.backends.cuda.awq_modules as backend

        root=Path(__file__).resolve().parents[1]
        recipe_path=Path(os.environ['PROJECT2_LATENCY_HYBRID_TABLE']).resolve()
        launch=json.loads((self._fix_dir/'launch.json').read_text(encoding='utf-8'))
        binding=load_hybrid_recipe(recipe_path,self._method,launch['model_weights_sha256'])
        recipe=binding['recipe']
        wide_path=(root/recipe['wide']['table_path']).resolve()
        receipt=wide.load_table(wide_path,self._method,launch['model_weights_sha256'])
        shapes={(r['m'],r['k'],r['n']) for r in receipt['shapes']}
        replacement=recipe['replacement'];config=replacement['config']
        expected={f'language_model.layers.{i}.{p}' for i in range(36) for p in PROJECTIONS}
        down={f'language_model.layers.{i}.mlp.down_proj' for i in range(36)}
        objects=self._awq_objects
        if set(objects)!=expected or self._awq_loaded_weights!=self._awq_expected_weights:
            raise ValueError('Hybrid requires the complete verified 36-layer/144-projection inventory')
        fallback_receipt=json.loads((self._fix_dir/'repair_execution_audit.json').read_text(encoding='utf-8'))['frozen_launch_table']
        if (not fallback_receipt['frozen'] or fallback_receipt['selection_mode']!='replayed_frozen_table'
                or fallback_receipt['runtime_autotuning_allowed'] or fallback_receipt['runtime_cold_compilation_allowed']
                or not fallback_receipt['actual_m_preserved'] or not fallback_receipt['generic_m_specialization_disabled']):
            raise ValueError('Hybrid requires the exact replayed original native fallback')
        if fallback_receipt['frozen_table_source']['sha256']!=recipe['baseline_table']['sha256']:
            raise ValueError('Hybrid fallback differs from the probe baseline')
        if fallback_receipt['algorithm']!=self._method or fallback_receipt['model_weights_sha256']!=launch['model_weights_sha256']:
            raise ValueError('Hybrid fallback checkpoint identity differs')
        pointers={};warm=[]
        for name,module in objects.items():
            k,n=module.qweight.shape[0],module.scales.shape[1]
            if any((m,k,n) not in shapes for m in MANDATORY_ROWS):
                raise ValueError('Missing measured wide table shape: '+name)
            pointer=module.qweight.data_ptr()
            if pointer in pointers:
                raise ValueError('Projection weights unexpectedly share a pointer')
            pointers[pointer]=name
            rows=wide.prepare_projection(module.qweight,module.scales,module.qzeros)
            value=dict(module=name,k=k,n=n,wide_rows=rows,
                qweight_stride=list(module.qweight.stride()),scales_stride=list(module.scales.stride()),
                qzeros_stride=list(module.qzeros.stride()),decode_route='gemv_down' if name in down else 'wide')
            if name in down:
                if (k,n)!=(replacement['k'],replacement['n']):
                    raise ValueError('GEMV replacement layout differs: '+name)
                x=torch.zeros((1,k),device=module.scales.device,dtype=torch.float16)
                output,compiled=gemv.launch(x,module.qweight,module.scales,module.qzeros,config,return_metadata=True)
                value.update(gemv_config=config,gemv_compiled=compiled)
                del x,output
            warm.append(value)
        torch.cuda.synchronize()
        frozen_wide=wide.freeze()
        def reject_cold_compile(*args,**kwargs):
            raise RuntimeError('Cold hybrid GEMV compilation after actual-module warmup')
        gemv._gemv.compile=reject_cold_compile
        gemv._sum_partials.compile=reject_cold_compile
        fallback=backend.wq_gemm_forward

        def dispatch(x,qweight,qzeros,scales,w_bit=4,group_size=128,bias=None,out_features=0):
            if w_bit!=4 or group_size!=128 or out_features!=scales.shape[1]:
                raise ValueError('Hybrid W4A16 native protocol differs')
            name=pointers.get(qweight.data_ptr())
            if name is None:
                raise ValueError('Unregistered hybrid projection')
            flat=x.flatten(0,-2).contiguous();m,k,n=flat.shape[0],flat.shape[1],scales.shape[1]
            route='gemv_down' if m==1 and name in down else 'wide' if (m,k,n) in shapes else 'baseline_fallback'
            if m in MANDATORY_ROWS and route=='baseline_fallback':
                raise RuntimeError('Mandatory hybrid shape silently fell back')
            self._hybrid_calls[(route,m,name)]+=1
            if m==1 and torch.cuda.is_current_stream_capturing():
                self._hybrid_capture_calls[(route,name)]+=1
            if route=='baseline_fallback':
                return fallback(x,qweight,qzeros,scales,w_bit=w_bit,group_size=group_size,
                                bias=bias,out_features=out_features)
            output=(gemv.launch(flat,qweight,scales,qzeros,config) if route=='gemv_down'
                    else wide.precision_linear(flat,qweight,scales,qzeros))
            if bias is not None: output=output+bias
            output=output.reshape(x.shape[:-1]+(out_features,))
            return output.unsqueeze(0) if output.ndim==2 else output

        backend.wq_gemm_forward=dispatch
        self._hybrid_installed=True
        self._hybrid_module_names=expected
        self._hybrid_down_names=down
        self._hybrid_table_ms={row['m'] for row in receipt['shapes']}
        self._fix_write('hybrid_launch_install.json',dict(
            algorithm=self._method,checkpoint_sha256=launch['model_weights_sha256'],
            recipe_path=str(recipe_path),recipe_sha256=hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            scope='m1_down_gemv_over_wide',recipe=recipe,proof=binding['proof'],
            actual_projection_count=144,gemv_decode_projection_count=36,wide_decode_projection_count=108,
            gemv_decode_modules=sorted(down),wide_decode_modules=sorted(expected-down),
            mandatory_m=list(MANDATORY_ROWS),wide_prefill_m=[701,2815],
            unknown_m_route='baseline_fallback',original_first_synthetic_forward='original frozen baseline before hybrid install',
            actual_module_warmup=warm,wide_table=frozen_wide,
            cold_compile_guards=dict(native=not fallback_receipt['runtime_cold_compilation_allowed'],
                                     wide=frozen_wide['frozen'],gemv=True,gemv_partial_reduce=True),
            graph_evidence='Capture proves the36/108 routing; replay observations come from the existing outside-graph observer',
            model_output_equivalence='pending_actual_full_model_gate'))

    def _fix_before_step(self,kwargs):
        value=super()._fix_before_step(kwargs)
        if value and self._fix_current_step['phase']=='prefill':
            if not self._hybrid_installed:
                raise RuntimeError('Client request before hybrid install')
            self._hybrid_request_start=self._hybrid_calls.copy()
        return value

    def _fix_flush(self):
        if not self._fix_steps:
            return super()._fix_flush()
        index=self._fix_index;steps=list(self._fix_steps)
        prefill=[step['rows'] for step in steps if step['phase']=='prefill']
        if len(prefill)!=1:
            raise RuntimeError('Hybrid requires one full prefill per serial request')
        m=prefill[0];delta=self._hybrid_calls-self._hybrid_request_start
        expected={name:('gemv_down' if m==1 and name in self._hybrid_down_names else
                       'wide' if m in self._hybrid_table_ms else 'baseline_fallback')
                  for name in self._hybrid_module_names}
        actual={}
        for (route,rows,name),count in delta.items():
            if name in actual or rows!=m or expected.get(name)!=route or count!=1:
                raise RuntimeError('Unexpected hybrid per-request Python route/count/M')
            actual[name]=route
        if actual!=expected:
            raise RuntimeError('Hybrid prefill did not cover all144 projections')
        expected_decode={name:('gemv_down' if name in self._hybrid_down_names else 'wide')
                         for name in self._hybrid_module_names}
        captures={route:{name:count for (r,name),count in self._hybrid_capture_calls.items() if r==route}
                  for route in ('gemv_down','wide','baseline_fallback')}
        bad={f'{route}:{name}':count for (route,name),count in self._hybrid_capture_calls.items()
             if expected_decode.get(name)!=route}
        modes=Counter(s.get('execution_mode') for s in steps)
        if modes['replay'] and (set(captures['gemv_down'])!=self._hybrid_down_names or
            set(captures['wide'])!=self._hybrid_module_names-self._hybrid_down_names or bad or captures['baseline_fallback']):
            raise RuntimeError('Graph replay lacks exact36 GEMV/108 wide capture coverage')
        super()._fix_flush()
        self._fix_write(f'hybrid_execution_{index:03d}.json',dict(index=index,prefill_m=prefill,
            python_projection_calls=[dict(route=route,m=rows,module=name,calls=count)
                                     for (route,rows,name),count in sorted(delta.items())],
            prefill_route_projection_counts=dict(Counter(expected.values())),
            gemv_capture_projection_calls=captures['gemv_down'],wide_capture_projection_calls=captures['wide'],
            baseline_capture_projection_calls=captures['baseline_fallback'],unexpected_capture_projection_calls=bad,
            expected_decode_route_projection_counts=dict(gemv_down=36,wide=108),
            outside_graph_execution_modes=dict(modes),graph_replay_python_counters_increment=False,
            prefill_route_coverage_passed=True,
            graph_evidence='Captured36 down GEMV and108 wide projections; replay counts are outside-graph observations, not Python GEMV calls'))
