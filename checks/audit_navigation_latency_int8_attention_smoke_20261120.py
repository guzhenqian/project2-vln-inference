"""Independent new-precision qkv/o actual-smoke audit; no GPU/weight imports.

Only two excluded natural-input smokes are accepted. Their local errors do not
prove whole-model accuracy or original-W4 output equivalence. Timing and memory
are diagnostic observations, never a benchmark or navigation acceptance.
"""
import argparse
import ast
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

import summarize_navigation_latency_20261120 as native
from navigation_latency_int8_attention_contract_20261120 import load_contract, CHECKPOINT, SHAPES

ROOT=Path(__file__).resolve().parents[1]
ENTRY='serve_navigation_latency_int8_attention_20261120.py'
ADAPTER='navigation_latency_int8_attention_adapter_20261120.py'
MODULE_MAP='navigation_latency_int8_attention_module_map_20261120.py'
MODEL_CLASS='navigation_latency_int8_attention_adapter_20261120.NavigationLatencyInt8AttentionModel'
SUFFIX={'qkv':'self_attn.qkv_proj','o':'self_attn.o_proj','gate_up':'mlp.gate_up_proj','down':'mlp.down_proj'}
PLAN=[dict(phase='input_warmup',sample=i,tokens=0,repeat=0) for i in (0,1)]+[dict(phase='flush',sample=0,tokens=18,repeat=0)]
AUTHOR='tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'
AUTHOR_SHA='72efa80174a832cec519eb7bb73ba2f635848b47af4dcaafaa03078e0d8d3a1f'
OUTPUT_SHA='46d634fc4998793f6079ca380b005ce5fae54bfd607325f118d6817597f122c1'
read,digest,require=native.read,native.digest,native.require


def inventory():
    return {f'language_model.layers.{l}.{SUFFIX[p]}':dict(layer=l,projection=p,k=k,n=n)
            for l in range(36) for p,(k,n) in SHAPES.items()}


def sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def error_pass(value):
    require(value['threshold']==.02 and value['finite'] is True and value['passed'] is True and
            all(type(value[k]) in (int,float) and math.isfinite(value[k]) and value[k]>=0 for k in ('relative_l2','max_absolute_error')) and
            value['relative_l2']<=.02,'Actual local/weight error failed the unchanged two-percent gate')


@lru_cache(maxsize=1)
def output_parser():
    # Execute only the two pinned stdlib parsing methods and the unchanged
    # output-gate function, avoiding full policy/PIL/model imports.
    require(digest(ROOT/AUTHOR)==AUTHOR_SHA and digest(ROOT/'checks/navigation_repair_output_gate.py')==OUTPUT_SHA,
            'Pinned author/output parser changed')
    tree=ast.parse((ROOT/AUTHOR).read_text(encoding='utf-8'))
    author=next(v for v in tree.body if isinstance(v,ast.ClassDef) and v.name=='SimpleVLN_Agent')
    methods=[v for v in author.body if isinstance(v,ast.FunctionDef) and v.name in ('extract_result','extract_multi_result')]
    require(len(methods)==2,'Pinned action parser methods absent')
    cls=ast.ClassDef(name='AuthorAgent',bases=[],keywords=[],body=methods,decorator_list=[])
    gate=next(v for v in ast.parse((ROOT/'checks/navigation_repair_output_gate.py').read_text(encoding='utf-8')).body
              if isinstance(v,ast.FunctionDef) and v.name=='inspect_answer')
    scope={'re':re};exec(compile(ast.fix_missing_locations(ast.Module(body=[cls,gate],type_ignores=[])),'<pinned-output-parser>','exec'),scope)
    return scope['inspect_answer']


def natural_output_gate(answer):
    return output_parser()(answer)


def memory_check(value):
    keys={'allocated','reserved','peak_allocated','peak_reserved','device_free','device_total'}
    require(set(value)==keys and all(type(value[k]) is int and value[k]>=0 for k in keys),'Malformed actual CUDA memory receipt')
    require(value['allocated']<=value['reserved']<=value['device_total'] and value['allocated']<=value['peak_allocated'] and
            value['reserved']<=value['peak_reserved'] and value['device_free']<=value['device_total'],'Inconsistent actual CUDA memory counters')


def audit_actual_kv(value,model_config=None):
    require(value['layer_count']==36 and value['allocated_blocks']==value['block_size']==64 and
            value['quant_policy']==0 and value['custom_cache_shapes']==[], 'Actual KV policy/capacity differs')
    shapes=value['expected_shapes']
    require(set(shapes)=={'key','value'},'Actual KV key/value shape inventory differs')
    for shape in shapes.values():
        require(len(shape)==4 and shape[:2]==[64,64] and all(type(v) is int and v>0 for v in shape),'Actual KV shape invalid')
    if model_config is not None:
        cfg=model_config.get('text_config',model_config)
        require(cfg['num_hidden_layers']==36,'Model configuration layer count changed')
        heads=cfg['num_key_value_heads'];dim=cfg.get('head_dim',cfg['hidden_size']//cfg['num_attention_heads'])
        expected={kind:[64,64,heads,cfg.get(letter+'_head_dim') or dim] for kind,letter in (('key','k'),('value','v'))}
        require(shapes==expected,'Actual KV dimensions differ from frozen model config')
    layers=value['layers'];require(len(layers)==36,'Actual KV layer inventory incomplete')
    pointers=set();logical_bytes=0
    for index,layer in enumerate(layers):
        require(layer['layer']==index and layer['tuple_length']==2 and layer['metadata_tail']==[] and len(layer['tensors'])==2,
                'Actual KV layer tuple differs')
        for kind,tensor in zip(('key','value'),layer['tensors']):
            shape=shapes[kind];stride=[math.prod(shape[i+1:]) for i in range(4)]
            require(tensor['kind']==kind and tensor['shape']==shape and tensor['stride']==stride and
                    tensor['dtype']=='torch.float16' and tensor['device']=='cuda:0' and
                    tensor['allocated_blocks']==tensor['block_size']==64,'Actual KV tensor metadata differs')
            require(all(type(tensor[k]) is int and tensor[k]>0 and tensor[k]%16==0 for k in ('data_pointer','storage_pointer')) and
                    tensor['data_pointer'] not in pointers,'KV tensor address invalid or aliased')
            pointers.add(tensor['data_pointer']);logical_bytes+=math.prod(shape)*2
    require(type(value['unique_storage_bytes']) is int and value['unique_storage_bytes']>=logical_bytes,'Actual KV storage is undersized')
    return dict(layers=36,blocks=64,block_size=64,logical_bytes=logical_bytes,unique_storage_bytes=value['unique_storage_bytes'])


def native_fallback_check(table,binding,proof):
    require(table['frozen'] and table['actual_m_preserved'] and table['generic_m_specialization_disabled'] and
            not table['runtime_autotuning_allowed'] and not table['runtime_cold_compilation_allowed'] and
            table['algorithm']=='awq' and table['model_weights_sha256']==CHECKPOINT and table['selection_mode']=='replayed_frozen_table' and
            table['frozen_table_source']['sha256']==proof['baseline_table_sha256'],'Original native fallback identity/config is not frozen')
    require(native.sha_object({k:table[k] for k in ('shapes','representatives','max_rows')})==binding['evidence']['frozen_launch_configuration_sha256'],
            'New-precision fallback table differs from independently audited native reference')


def audit_launch(service,binding,env):
    launch=read(service/'launch.json');evidence=binding['evidence']
    require(launch['group']=='awq_w4a16' and launch['model_name']=='navigation-repair-awq_w4a16' and
            launch['model_weights_sha256']==evidence['model_weights_sha256']==CHECKPOINT,'Smoke model checkpoint differs')
    original_launch=ROOT/evidence['fixed_session']/'service/launch.json'
    require(digest(original_launch)==binding['native_launch_sha256'] and launch['model_config_sha256']==binding['native_model_config_sha256'],
            'Model configuration differs from original native provenance')
    require(launch['candidate_environment']==env,'Candidate environment differs from declared smoke')
    expected=dict(evidence['requested_engine'],custom_module_map=str(ROOT/'checks'/MODULE_MAP))
    require(launch['engine']==expected,'Requested engine differs beyond explicit new-precision mapping')
    require(expected['num_gpu_blocks']==64 and expected['block_size']==64 and not expected['enable_prefix_caching'] and
            expected['quant_policy']==0 and expected['prefill_interval']==2 and expected['max_batch_size']==1,'KV/scheduler contract differs')
    mapping=native.source_constant(ROOT/'checks'/MODULE_MAP,'MODULE_MAP')
    require(mapping=={'Qwen3VLForConditionalGeneration':MODEL_CLASS},'Unexpected new-precision model class')
    for name,value in launch['sources_sha256'].items():
        require(digest(ROOT/'checks'/name)==value,'Runtime source changed: '+name)
        if name in evidence['source_sha256']:require(value==evidence['source_sha256'][name],'Common native source differs: '+name)
    for name in (ENTRY,ADAPTER,MODULE_MAP,'navigation_latency_int8_attention_contract_20261120.py'):
        require(name in launch['sources_sha256'],'Required candidate source not retained')
    contract=launch['int8_attention_contract']
    require(contract['schema']=='navigation_int8_attention/v1' and contract['mode']=='qkv_o' and contract['enabled_count']==72 and
            contract['enabled_projections']==['qkv','o'] and contract['layout']=='nk_transposed' and
            contract['original_checkpoint_changed'] is False and contract['kv_blocks']==contract['kv_block_size']==64 and
            contract['same_w4_output_equivalence_claim'] is False and contract['whole_model_quality']=='not_evaluated',
            'Candidate must remain explicitly new qkv/o-only precision')
    proof=load_contract(contract['micro_probe_directory'],'awq',CHECKPOINT,verify_sources=True)
    require(proof==contract['micro_proof'],'Bound actual micro evidence changed')
    return launch,proof


def audit_engine(service,launch,binding):
    receipt=read(service/'effective_inprocess_engine.json');owner=read(service.parent/'ownership.json')
    effective=dict(launch['engine'],enable_mp_engine=False)
    require(receipt['native_engine_observed'] and receipt['only_bridge_setting_changed']=='enable_mp_engine' and
            receipt['builder_input_config']==launch['engine'] and receipt['delegated_config']==receipt['effective_engine_config']==effective and
            receipt['pid']==owner['service_pid'] and receipt['engine_class']=='lmdeploy.pytorch.engine.engine.Engine' and
            receipt['executor_class']=='lmdeploy.pytorch.engine.executor.uni_executor.UniExecutor' and receipt['model_class']==MODEL_CLASS,
            'Actual new-precision engine/executor/model differs')
    guards=native.source_constant(ROOT/'checks/navigation_latency_inprocess_20261120.py','_SOURCE_SHA256')
    require(receipt['source_sha256']==guards,'In-process source guard differs');native.guarded_package_sources(guards)
    expected=dict(binding['evidence']['engine'],custom_module_map=str(ROOT/'checks'/MODULE_MAP))
    require(effective==expected,'Effective engine changed beyond the explicit precision map')
    execution=read(service/'execution_audit.json');repair=read(service/'repair_execution_audit.json')
    require(execution['first_native_forward_verified'] and execution['native_forward_preserved'] and execution['graph_requested']=='on',
            'Initial original-native warmup/capture protocol absent')
    original=execution['native_awq_receipt']
    require(original['verified'] and original['module_count']==144 and original['loaded_quantized_tensor_count']==756 and
            {v['name'] for v in original['module_calls']}==set(inventory()),'Original packed checkpoint module loading incomplete')
    require(repair['native_forward'] and not repair['extension'] and repair['weight_bits']==4 and repair['activation_bits']==16 and
            repair['input_sha_scope']=='Explicit first two fixed smoke requests only','Original fallback/source input hashing scope differs')
    graph=read(service/'startup_graph_capture.json')
    require(graph['actual_capture_observed'] and graph['counts'].get('capture',0)>0 and graph['graph_keys'],'Actual startup graph capture absent')
    return dict(requested_engine=launch['engine'],effective_engine=effective,model_class=MODEL_CLASS,
        source_guard_sha256=guards,initial_original_native_receipt=original,graph_capture=graph),repair


def audit_install(value,proof,binding):
    inv=inventory();enabled={n for n,v in inv.items() if v['projection'] in ('qkv','o')};mlp=set(inv)-enabled
    require(value['schema']=='navigation_int8_attention_install/v1' and value['method']=='awq' and value['checkpoint_sha256']==CHECKPOINT and
            value['micro_proof']==proof and value['enabled_projections']==['qkv','o'] and value['layout']=='nk_transposed', 'Installation identity differs')
    require(value['enabled_modules']==sorted(enabled) and value['native_prefill_modules']==sorted(mlp) and value['all_decode_modules']==sorted(inv) and
            (value['enabled_count'],value['native_prefill_count'],value['native_decode_count'])==(72,72,144), 'Precision mask/decode coverage differs')
    require(value['prefill_rows']==dict(candidate_min=17,candidate_max=3072,small_rows_route='original_native') and
            value['cold_compile_guard'] is True and value['actual_smoke_gate']=='pending_two_real_inputs' and value['whole_model_quality']=='not_evaluated',
            'Startup must not invent actual-input/quality approval')
    native_fallback_check(value['native_fallback_receipt'],binding,proof)
    native_launch=read(ROOT/binding['evidence']['fixed_session']/'service/launch.json')
    config_path=ROOT/native_launch['recipe']['model']/'config.json'
    require(digest(config_path)==binding['native_model_config_sha256'],'Actual KV model configuration provenance changed')
    kv=audit_actual_kv(value['actual_kv'],read(config_path))
    rows={};resident=0
    for row in value['module_inventory']:
        name=row['module'];require(name in inv and name not in rows,'Installation module missing/duplicate')
        meta=inv[name];require(all(row[k]==v for k,v in meta.items()),'Installed module shape/layer differs')
        k,n=meta['k'],meta['n'];packed=row['original_tensors']
        require(set(packed)=={'qweight','scales','qzeros'},'Installed original tensor inventory differs')
        for suffix,shape,dtype in (('qweight',[k,n//8],'torch.int32'),('scales',[k//128,n],'torch.float16'),('qzeros',[k//128,n//8],'torch.int32')):
            require(packed[suffix]['shape']==shape and packed[suffix]['dtype']==dtype and sha(packed[suffix]['sha256']), 'Malformed actual packed tensor evidence')
        prior_key=f"{meta['layer']}/{meta['projection']}"
        if prior_key in proof['recorded_packed_tensor_receipts']:
            require(packed==proof['recorded_packed_tensor_receipts'][prior_key],'Actual packed tensors differ from tested three-layer fusion')
        require(row['decode_route']=='native_w4','A decode projection changed precision')
        if name in enabled:
            require(row['prefill_route']=='expanded_int8' and row['warm_m17'] is True,'Enabled module was not converted/warmed')
            error_pass(row['weight_error'])
            require(row['expanded_weight']['shape']==[k,n] and row['expanded_weight']['dtype']=='torch.int8' and
                    row['channel_scale']['shape']==[n] and row['channel_scale']['dtype']=='torch.float32' and
                    sha(row['expanded_weight']['sha256']) and sha(row['channel_scale']['sha256']) and row['expanded_stride']==[1,k] and
                    row['resident_bytes']==k*n+n*4,'Expanded actual tensor layout/resident bytes differ')
            prior=proof['recorded_expanded_weight_receipts'].get(prior_key+'/nk_transposed')
            if prior:require(row['expanded_weight']['sha256']==prior['expanded_weight_logical_sha256'] and
                             row['channel_scale']['sha256']==prior['channel_scale_sha256'],'Expanded actual tensors differ from frozen converter')
            memory_check(row['memory_before']);memory_check(row['memory_after']);resident+=row['resident_bytes']
        else:require(row['prefill_route']=='native_w4' and row['weight_error'] is None and 'expanded_weight' not in row,'MLP was requantized')
        rows[name]=row
    require(set(rows)==set(inv) and len(value['module_inventory'])==144,'Incomplete actual module inventory')
    scratch=value['scratch'];storage=scratch['storage']
    expected={'qx':(3072*4096,'torch.int8'),'integer':(3072*6144*4,'torch.int32'),'row_scale':(3072*4,'torch.float32')}
    require(scratch['max_m']==3072 and scratch['tight_flat_prefix_views'] is True and set(storage)==set(expected),'Shared scratch scope differs')
    require(type(scratch['owner_stream']) is int and scratch['owner_stream']>=0,'Missing owner compute stream')
    for name,(size,dtype) in expected.items():
        require(storage[name]['bytes']==size and storage[name]['dtype']==dtype and type(storage[name]['pointer']) is int and
                storage[name]['pointer']>0 and storage[name]['pointer']%16==0,'Scratch capacity/type/alignment differs')
    require(len({v['pointer'] for v in storage.values()})==3 and scratch['total_bytes']==sum(v[0] for v in expected.values()),'Scratch storage overlap/count differs')
    checks={}
    for row in value['shared_invocation_checks']:
        key=(row['module'],row['m']);require(key not in checks and row['module'] in enabled and inv[row['module']]['layer']==0 and
            row['m'] in (17,701,2815,3072) and (row['k'],row['n'])==(inv[row['module']]['k'],inv[row['module']]['n']) and
            row['bitwise_equal_to_frozen_pipeline'] is True and row['independent_fp16_output_storage'] is True and sha(row['input_sha256']),
            'Shared scratch real invocation does not match independent pipeline or output storage')
        checks[key]=row
    require(set(checks)=={(f'language_model.layers.0.{SUFFIX[p]}',m) for p in ('qkv','o') for m in (17,701,2815,3072)},'Require eight shared-scratch shape checks')
    memory_check(value['startup_before']);memory_check(value['startup_after'])
    guards=native.source_constant(ROOT/'checks'/ADAPTER,'PACKAGE_GUARDS')
    require(value['guarded_package_sources']==guards,'Actual adapter package guard differs');native.guarded_package_sources(guards)
    return dict(enabled_count=72,original_packed_modules=144,expanded_resident_bytes=resident,scratch_bytes=scratch['total_bytes'],
        maximum_startup_weight_relative_l2=max(rows[n]['weight_error']['relative_l2'] for n in enabled),
        startup_before=value['startup_before'],startup_after=value['startup_after'],shared_invocation_checks=8,actual_kv=kv,
        checkpoint_scope='Full checkpoint SHA verified by service; runtime hashes all144 packed modules and72 expanded tensors; three-layer receipts also match frozen micro evidence')


def audit_smoke(value,index,input_receipt,expected_kv=None,owner_stream=None):
    inv=inventory();enabled={n for n,v in inv.items() if v['projection'] in ('qkv','o')};m=(701,2815)[index]
    require(value['index']==index and value['status']=='passed_local_error_gate' and value['input_binding']==input_receipt and
            value['enabled_count']==72 and value['relative_l2_limit']==.02 and value['extra_reference_linear_calls']==72 and
            value['extra_model_forwards']==0 and value['whole_model_quality']=='not_evaluated','Actual smoke scope/input/gate differs')
    audit_actual_kv(value['actual_kv'])
    if expected_kv is not None:require(value['actual_kv']==expected_kv,'KV allocation changed between startup and real-input smoke')
    rows={};streams=set()
    for row in value['modules']:
        name=row['module'];require(name in enabled and name not in rows and all(row[k]==v for k,v in inv[name].items()),'Local module identity/coverage differs')
        require(row['m']==m and row['input_dtype']=='torch.float16' and row['input_stride']==[row['k'],1] and
                row['sampling_rows']==[0,m//2,m-1] and sha(row['activation_selected_rows_sha256']) and row['reference_extra_linear_calls']==1,
                'Actual same-input local linear evidence differs')
        metadata=row['output_metadata'];require(set(metadata)=={'candidate','reference'},'Missing output metadata')
        a,b=metadata['candidate'],metadata['reference']
        require(a==b and a['shape'] in ([m,row['n']],[1,m,row['n']]) and a['dtype']=='torch.float16' and a['device']=='cuda:0' and
                a['stride']==[math.prod(a['shape'][i+1:]) for i in range(len(a['shape']))], 'Actual output shape/dtype/device/layout differs')
        execution=row['execution'];stream=execution['owner_stream']
        require(execution['route']=='eager_prefill' and execution['cuda_stream_capturing'] is False and type(stream) is int and stream>=0 and
                execution['current_stream']==stream and (owner_stream is None or stream==owner_stream),'INT8 prefill stream/capture ownership differs')
        streams.add(stream);error_pass(row['comparison']);rows[name]=row
    require(set(rows)==enabled and len(value['modules'])==72,'Require exactly all72 local module comparisons')
    require(len(streams)==1,'Local smokes used multiple compute streams')
    return dict(index=index,input_tokens=m,modules=72,maximum_local_relative_l2=max(v['comparison']['relative_l2'] for v in rows.values()),
        candidate_trajectory_activations=True,original_native_model_trajectory_equivalence=False)


def audit_request_execution(receipt,timing,index,row):
    inv=inventory();m=(701,2815)[index];steps=timing['steps']
    require(timing['index']==index and timing['plan']==PLAN[index] and steps and steps[0]['phase']=='prefill' and
            steps[0]['rows']==m and steps[0]['execution_mode']=='eager','Actual prefill timing/index/layout differs')
    decode=steps[1:]
    require(decode and len(decode)>=row['usage']['completion_tokens']-1 and
            all(s['phase']=='decode' and s['rows']==1 and s['execution_mode']=='replay' for s in decode),'Actual decode graph replay incomplete')
    modes=dict(Counter(s['execution_mode'] for s in steps))
    require(timing['actual_execution_modes']==modes and timing['capture_replay_observed_outside_graph'] is True,'Timing execution receipt differs')
    for step in steps:
        require('model' in step['timers'],'Model interval absent')
        for value in step['timers'].values():
            require(all(type(value[k]) in (int,float) and math.isfinite(value[k]) and value[k]>=0 for k in ('cuda_interval_ms','host_wall_ms')), 'Invalid diagnostic interval')
    require(receipt['index']==index and receipt['prefill_m']==[m] and receipt['prefill_route_counts']=={'int8_prefill':72,'native_prefill':72} and
            receipt['outside_graph_execution_modes']==modes and receipt['graph_replay_python_counters_increment'] is False and
            not receipt['unexpected_capture_projection_calls'] and receipt['extra_reference_calls']==72 and
            receipt['candidate_trajectory_local_error_only'] is True and receipt['whole_model_quality']=='not_evaluated', 'Route/capture/local-gate scope differs')
    calls={}
    for value in receipt['python_projection_calls']:
        name=value['module'];require(name in inv and name not in calls and value['m']==m and value['calls']==1 and
            value['route']==('int8_prefill' if inv[name]['projection'] in ('qkv','o') else 'native_prefill'), 'Wrong actual prefill route/count')
        calls[name]=value
    require(set(calls)==set(inv) and len(receipt['python_projection_calls'])==144,'Actual route inventory incomplete')
    captures=receipt['native_decode_capture_projection_calls']
    require(set(captures)==set(inv) and all(type(v) is int and v>0 for v in captures.values()) and
            set(receipt['real_smoke_passed_indices'])>=set(range(index+1)), 'Original-native144 decode capture or smoke lifecycle incomplete')
    return dict(index=index,input_tokens=m,actual_decode_forwards=len(decode),execution_modes=modes,
        diagnostic_prefill_model_ms=steps[0]['timers']['model']['cuda_interval_ms'],
        diagnostic_decode_model_ms=sum(v['timers']['model']['cuda_interval_ms'] for v in decode),
        diagnostic_only_includes_extra_reference_and_hashing=True)


def audit_response(row,index):
    require(row['index']==index and row['spec']==PLAN[index] and row['http_mode']=='nonstream' and row['idle_seconds_outside_timer']==0,
            'Request plan/mode differs')
    answer=row['response'];body=row['response_body_utf8'];require(json.loads(body)==answer and
        hashlib.sha256(body.encode('utf-8')).hexdigest()==row['response_bytes_sha256'],'Raw HTTP response differs')
    require(answer['model']=='navigation-repair-awq_w4a16' and len(answer['choices'])==1 and row['usage']==answer['usage'] and
            row['usage']['prompt_tokens']==(2815 if index==1 else 701),'Response model/input differs')
    tokens=answer['choices'][0]['message'].get('gen_tokens') or []
    require(tokens==row['output_tokens'] and len(tokens)==row['usage']['completion_tokens']>0 and
            all(type(t) is int for t in tokens) and native.sha_object(tokens)==row['output_tokens_sha256'],'Returned token evidence differs')
    if index<2:
        result=natural_output_gate(answer);require(result['passed'],'Natural output action structure failed');return result
    require(len(tokens)==18 and answer['choices'][0]['finish_reason']=='length','Tail must flush with exactly18 forced tokens')
    return dict(passed=True,scope='Excluded tail only')


def audit_completed(run):
    protocol=read(run/'protocol.json');binding=read(run/'native_reference_binding.json');service=run/'service'
    require(protocol['schema']=='navigation_int8_attention_actual_smoke/v1' and protocol['measured_requests']==0 and
            protocol['excluded_input_warmups']==2 and protocol['tail_requests']==1 and protocol['no_latency_claim'] and protocol['no_quality_claim'],
            'This auditor only accepts the dedicated excluded smoke protocol')
    require(read(service/'request_plan.json')==PLAN,'Smoke plan must be exactly two natural inputs plus tail')
    for name,value in protocol['source_sha256'].items():
        require(digest(run/'source_snapshot'/name)==value and digest(ROOT/name)==value,'Frozen smoke source/snapshot differs: '+name)
    require(binding==protocol['native_reference_binding'],'Native provenance binding changed')
    launch,proof=audit_launch(service,binding,protocol['candidate_environment'])
    for name,value in launch['sources_sha256'].items():require(protocol['source_sha256']['checks/'+name]==value,'Launch source not in frozen smoke snapshot')
    engine,repair=audit_engine(service,launch,binding)
    install=read(service/'int8_attention_install.json');installation=audit_install(install,proof,binding)
    native_fallback_check(repair['frozen_launch_table'],binding,proof)
    require(repair['frozen_launch_table']==install['native_fallback_receipt'],'Original fallback changed after conversion')
    rows=read(run/'requests.json');require(len(rows)==3,'Require both natural requests and tail')
    responses=[audit_response(row,i) for i,row in enumerate(rows)]
    require(sorted(p.name for p in service.glob('input_*.json'))==['input_2815.json','input_701.json'],'Actual input SHA scope changed')
    smokes=[];execution=[]
    for i,m in enumerate((701,2815)):
        value=read(service/f'input_{m}.json')
        require(value['request_index']==i and {k:value[k] for k in ('input_ids','pixel_values')}==binding['evidence']['actual_input_tensors'][str(m)],
                'Candidate actual input tensors differ from native provenance')
        smokes.append(audit_smoke(read(service/f'int8_attention_smoke_{i:03d}.json'),i,value,install['actual_kv'],install['scratch']['owner_stream']))
        execution.append(audit_request_execution(read(service/f'int8_attention_execution_{i:03d}.json'),read(service/f'timing_{i:03d}.json'),i,rows[i]))
    require(sorted(p.name for p in service.glob('int8_attention_smoke_*.json'))==['int8_attention_smoke_000.json','int8_attention_smoke_001.json'],
            'Extra/missing local smoke or retained failure')
    require(sorted(p.name for p in service.glob('int8_attention_execution_*.json'))==['int8_attention_execution_000.json','int8_attention_execution_001.json'],
            'Unexpected flushed formal requests')
    require(not list(service.glob('*failed*.json')) and not (run/'failure.json').exists(),'Retained failure prevents approval')
    cleanup=read(run/'cleanup.json');owner=read(run/'ownership.json')
    require(cleanup['service_pid']==owner['service_pid'] and cleanup['owned_process_exited'] is True,'Owned service cleanup not confirmed')
    return dict(status='verified_actual_smoke',run=run.relative_to(ROOT).as_posix(),installation=installation,engine=engine,
        local_smokes=smokes,requests=execution,natural_output_structure=responses[:2],
        gpu_samples=read(run/'gpu_samples.json'),measured_requests=0,no_latency_claim=True,no_quality_claim=True,
        benefit_claim_allowed=False,full_model_accuracy_approved=False,original_w4_output_equivalence_claim=False,
        auditor_sha256=digest(__file__),micro_proof_sha256=native.sha_object(proof),
        scope='Two excluded candidate-trajectory natural smokes,72 local references each. No full fixed/replay/navigation acceptance.')


def audit_run(run):
    run=Path(run).resolve();require(run.is_relative_to(ROOT/'logs'),'Smoke must be under project logs')
    evidence={p.relative_to(run).as_posix():digest(p) for p in sorted(run.glob('*.json')) if p.name not in ('audit.json',)}
    evidence.update({p.relative_to(run).as_posix():digest(p) for p in sorted((run/'service').glob('*.json'))})
    try:result=audit_completed(run)
    except Exception as error:
        result=dict(status='failed_or_incomplete',run=run.relative_to(ROOT).as_posix(),audit_error=repr(error),
            retained_failure=read(run/'failure.json') if (run/'failure.json').exists() else None,
            measured_requests=0,no_latency_claim=True,no_quality_claim=True,benefit_claim_allowed=False,
            full_model_accuracy_approved=False,original_w4_output_equivalence_claim=False,auditor_sha256=digest(__file__))
    result['evidence_sha256']=evidence
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve();require(output.is_relative_to(ROOT) and not output.exists(),'Use fresh project output')
    result=audit_run(args.run);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],output=str(output))))
    if result['status']!='verified_actual_smoke':raise SystemExit(2)


if __name__=='__main__':main()
