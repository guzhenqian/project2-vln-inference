"""Independent, cold-per-episode CPU visual-cache experiment on all 343 inputs.

The immutable uncached replay is neither edited nor relabeled. Fixed studies
bind the native checkpoint/kernel/input contract only; this separate entry and
its cache-aware execution receipts are audited explicitly. Payload preparation
precedes service startup. Numerical gates and all measured SHA/copy costs stay
inside their respective HTTP requests. No cache benefit is claimed by this tool.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time
import traceback
from urllib.request import Request

from run_navigation_latency_replay_20261120 import load_helpers, build_plan


ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'checks/serve_navigation_latency_vision_cache_v2_20261120.py'
MAP = 'navigation_latency_vision_cpu_cache_module_map_v2_20261120.py'
MODULE = 'navigation_latency_vision_cpu_cache_v2_20261120.py'
EPISODES = ['1250','159','1413','558','1123','384']
COUNT = 343
H = None


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def build_cache_plan(entries, plan, native_sha, manifest_sha):
    requests = []
    for index,spec in enumerate(plan):
        entry = entries[index-2] if 2 <= index < len(plan)-1 else None
        sample = spec['sample'] if entry is None else None
        requests.append(dict(index=index,spec=spec,
            expected_images=entry['image_count'] if entry else (2 if sample == 0 else 9),
            expected_input_tokens=entry['source_prompt_tokens'] if entry else (701 if sample == 0 else 2815),
            replay_index=entry['index'] if entry else None,episode_id=entry['episode_id'] if entry else None,
            episode_start=entry['is_episode_start'] if entry else False,
            reset_before=entry['is_episode_start'] if entry else False))
    return dict(schema='navigation_vision_cpu_cache_plan/v1',native_plan_sha256=native_sha,
        replay_manifest_sha256=manifest_sha,requests=requests,
        reset_contract='Actual LRU clear before each first model call of six episodes, including after both excluded warmups')


def normalize(config):
    sessions = H.normalize_sessions(config,ENTRY)
    defaults = dict(PROJECT2_LATENCY_EXECUTOR='uni',PROJECT2_LATENCY_PREFILL_INTERVAL='2',
        PROJECT2_LATENCY_INPROCESS='1',PROJECT2_LATENCY_VISION_CACHE_CAPACITY='96',PROJECT2_LATENCY_VISION_CACHE_VALIDATE='1')
    allowed = set(defaults)|{'PROJECT2_LATENCY_VISION_CACHE'}
    keys = set()
    for session in sessions:
        require(session['group'] in ('fp16','awq_w4a16') and not session.get('extra_args'), 'Only FP16/AWQ cache experiment without extra args')
        require((ROOT/session['service_entry']).resolve() == (ROOT/ENTRY).resolve(), 'Use the independent cache entry')
        env = session.setdefault('env',{})
        require(set(env) <= allowed, 'Unregistered cache experiment environment')
        for key,value in defaults.items():
            env.setdefault(key,value)
            if key != 'PROJECT2_LATENCY_VISION_CACHE_CAPACITY':
                require(env[key] == value, 'Cache experiment requires '+key+'='+value)
        require(env.get('PROJECT2_LATENCY_VISION_CACHE') in ('off','cpu') and env['PROJECT2_LATENCY_VISION_CACHE_CAPACITY'] in ('96','200'),
                'Specify off/cpu mode and capacity96/200')
        require((session['group'] == 'fp16') == (not bool(session.get('frozen_table'))), 'AWQ requires its exact frozen launch table')
        key = object_sha(H.session_key(session)); require(key not in keys,'Repeated whole-sequence configuration is forbidden'); keys.add(key)
    return sessions


def reference_bindings(studies,references,sessions):
    refs = H.normalize_sessions(references,'checks/serve_navigation_latency_20261120.py')
    result = []
    for session in sessions:
        candidates = [r for r in refs if r['group'] == session['group']]
        require(len(candidates) == 1, 'Exactly one native reference configuration required per group')
        reference = candidates[0]; env = reference.get('env',{})
        require(env.get('PROJECT2_LATENCY_EXECUTOR') == 'uni' and env.get('PROJECT2_LATENCY_PREFILL_INTERVAL') == '2' and
                env.get('PROJECT2_LATENCY_INPROCESS') == '1' and not any('CACHE' in k for k in env), 'Reference must be native inprocess2 without cache')
        require(H.session_key(reference)['frozen_table_sha256'] == H.session_key(session)['frozen_table_sha256'], 'Reference frozen W4 table differs')
        evidence = H.fixed_evidence(studies,reference)
        require(not evidence['wide_evidence'] and not evidence['vision_qmax_evidence'], 'Cache reference must use unchanged native vision/W4 kernels')
        native_launch_path = ROOT/evidence['fixed_session']/'service/launch.json'
        native_launch = H.read(native_launch_path)
        require(native_launch['model_weights_sha256'] == evidence['model_weights_sha256'] and
                native_launch['engine'] == evidence['requested_engine'], 'Bound native launch differs from audited evidence')
        result.append(dict(reference_session=reference,evidence=evidence,
            native_launch_sha256=H.digest(native_launch_path),native_model_config_sha256=native_launch['model_config_sha256'],
            scope='Native configuration provenance only; cache entry is separately audited and fixed latencies are not replay measurements'))
    return result


def match_launch(service,session,binding):
    evidence = binding['evidence']; launch = H.read(service/'launch.json')
    require(launch['group'] == session['group'] and launch['candidate_environment'] == session['env'], 'Actual cache launch configuration differs')
    expected = dict(evidence['requested_engine'],custom_module_map=str(ROOT/'checks'/MAP))
    require(launch['engine'] == expected, 'Cache entry changes engine settings beyond its explicit module-map name')
    # The new map deliberately has the same native class target, but a distinct
    # filename prevents the frozen native auditor accepting this experiment.
    import ast
    tree = ast.parse((ROOT/'checks'/MAP).read_text())
    values = [ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'MODULE_MAP' for t in n.targets)]
    require(values == [{'Qwen3VLForConditionalGeneration':'navigation_repair_runtime.NavigationRepairModel'}], 'Cache map changed native model class')
    require(launch['model_weights_sha256'] == evidence['model_weights_sha256'], 'Checkpoint changed')
    require(H.digest(ROOT/evidence['fixed_session']/'service/launch.json') == binding['native_launch_sha256'] and
            launch['model_config_sha256'] == binding['native_model_config_sha256'], 'Model config differs from the bound native launch')
    for name,sha in evidence['source_sha256'].items():
        if name in launch['sources_sha256']:
            require(launch['sources_sha256'][name] == sha, 'Native dependency differs from reference: '+name)
    for name,sha in launch['sources_sha256'].items():
        require(H.digest(ROOT/'checks'/name) == sha, 'Cache launch source changed: '+name)
    require(H.read(service/'execution_audit.json')['first_native_forward_verified'] and
            H.read(service/'startup_graph_capture.json')['actual_capture_observed'], 'Native execution/graph proof absent')
    cache = launch['vision_cache_contract']
    require(cache['mode'] == session['env']['PROJECT2_LATENCY_VISION_CACHE'] and
            cache['capacity'] == int(session['env']['PROJECT2_LATENCY_VISION_CACHE_CAPACITY']) and
            cache['storage'] == 'cpu_pageable' and cache['full_visual_tokens'] and cache['deepstack_arrays'] == 3 and
            not cache['prefix_kv_cache'] and cache['request_plan_sha256'] == H.digest(service/'cache_request_plan.json'), 'Actual cache contract differs')
    return launch


def match_execution(service,launch,binding):
    evidence = binding['evidence']; receipt = H.read(service/'effective_inprocess_engine.json')
    expected = dict(launch['engine'],enable_mp_engine=False)
    require(receipt['native_engine_observed'] and receipt['only_bridge_setting_changed'] == 'enable_mp_engine' and
            receipt['builder_input_config'] == launch['engine'] and receipt['delegated_config'] == expected and
            receipt['effective_engine_config'] == expected and receipt['pid'] == H.read(service.parent/'ownership.json')['service_pid'] and
            receipt['engine_class'] == 'lmdeploy.pytorch.engine.engine.Engine' and
            receipt['executor_class'] == 'lmdeploy.pytorch.engine.executor.uni_executor.UniExecutor' and
            receipt['model_class'] == 'navigation_repair_runtime.NavigationRepairModel', 'Observed native inprocess execution differs')
    require(dict(evidence['engine'],custom_module_map=str(ROOT/'checks'/MAP)) == expected, 'Observed reference engine differs beyond explicit map')
    from summarize_navigation_latency_20261120 import source_constant, guarded_package_sources
    guards = source_constant(ROOT/'checks/navigation_latency_inprocess_20261120.py','_SOURCE_SHA256')
    require(receipt['source_sha256'] == guards,'Inprocess installed source guard differs'); guarded_package_sources(guards)
    repair = H.read(service/'repair_execution_audit.json'); execution = H.read(service/'execution_audit.json')
    require(repair['native_forward'] and not repair['extension'] and repair['activation_bits'] == 16 and
            repair['input_sha_scope'] == 'Explicit first two fixed smoke requests only','Native language/precision contract changed')
    if launch['group'] == 'awq_w4a16':
        native = execution['native_awq_receipt']; table = repair['frozen_launch_table']
        require(native['verified'] and native['module_count'] == 144 and native['loaded_quantized_tensor_count'] == 756, 'Actual native AWQ weights unverified')
        require(table['frozen'] and table['actual_m_preserved'] and table['generic_m_specialization_disabled'] and
                not table['runtime_autotuning_allowed'] and not table['runtime_cold_compilation_allowed'] and
                table['algorithm'] == 'awq' and table['model_weights_sha256'] == evidence['model_weights_sha256'] and
                table['selection_mode'] == 'replayed_frozen_table' and
                table['frozen_table_source']['sha256'] == H.read(service.parent/'command.json')['configuration']['frozen_table_sha256'] and
                object_sha({k:table[k] for k in ('shapes','representatives','max_rows')}) == evidence['frozen_launch_configuration_sha256'],
                'Actual native AWQ launch table changed')
    return dict(effective_engine=expected,inprocess_receipt_sha256=H.digest(service/'effective_inprocess_engine.json'),
        native_execution_sha256=H.digest(service/'execution_audit.json'),repair_execution_sha256=H.digest(service/'repair_execution_audit.json'),
        scope='Cache-aware native execution check; not the immutable native latency audit')


def audit_feature_comparison(comparison,output_features):
    labels = ['main','deepstack0','deepstack1','deepstack2']
    require(comparison['passed'] is True and len(output_features) == 4 and
            [r['feature'] for r in comparison['features']] == labels,'Numerical comparison lacks exactly main plus three DeepStack arrays')
    for row,expected in zip(comparison['features'],output_features):
        require(all(row[key] is True for key in ('metadata_equal','finite','distinct_storage','torch_equal','bitwise_equal','passed')) and
                row['reference_shape'] == row['candidate_shape'] == expected['shape'] and
                row['reference_dtype'] == row['candidate_dtype'] == expected['dtype'] == 'torch.float16' and
                row['reference_device'] == row['candidate_device'] == expected['device'] and expected['device'].startswith('cuda') and
                isinstance(row['max_abs_error'],(int,float)) and not isinstance(row['max_abs_error'],bool) and row['max_abs_error'] == 0 and
                isinstance(row['relative_l2'],(int,float)) and not isinstance(row['relative_l2'],bool) and row['relative_l2'] == 0,
                'Per-feature exact bytewise/finite/metadata/independent-storage evidence failed: '+row['feature'])


def audit_numeric_execution(execution,reference,miss_keys=None):
    keys = reference['ordered_key_sha256']; distinct = list(dict.fromkeys(keys)); slots = len(keys)
    require(execution['ordered_key_sha256'] == keys and execution['grids'] == reference['grids'] and
            execution['output_features'] == reference['output_features'] and execution['image_slots'] == slots and
            execution['distinct_images'] == len(distinct) and execution['within_request_duplicate_slots'] == slots-len(distinct) and
            execution['historical_hit_distinct']+execution['encoded_distinct_misses'] == len(distinct) and
            execution['historical_hit_slots']+execution['encoded_distinct_misses']+execution['within_request_deduplicated_miss_slots'] == slots and
            execution['native_vision_forward_calls'] == int(bool(execution['encoded_distinct_misses'])),
            'Numeric execution changed original image order/grid/four-feature layout or hit/miss accounting')
    require(all(isinstance(execution[k],int) and not isinstance(execution[k],bool) and 0 <= execution[k] <= slots for k in
                ('image_slots','distinct_images','within_request_duplicate_slots','historical_hit_distinct',
                 'encoded_distinct_misses','historical_hit_slots','within_request_deduplicated_miss_slots','native_vision_forward_calls')),
            'Numerical execution counters must be bounded nonnegative integers')
    require(len(reference['grids']) == slots and all(len(grid) == 3 and grid[0] == 1 and grid[1] > 0 and grid[2] > 0 for grid in reference['grids']),
            'Numerical execution has invalid static-image grids')
    merged = sum(math.prod(grid)//4 for grid in reference['grids'])
    require(all(meta['shape'] == [merged,2560] for meta in reference['output_features']), 'Numeric arrays do not retain all merged image tokens')
    if miss_keys is not None:
        missed = set(miss_keys); miss_slots = sum(key in missed for key in keys)
        require(len(missed) == len(miss_keys) and missed <= set(distinct) and
                execution['encoded_distinct_misses'] == len(missed) and execution['historical_hit_distinct'] == len(distinct)-len(missed) and
                execution['historical_hit_slots'] == slots-miss_slots and
                execution['within_request_deduplicated_miss_slots'] == miss_slots-len(missed) and
                execution['native_vision_forward_calls'] == int(bool(missed)), 'Actual numeric partial-hit mask differs from requested keys')


def numeric_gate(service,launch):
    mode = launch['vision_cache_contract']['mode']; expected = [] if mode == 'off' else ['vision_cache_numeric_0000.json','vision_cache_numeric_0001.json']
    require(sorted(p.name for p in service.glob('vision_cache_numeric_*.json')) == expected,'Missing or extra numerical comparisons')
    result = []
    for index,name in enumerate(expected):
        value = H.read(service/name)
        require(value['kind'] == 'warmup_numeric_gate' and value['request_index'] == index and value['status'] == 'passed' and
                value['mode'] == mode and value['patch_module_sha256'] == launch['sources_sha256'][MODULE] and
                value['plan_sha256'] == launch['vision_cache_contract']['request_plan_sha256'] and
                value['reference_vs_candidate']['passed'] and value['reference_vs_all_hit_cpu_roundtrip']['passed'] and
                value['all_hit_execution']['native_vision_forward_calls'] == 0 and value['extra_language_forwards'] == 0,
                'Excluded strict visual/DeepStack numerical gate failed or was not all-hit roundtripped')
        initial = value['initial_execution']; unique_keys = list(dict.fromkeys(initial['ordered_key_sha256']))
        require(initial['image_slots'] == (2 if index == 0 else 9),'Numerical warmup image scope differs')
        audit_numeric_execution(initial,initial)
        audit_feature_comparison(value['reference_vs_candidate'],initial['output_features'])
        audit_numeric_execution(value['all_hit_execution'],initial,[])
        audit_feature_comparison(value['reference_vs_all_hit_cpu_roundtrip'],initial['output_features'])
        masks = value['eligible_partial_distinct_miss_counts']
        require([gate['distinct_miss_count'] for gate in value['partial_hit_gates']] == masks and
                masks == ([n for n in (1,3,4,8) if n < value['initial_execution']['distinct_images']] if index == 1 else []) and
                all(gate['passed'] and gate['reference_vs_partial_hit']['passed'] and
                    gate['execution']['encoded_distinct_misses'] == gate['distinct_miss_count'] and
                    gate['execution']['native_vision_forward_calls'] == 1 for gate in value['partial_hit_gates']),
                'Partial-hit miss-batch numerical coverage incomplete or failed')
        for gate in value['partial_hit_gates']:
            require(gate['evicted_key_sha256'] == unique_keys[:gate['distinct_miss_count']], 'Partial numeric mask is not the prescribed distinct-key prefix')
            audit_numeric_execution(gate['execution'],initial,gate['evicted_key_sha256'])
            audit_feature_comparison(gate['reference_vs_partial_hit'],initial['output_features'])
        result.append(dict(path=name,sha256=H.digest(service/name),value=value))
    return result


def inspect_stage(service,index,row,cache_plan,launch):
    timing = H.read(service/f'timing_{index:03d}.json'); steps = timing['steps']; pre = steps[0]
    execution = H.read(service/f'vision_cache_request_{index:04d}.json'); expected = cache_plan['requests'][index]
    require(timing['index'] == index and timing['plan'] == row['spec'] and pre['phase'] == 'prefill' and
            pre['rows'] == row['usage']['prompt_tokens'] == expected['expected_input_tokens'] and pre['rows'] <= 3072,
            'Cache timing/input/plan mapping differs')
    require(execution['request_index'] == index and execution['spec'] == expected and
            execution['plan_sha256'] == launch['vision_cache_contract']['request_plan_sha256'] and
            execution['patch_module_sha256'] == launch['sources_sha256'][MODULE] and execution['mode'] == launch['vision_cache_contract']['mode'] and
            execution['image_slots'] == expected['expected_images'] and execution['completed_host_model_forward'], 'Actual cache execution differs from plan/source')
    mode = launch['vision_cache_contract']['mode']; formal = expected['spec']['phase'] == 'measured'
    if mode == 'cpu' and formal:
        require(execution['numerical_gate']['status'] == 'not_performed_for_formal_request' and
                execution['native_vision_forward_calls'] in (0,1) and execution['cache_entries'] <= execution['cache_capacity'] and
                execution['encoded_distinct_misses']+execution['historical_hit_distinct'] == execution['distinct_images'] and
                execution['historical_hit_slots']+execution['encoded_distinct_misses']+execution['within_request_deduplicated_miss_slots'] == execution['image_slots'],
                'Formal cache counts/numerical scope differs')
        if expected['episode_start']:
            require(execution['historical_hit_slots'] == 0, 'Episode begins with an unexpectedly warm cache')
    if expected['reset_before']:
        reset = H.read(service/f'vision_cache_reset_{index:04d}.json')
        require(reset['kind'] == 'actual_cache_clear' and reset['request_index'] == index and reset['episode_id'] == expected['episode_id'] and
                reset['plan_sha256'] == execution['plan_sha256'] and reset['generation'] == execution['generation'] and
                reset['after'] == dict(entries=0,bytes=0), 'Episode reset was not observed')
    decode = steps[1:]
    require(decode and len(steps) >= row['usage']['completion_tokens'] and all(s['phase'] == 'decode' and s['rows'] == 1 and s['execution_mode'] == 'replay' for s in decode),
            'Decode graph sequence incomplete or changed')
    for step in steps:
        for timer in step['timers'].values():
            require(all(math.isfinite(timer[k]) and timer[k] >= 0 for k in ('cuda_interval_ms','host_wall_ms')), 'Invalid timer')
    vision = pre['timers'].get('vision'); total = pre['timers']['vision_cache_total']
    if formal:
        require(bool(vision) == bool(execution['native_vision_forward_calls']), 'Native encoder timer differs from actual miss execution')
    return dict(index=index,replay_index=row.get('frozen_replay_index'),episode_id=row.get('episode_id'),source_step=row.get('source_step'),
        http_total_s=row['http_total_s'],prompt_tokens=row['usage']['prompt_tokens'],output_tokens=row['output_tokens'],output_tokens_sha256=row['output_tokens_sha256'],
        prefill_language_ms=pre['timers']['language']['cuda_interval_ms'],prefill_model_ms=pre['timers']['model']['cuda_interval_ms'],
        prefill_vision_encoder_ms=vision['cuda_interval_ms'] if vision else 0.,
        prefill_visual_cache_cuda_interval_ms=total['cuda_interval_ms'],prefill_visual_cache_host_ms=total['host_wall_ms'],
        actual_decode_forwards=len(decode),decode_model_total_ms=sum(s['timers']['model']['cuda_interval_ms'] for s in decode),
        decode_model_ms_per_actual_forward=statistics.mean(s['timers']['model']['cuda_interval_ms'] for s in decode),
        execution_modes=dict(Counter(s['execution_mode'] for s in decode)),cache_execution=execution,
        timing_sha256=H.digest(service/f'timing_{index:03d}.json'),cache_execution_sha256=H.digest(service/f'vision_cache_request_{index:04d}.json'))


def post(port,model,spec,wire,proof,entry):
    started = time.perf_counter()
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'},method='POST')
    with H.OPENER.open(req,timeout=300) as response:
        raw = response.read(); answer = json.loads(raw)
    ended = time.perf_counter(); choice = answer['choices'][0]; tokens = choice['message'].get('gen_tokens') or []; usage = answer['usage']
    require(answer['model'] == model and len(answer['choices']) == 1 and usage['prompt_tokens'] == entry['source_prompt_tokens'] and
            len(tokens) == usage['completion_tokens'] == 18 and choice['finish_reason'] == 'length' and
            all(isinstance(t,int) and not isinstance(t,bool) for t in tokens),'Replay input/forced18 response changed')
    return dict(index=entry['index']+2,spec=spec,http_mode='nonstream',request_sha256=proof['wire_bytes_sha256'],input_proof=proof,
        semantic_payload_sha256=entry['semantic_payload_sha256'],normalized_payload_bytes_sha256=entry['normalized_payload_bytes_sha256'],
        frozen_replay_index=entry['index'],episode_id=entry['episode_id'],source_step=entry['step'],
        episode_start=entry['is_episode_start'],episode_end=entry['is_episode_end'],source_image_sha256=entry['request_image_sha256'],
        usage=usage,response=answer,response_body_utf8=raw.decode('utf-8'),response_bytes_sha256=hashlib.sha256(raw).hexdigest(),
        output_tokens=tokens,output_tokens_sha256=object_sha(tokens),output_text_sha256=object_sha(choice['message']['content'] or ''),
        http_total_s=ended-started,client_perf_start_s=started,client_perf_end_s=ended,idle_seconds_outside_timer=0)


def run_session(run,number,session,replay_dir,manifest,binding,port,timeout,smoke_only=False):
    directory = run/f"{number:02d}_{session['label']}"; service = directory/'service'; service.mkdir(parents=True)
    entries = manifest['requests']; full_plan = build_plan(entries)
    plan = full_plan[:2]+[full_plan[-1]] if smoke_only else full_plan
    H.write(service/'request_plan.json',plan)
    cache_plan = build_cache_plan(entries,plan,H.digest(service/'request_plan.json'),H.digest(replay_dir/'manifest.json'))
    cache_plan['scope'] = 'excluded_smoke_only_no_historical_replay' if smoke_only else 'complete343_cold_per_episode_replay'
    H.write(service/'cache_request_plan.json',cache_plan)
    prepared = [] if smoke_only else [H.load_request(replay_dir,e,session['model_name']) for e in entries]
    for entry,(wire,proof) in zip(entries,prepared):
        require(proof['model_only_substitution'] and proof['semantic_payload_sha256'] == entry['semantic_payload_sha256'] and
                proof['normalized_payload_bytes_sha256'] == entry['normalized_payload_bytes_sha256'] and
                hashlib.sha256(wire).hexdigest() == proof['wire_bytes_sha256'],'Prepared wire differs from frozen payload')
    payloads = {s:json.loads(H.prepare_input(1.,s,'navigation')[-1]) for s in (0,1)}
    env = {k:v for k,v in os.environ.items() if not k.startswith('PROJECT2_')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'); env.update(session['env'])
    if session.get('frozen_table'): env['PROJECT2_W4A16_REPAIR_TABLE'] = str((ROOT/session['frozen_table']).resolve())
    command = [H.PYTHON,'-B','-u',str(ROOT/ENTRY),'--group',session['group'],'--run-dir',str(service),'--port',str(port)]
    H.write(directory/'native_reference_binding.json',binding); H.write(directory/'command.json',dict(argv=command,configuration=H.session_key(session)))
    with socket.socket() as probe: probe.bind(('127.0.0.1',port))
    child = None; rows = []
    try:
        with (directory/'service_console.log').open('x') as console:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
        H.write(directory/'ownership.json',dict(service_pid=child.pid,service_process_group=child.pid,controller_pid=os.getpid(),port=port,created_utc=utc()))
        deadline = time.monotonic()+timeout
        while True:
            require(child.poll() is None,'Service exited during startup')
            try:
                with H.OPENER.open(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:
                    require(json.load(response)['data'][0]['id'] == session['model_name'],'Wrong served model')
                break
            except OSError:
                require(time.monotonic() < deadline,'Service startup timed out'); time.sleep(2)
        launch = match_launch(service,session,binding)
        for index in (0,1):
            row = H.request(port,session['model_name'],plan[index],payloads,'nonstream',0.); row['index'] = index; rows.append(row)
            gate = H.inspect_answer(row['response']); H.write(directory/f'natural_output_gate_{index}.json',gate); require(gate['passed'],'Natural warmup failed')
        H.write(directory/'input_warmups.json',rows)
        for index,length in enumerate((701,2815)):
            value = H.read(service/f'input_{length}.json')
            require(value['request_index'] == index and {k:value[k] for k in ('input_ids','pixel_values')} == binding['evidence']['actual_input_tensors'][str(length)],'Warmup actual tensors differ')
        proof = numeric_gate(service,launch); match_execution(service,launch,binding)
        if smoke_only:
            # A distinct protocol: do not truncate or relabel a 343-request run.
            # No historical payload has been loaded, posted, or measured.
            tail = H.request(port,session['model_name'],plan[-1],payloads,'nonstream',0.)
            tail['index'] = 2; rows.append(tail)
            H.write(directory/'requests.json',rows); H.write(directory/'tail_request.json',tail)
            require(len(rows) == 3 and {p.name for p in service.glob('input_*.json')} == {'input_701.json','input_2815.json'},'Smoke request/input scope changed')
            warmup_stages = [inspect_stage(service,i,row,cache_plan,launch) for i,row in enumerate(rows[:2])]
            require(not list(service.glob('vision_cache_reset_*.json')) and len(list(service.glob('vision_cache_request_*.json'))) == 2,
                    'Smoke produced formal reset or unexpected execution receipts')
            require(numeric_gate(service,launch) == proof,'Smoke numeric proof changed on tail')
            H.write(directory/'excluded_warmup_stages.json',warmup_stages)
            result = dict(status='smoke_only',session=session,native_reference_binding=binding,
                execution_contract=match_execution(service,launch,binding),numeric_proof=proof,
                replay_manifest_sha256=H.digest(replay_dir/'manifest.json'),
                cache_plan_sha256=H.digest(service/'cache_request_plan.json'),
                excluded_input_warmups=2,tail_requests=1,historical_payloads_loaded=0,historical_requests_executed=0,
                measured_requests=0,no_latency_claim=True,no_quality_claim=True,complete_replay_evidence=False,
                benefit_claim_allowed=False,production_ready=False,
                scope='Dedicated import/memory/bytewise smoke only; never complete replay or navigation evidence')
            H.write(directory/'result.json',result); return result
        previous_end = None
        with (directory/'replay_responses.jsonl').open('x',encoding='utf-8') as stream:
            for entry,(wire,input_proof) in zip(entries,prepared):
                row = post(port,session['model_name'],plan[entry['index']+2],wire,input_proof,entry)
                row['client_gap_since_previous_replay_s'] = None if previous_end is None else row['client_perf_start_s']-previous_end
                previous_end = row['client_perf_end_s']; rows.append(row)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        tail = H.request(port,session['model_name'],plan[-1],payloads,'nonstream',0.); tail['index'] = len(plan)-1; rows.append(tail)
        H.write(directory/'requests.json',rows); H.write(directory/'tail_request.json',tail)
        require(len(rows) == COUNT+3 and {p.name for p in service.glob('input_*.json')} == {'input_701.json','input_2815.json'},'Full sequence or input SHA scope differs')
        all_stages = [inspect_stage(service,i,row,cache_plan,launch) for i,row in enumerate(rows[:-1])]
        require(sorted(p.name for p in service.glob('vision_cache_reset_*.json')) == [f'vision_cache_reset_{e["index"]+2:04d}.json' for e in entries if e['is_episode_start']], 'Unexpected cache resets')
        require(len(list(service.glob('vision_cache_request_*.json'))) == COUNT+2,'Missing or extra request cache receipts')
        require(numeric_gate(service,launch) == proof,'Numerical gates changed during formal sequence')
        stages = all_stages[2:]; H.write(directory/'stages.json',stages)
        contract = match_execution(service,launch,binding); measured = rows[2:-1]
        result = dict(status='complete',session=session,native_reference_binding=binding,execution_contract=contract,
            replay_manifest_sha256=H.digest(replay_dir/'manifest.json'),cache_plan_sha256=H.digest(service/'cache_request_plan.json'),
            measured_requests=COUNT,total_http_s=sum(r['http_total_s'] for r in measured),p50_http_s=statistics.median(r['http_total_s'] for r in measured),
            sequence_wall_s=measured[-1]['client_perf_end_s']-measured[0]['client_perf_start_s'],numeric_proof=proof,
            cache_mode=launch['vision_cache_contract']['mode'],cache_capacity=launch['vision_cache_contract']['capacity'],
            episode_actual_resets=6,artificial_idle_seconds=0,sequence_repetitions=1,outputs_fed_back=False,
            measured_request_tensor_sha_collected=False,production_ready=False,benefit_claim_allowed=False,
            scope='Independent cache experiment; needs same-weight off/on all343 exact token gate and quality validation before claims')
        H.write(directory/'result.json',result); return result
    except BaseException:
        if rows and not (directory/'requests.json').exists(): H.write(directory/'requests_partial.json',rows)
        raise
    finally:
        H.stop_owned(child); H.write(directory/'cleanup.json',dict(completed_utc=utc(),service_pid=child.pid if child else None))


def main():
    global H
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',required=True,type=Path); parser.add_argument('--session-config',required=True,type=Path)
    parser.add_argument('--native-reference-config',required=True,type=Path); parser.add_argument('--fixed-study',required=True,nargs='+',type=Path)
    parser.add_argument('--output',type=Path); parser.add_argument('--base-port',type=int,default=24610)
    parser.add_argument('--startup-timeout',type=int,default=1200); parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true',help='Only two excluded input/numeric warmups plus tail; never load or execute historical343')
    args = parser.parse_args(); H = load_helpers()
    require(os.name == 'posix' or args.check_only,'Execute with the project WSL Python')
    replay_dir = args.replay.resolve(); manifest = H.load_manifest(replay_dir); entries = manifest['requests']
    require(len(entries) == COUNT and list(dict.fromkeys(e['episode_id'] for e in entries)) == EPISODES,'Only original complete first6/343 replay is permitted')
    sessions = normalize(args.session_config)
    if args.check_only:
        bindings = reference_bindings(args.fixed_study,args.native_reference_config,sessions)
        print(json.dumps(dict(status='configuration_checked',sessions=len(sessions),bindings=len(bindings),
            measured_requests=0 if args.smoke_only else COUNT,smoke_only=args.smoke_only,gpu_started=False,payloads_loaded=False))); return
    require(args.output is not None and 1024 <= args.base_port < 65536-len(sessions),'Fresh output and valid port range required')
    run = args.output.resolve(); require(run.is_relative_to(ROOT/'logs') and not run.exists(),'Use a fresh project logs directory')
    import fcntl
    locks = []; completed = []
    try:
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock',ROOT/'tmp/navigation_repair_gpu.lock'):
            lock = path.open('a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(lock)
        bindings = reference_bindings(args.fixed_study,args.native_reference_config,sessions); run.mkdir(parents=True)
        names = {'checks/'+n for b in bindings for n in b['evidence']['source_sha256']}
        names.update({ENTRY,'checks/'+MAP,'checks/'+MODULE,'checks/run_navigation_latency_vision_cache_replay_v2_20261120.py',
            'checks/run_navigation_latency_replay_20261120.py','checks/prepare_navigation_latency_replay_20261120.py',
            'checks/run_navigation_latency_navigation_20261120.py','checks/run_navigation_latency_study_20261120.py',
            'checks/summarize_navigation_latency_20261120.py','checks/check_navigation_awq_dart_request.py','checks/navigation_repair_output_gate.py'})
        sources = {}
        for name in sorted(names):
            target = run/'source_snapshot'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,target); sources[name] = H.digest(target)
        H.write(run/'replay_manifest_binding.json',dict(path=str(replay_dir/'manifest.json'),sha256=H.digest(replay_dir/'manifest.json'),requests=COUNT,episodes=EPISODES))
        H.write(run/'protocol.json',dict(schema='navigation_vision_cpu_cache_smoke/v1' if args.smoke_only else 'navigation_vision_cpu_cache_replay/v1',
            created_utc=utc(),sessions=sessions,native_reference_bindings=bindings,smoke_only=args.smoke_only,
            no_latency_claim=args.smoke_only,no_quality_claim=True,complete_replay_evidence=False,
            replay_manifest_sha256=H.digest(replay_dir/'manifest.json'),source_sha256=sources,measured_requests=0 if args.smoke_only else COUNT,
            output_contract='18 tokens, greedy top_k1/top_p1/temperature0, seed7, min_new_tokens18, ignore_eosTrue',
            artificial_idle_seconds=0,sequence_repetitions=1,outputs_fed_back=False,navigation_success_claim=False,
            cache_reset='No historical episode executed in smoke' if args.smoke_only else 'Actual LRU clear before first measured request and every episode boundary',
            input_scope='Same immutable payloads/model-only substitution; tensor SHA restricted to two excluded warmups',
            numerical_scope='Four-feature bytewise gate on two excluded input warmups only; formal all343 exact-output pair audit still required'))
        for number,(session,binding) in enumerate(zip(sessions,bindings)):
            H.write(run/'status.json',dict(status='running',current_session=session,completed_sessions=len(completed),updated_utc=utc()),exclusive=False)
            completed.append(run_session(run,number,session,replay_dir,manifest,binding,args.base_port+number,args.startup_timeout,args.smoke_only))
            H.write(run/'completed.json',completed,exclusive=False)
            for name,sha in sources.items(): require(H.digest(ROOT/name) == sha,'Frozen cache experiment source changed: '+name)
        final_status = 'smoke_only' if args.smoke_only else 'complete'
        H.write(run/'result.json',dict(status=final_status,sessions=completed,benefit_claim_allowed=False,
            no_latency_claim=args.smoke_only,no_quality_claim=True,complete_replay_evidence=not args.smoke_only,
            scope='Two excluded warmups plus tail only; no historical payloads or measured requests' if args.smoke_only else
                  'Retained complete cache replay data, not navigation SR; independent paired audit and subset20 quality required'))
        H.write(run/'status.json',dict(status=final_status,completed_sessions=len(completed),updated_utc=utc()),exclusive=False)
    except BaseException as error:
        if run.exists():
            H.write(run/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),at_utc=utc()))
            H.write(run/'status.json',dict(status='failed',completed_sessions=len(completed),updated_utc=utc()),exclusive=False)
        raise
    finally:
        for lock in locks: fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    main()
