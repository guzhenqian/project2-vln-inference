"""Independent paired audit of the frozen 343-request uncached HTTP replay.

Run after GPU timing has stopped. This reads frozen encoded payloads, source
snapshots, responses and CUDA-event receipts; it never decodes PNGs, imports
torch or opens checkpoint weights. Fixed studies provide configuration/input
provenance only: their idle-separated latencies never enter replay statistics.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import summarize_navigation_latency_20261120 as fixed
from prepare_navigation_latency_replay_20261120 import load_manifest, load_request
from navigation_repair_output_gate import inspect_answer


ROOT = Path(__file__).resolve().parents[1]
EPISODES = ['1250','159','1413','558','1123','384']
COUNT = 343
_FIXED_CACHE = {}
_PAYLOAD_CACHE = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def object_sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def project_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    require(path.is_relative_to(ROOT), 'Evidence outside project: '+str(path))
    return path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\','/')


def close(a, b):
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)


def stats(values):
    values = list(values)
    require(values and all(isinstance(v,(int,float)) and math.isfinite(v) for v in values), 'Invalid numeric samples')
    ordered = sorted(values); pos = (len(ordered)-1)*.95; lo = math.floor(pos); hi = math.ceil(pos)
    return dict(n=len(values),p50=statistics.median(values),p95=ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo),
                mean=statistics.mean(values),sum=sum(values),min=min(values),max=max(values))


def configuration_key(session):
    return dict(group=session['group'],service_entry=str(project_path(session['service_entry'])),
        model_name=session.get('model_name','navigation-repair-'+session['group']),env=session.get('env',{}),
        extra_args=session.get('extra_args',[]),
        frozen_table_sha256=digest(project_path(session['frozen_table'])) if session.get('frozen_table') else None)


def protocol_audit(run):
    state = read(run/'status.json')
    require(state['status'] in ('complete','failed'), 'Replay is still running; audit only terminal runs')
    protocol = read(run/'protocol.json')
    for key,expected in dict(measured_requests=343,artificial_idle_seconds=0,sequence_repetitions=1,
        outputs_fed_back=False,navigation_success_claim=False,vision_cache=False,prefix_kv_cache=False).items():
        require(protocol[key] == expected, 'Replay protocol differs: '+key)
    require(protocol['output_contract'] == '18 tokens, greedy top_k1/top_p1/temperature0, seed7, min_new_tokens18, ignore_eosTrue',
            'Replay generation protocol differs')
    for name,sha in protocol['source_sha256'].items():
        snapshot = run/'source_snapshot'/name
        require(snapshot.is_file() and digest(snapshot) == sha, 'Replay source snapshot differs: '+name)
    # These imported helpers must be the exact code frozen by this controller.
    for name in ('checks/summarize_navigation_latency_20261120.py','checks/prepare_navigation_latency_replay_20261120.py',
                 'checks/navigation_repair_output_gate.py'):
        require(digest(ROOT/name) == protocol['source_sha256'][name], 'Imported audit helper changed: '+name)
    binding = read(run/'replay_manifest_binding.json'); manifest_path = project_path(binding['path'])
    require(digest(manifest_path) == binding['sha256'] == protocol['replay_manifest_sha256'], 'Replay manifest binding differs')
    manifest = load_manifest(manifest_path.parent)
    require(binding['status'] == 'complete' and binding['requests'] == COUNT and binding['episodes'] == EPISODES and
            binding['source_files_sha256'] == manifest['source_files_sha256'], 'Frozen replay manifest provenance differs')
    return state,protocol,manifest_path.parent,manifest


def expected_plan(entries):
    plan = [dict(phase='input_warmup',sample=i,tokens=0,repeat=0) for i in (0,1)]
    plan += [dict(phase='measured',sample=e['index'],tokens=18,repeat=0,replay_index=e['index'],
                  episode_id=e['episode_id'],source_step=e['step']) for e in entries]
    return plan+[dict(phase='flush',sample=0,tokens=18,repeat=0)]


def payload_proofs(directory, manifest, model):
    key = (digest(directory/'manifest.json'),model)
    if key not in _PAYLOAD_CACHE:
        values = []
        for entry in manifest['requests']:
            wire,proof = load_request(directory,entry,model)
            require(hashlib.sha256(wire).hexdigest() == proof['wire_bytes_sha256'], 'Reconstructed wire SHA differs')
            values.append(proof)
        _PAYLOAD_CACHE[key] = values
    return _PAYLOAD_CACHE[key]


def fixed_binding_audit(binding, session):
    run = project_path(binding['study']); sid = binding['fixed_session']
    require(read(run/'status.json')['status'] == 'complete', 'Bound fixed study did not complete')
    require(digest(project_path(sid)/'result.json') == binding['fixed_result_sha256'], 'Bound fixed result changed')
    protocol = read(run/'protocol.json')
    require(set(protocol['cells']) == {'0_18','0_64','1_18','1_64'} and protocol['measured_repeats'] >= 10 and
            protocol['http_mode'] == 'nonstream', 'Fixed binding lacks complete four-cell nonstream evidence')
    matches = [(i,s) for i,s in enumerate(protocol['sessions']) if relative(run/f"{i:02d}_{s['label']}") == sid]
    require(len(matches) == 1 and configuration_key(matches[0][1]) == configuration_key(session), 'Fixed/replay session configuration differs')
    if sid not in _FIXED_CACHE:
        _FIXED_CACHE[sid] = fixed.audit_session(run,protocol,*matches[0])
    audited = _FIXED_CACHE[sid]
    for key in ('engine','requested_engine','execution_identity','actual_input_tensors','model_weights_sha256','frozen_launch_configuration_sha256'):
        require(binding[key] == audited[key], 'Fixed binding disagrees with original evidence: '+key)
    require(binding['source_sha256'] == {n:v['sha256'] for n,v in audited['source_verification'].items()}, 'Fixed binding source identity differs')
    require(all(v['output_sequences_identical'] for v in audited['cells'].values()), 'Bound fixed outputs were unstable')
    return audited


def response_audit(row, model, count=None):
    response = row['response']; choice = response['choices'][0]; usage = row['usage']; tokens = row['output_tokens']
    require(response['model'] == model and len(response['choices']) == 1 and response['usage'] == usage and
            choice['message']['gen_tokens'] == tokens and len(tokens) == usage['completion_tokens'] and
            all(isinstance(v,int) and not isinstance(v,bool) and v >= 0 for v in tokens) and
            row['output_tokens_sha256'] == object_sha(tokens), 'Raw response/token evidence differs')
    require(usage['prompt_tokens'] > 0 and usage['completion_tokens'] > 0 and
            usage.get('total_tokens',usage['prompt_tokens']+len(tokens)) == usage['prompt_tokens']+len(tokens), 'Response token usage differs')
    require(row['http_mode'] == 'nonstream' and row['ttft_s'] is None and row['stream_decode_ms_per_token'] is None and
            row['idle_seconds_outside_timer'] == 0 and math.isfinite(row['http_total_s']) and row['http_total_s'] > 0,
            'Replay timing mode/idle differs')
    text = choice['message']['content'] or ''
    require(isinstance(text,str) and row['output_text_sha256'] == object_sha(text) and row['token_arrival_s'] == [],
            'Nonstream output text/arrival evidence differs')
    if count is not None:
        require(len(tokens) == count and choice['finish_reason'] == 'length', 'Forced output length/finish differs')


def timing_audit(service, index, row):
    path = service/f'timing_{index:03d}.json'; value = read(path); steps = value['steps']
    require(value['index'] == index and value['plan'] == row['spec'] and steps and
            steps[0]['phase'] == 'prefill' and steps[0]['rows'] == row['usage']['prompt_tokens'] <= 3072,
            'Request/prefill/timing mapping differs')
    decode = steps[1:]
    require(decode and all(s['phase'] == 'decode' and s['rows'] == 1 and s['execution_mode'] == 'replay' for s in decode) and
            len(steps) >= row['usage']['completion_tokens'], 'Decode is incomplete or contains eager/capture')
    for step in steps:
        for timer in step['timers'].values():
            require(all(isinstance(timer[k],(int,float)) and math.isfinite(timer[k]) and timer[k] >= 0 for k in ('cuda_interval_ms','host_wall_ms')),
                    'Invalid stage timing')
    first = steps[0]; totals = [s['timers']['model']['cuda_interval_ms'] for s in decode]
    through = [s['model_through_logits_ms'] for s in decode]
    require(all(math.isfinite(v) and v >= 0 for v in through), 'Invalid decode-through-logits timing')
    return dict(index=index,sample=row['spec']['sample'],tokens=row['spec']['tokens'],repeat=row['spec']['repeat'],
        http_total_s=row['http_total_s'],ttft_s=None,stream_decode_ms_per_token=None,
        prefill_language_ms=first['timers']['language']['cuda_interval_ms'],prefill_vision_ms=first['timers']['vision']['cuda_interval_ms'],
        prefill_model_ms=first['timers']['model']['cuda_interval_ms'],prefill_model_host_ms=first['timers']['model']['host_wall_ms'],
        decode_model_ms_per_actual_forward=statistics.mean(totals),decode_model_total_ms=sum(totals),
        decode_through_logits_ms_per_actual_forward=statistics.mean(through),actual_decode_forwards=len(decode),
        execution_modes=dict(Counter(s['execution_mode'] for s in decode)),output_tokens_sha256=row['output_tokens_sha256'],
        output_tokens=row['output_tokens'],semantic_payload_sha256=row['semantic_payload_sha256'])


def aggregate(rows):
    keys = ('http_total_s','prefill_language_ms','prefill_vision_ms','prefill_model_ms','prefill_model_host_ms',
            'decode_model_ms_per_actual_forward','decode_model_total_ms','decode_through_logits_ms_per_actual_forward','actual_decode_forwards')
    result = {key:stats(r[key] for r in rows) for key in keys}
    result['aggregate_decode_ms_per_actual_forward'] = sum(r['decode_model_total_ms'] for r in rows)/sum(r['actual_decode_forwards'] for r in rows)
    return result


def audit_session(run, protocol, replay_dir, manifest, number, session):
    env = session.get('env',{})
    require(not env.get('PROJECT2_LATENCY_W4_TABLE') and not env.get('PROJECT2_LATENCY_HYBRID_TABLE'),
            'This replay auditor supports unchanged native quantized kernels only; wide/hybrid need their separate full-model gates')
    require(env.get('PROJECT2_LATENCY_PROFILE') != '1' and env.get('PROJECT2_LATENCY_SCHEDULE_TRACE') != '1',
            'Diagnostic profiling/trace cannot enter matched replay timing')
    directory = run/f"{number:02d}_{session['label']}"; service = directory/'service'; model = session.get('model_name','navigation-repair-'+session['group'])
    controller = read(directory/'result.json'); binding = read(directory/'fixed_evidence_binding.json')
    require(controller['status'] == 'complete' and controller['session'] == session and
            controller['fixed_evidence_binding'] == binding == protocol['fixed_study_bindings'][number], 'Session/fixed binding differs')
    fixed_evidence = fixed_binding_audit(binding,session)
    command = read(directory/'command.json'); ownership = read(directory/'ownership.json'); cleanup = read(directory/'cleanup.json')
    require(command['configuration'] == configuration_key(session) and ownership['service_pid'] == ownership['service_process_group'] == cleanup['service_pid'] and
            isinstance(ownership['service_pid'],int) and ownership['service_pid'] > 0, 'Service ownership/configuration evidence differs')
    launch = read(service/'launch.json'); sources = {}
    require(launch['group'] == session['group'] and launch['engine'] == binding['requested_engine'] and
            launch['model_weights_sha256'] == binding['model_weights_sha256'] and launch['sources_sha256'] == binding['source_sha256'] and
            launch['model_config_sha256'] == fixed_evidence['model_config_sha256'] and
            launch['candidate_environment'] == env, 'Actual launch differs from bound fixed configuration')
    for name,sha in launch['sources_sha256'].items():
        require(protocol['source_sha256']['checks/'+name] == sha and digest(run/'source_snapshot'/name) == sha,
                'Actual runtime source/snapshot differs: '+name)
        sources[name] = dict(sha256=sha,source='retained_snapshot')
    contract = fixed.execution_contract(service,launch,sources)
    require(contract['effective_engine'] == binding['engine'] and contract['execution_identity'] == binding['execution_identity'] and
            contract == controller['execution_contract'], 'Actual execution contract differs from binding/controller')
    require(contract['effective_engine']['enable_prefix_caching'] is False, 'Replay prefix cache is enabled')
    execution = read(service/'execution_audit.json'); repair = read(service/'repair_execution_audit.json')
    require(execution['first_native_forward_verified'] is True and read(service/'startup_graph_capture.json')['actual_capture_observed'] is True and
            repair['native_forward'] is True and repair['extension'] is False and repair['activation_bits'] == 16 and
            repair['input_sha_scope'] == 'Explicit first two fixed smoke requests only', 'Native/precision/input-hash execution contract differs')
    method = session['group'].split('_')[0]
    if method != 'fp16':
        receipt = execution['native_awq_receipt']; table = repair['frozen_launch_table']
        require(receipt['verified'] and receipt['module_count'] == 144 and receipt['loaded_quantized_tensor_count'] == 756,
                'Native W4 loaded tensors/modules not verified')
        require(table['frozen'] and table['actual_m_preserved'] and table['generic_m_specialization_disabled'] and
                not table['runtime_autotuning_allowed'] and not table['runtime_cold_compilation_allowed'] and
                table['algorithm'] == method and table['model_weights_sha256'] == binding['model_weights_sha256'] and
                object_sha({k:table[k] for k in ('shapes','representatives','max_rows')}) == binding['frozen_launch_configuration_sha256'],
                'Actual native frozen launch configurations differ')
        require(table['selection_mode'] == 'replayed_frozen_table' and
                table['frozen_table_source']['sha256'] == command['configuration']['frozen_table_sha256'], 'Runtime did not replay the bound native table')
    entries = manifest['requests']; requests = read(directory/'requests.json'); plan = expected_plan(entries)
    require(len(requests) == COUNT+3 and read(service/'request_plan.json') == plan and
            [r['index'] for r in requests] == list(range(COUNT+3)) and [r['spec'] for r in requests] == plan,
            'Full 343 replay plus two warmups/tail sequence missing/reordered/duplicated')
    require(read(directory/'input_warmups.json') == requests[:2] and read(directory/'tail_request.json') == requests[-1], 'Warmup/tail records differ')
    logged = [json.loads(line) for line in (directory/'replay_responses.jsonl').read_text(encoding='utf-8').splitlines()]
    require(logged == requests[2:-1], 'Buffered raw response ledger differs from requests')
    require({p.name for p in service.glob('input_*.json')} == {'input_701.json','input_2815.json'}, 'Unexpected measured-request input tensor hashing')
    actual_inputs = {}
    for index,length in enumerate((701,2815)):
        value = read(service/f'input_{length}.json'); actual_inputs[str(length)] = {k:value[k] for k in ('input_ids','pixel_values')}
        require(value['request_index'] == index and actual_inputs[str(length)] == fixed_evidence['actual_input_tensors'][str(length)],
                'Excluded warmup actual tensor SHA differs from fixed study')
        saved_input = controller['input_tensor_receipts'][f'input_{length}.json']
        require(saved_input['sha256'] == digest(service/f'input_{length}.json') and saved_input['first_request_index'] == index and
                all(saved_input[k] == value[k] for k in ('input_ids','pixel_values')), 'Controller warmup tensor receipt differs')
        response_audit(requests[index],model)
        gate = inspect_answer(requests[index]['response'])
        require(gate['passed'] and gate == read(directory/f'natural_output_gate_{index}.json'), 'Natural output gate differs')
        timing_audit(service,index,requests[index])
    response_audit(requests[-1],model,count=18)
    proofs = payload_proofs(replay_dir,manifest,model); stages = read(directory/'stages.json')
    require(len(stages) == COUNT, 'Stage receipt count differs')
    seen_lengths = {701,2815}; previous_end = None; measured = []; evidence_sha = {}
    for entry,row,saved,proof in zip(entries,requests[2:-1],stages,proofs):
        index = entry['index']+2; response_audit(row,model,count=18)
        require(row['frozen_replay_index'] == entry['index'] and row['episode_id'] == entry['episode_id'] and
                row['source_step'] == entry['step'] and row['episode_start'] == entry['is_episode_start'] and
                row['episode_end'] == entry['is_episode_end'] and row['source_image_sha256'] == entry['request_image_sha256'] and
                row['usage']['prompt_tokens'] == entry['source_prompt_tokens'], 'Replay row historical input/order differs')
        require(row['input_proof'] == proof and row['request_sha256'] == proof['wire_bytes_sha256'] and
                row['semantic_payload_sha256'] == entry['semantic_payload_sha256'], 'Actual wire/model-normalized payload identity differs')
        raw = row['response_body_utf8'].encode('utf-8')
        require(hashlib.sha256(raw).hexdigest() == row['response_bytes_sha256'] and json.loads(raw) == row['response'], 'Raw response bytes differ')
        start,end = row['client_perf_start_s'],row['client_perf_end_s']
        require(start > 0 and end > start and close(end-start,row['http_total_s']), 'Client HTTP clock interval differs')
        gap = None if previous_end is None else start-previous_end
        require((gap is None and row['client_gap_since_previous_replay_s'] is None) or
                (gap is not None and gap >= 0 and close(gap,row['client_gap_since_previous_replay_s'])), 'Client serial gap differs')
        previous_end = end; length = row['usage']['prompt_tokens']; first = length not in seen_lengths; seen_lengths.add(length)
        require(row['first_seen_prompt_length_after_explicit_warmups'] is first, 'Input-length novelty label differs')
        stage = timing_audit(service,index,row)
        for key,value in stage.items():
            require(close(value,saved.get(key)) if isinstance(value,float) else value == saved.get(key), 'Stored stage differs from raw timing: '+key)
        require(saved['replay_index'] == entry['index'] and saved['episode_id'] == entry['episode_id'] and saved['source_step'] == entry['step'] and
                saved['first_seen_prompt_length_after_explicit_warmups'] is first and
                saved['measured_request_tensor_sha_collected'] is False and saved['normalized_payload_bytes_sha256'] == entry['normalized_payload_bytes_sha256'] and
                saved['source_image_sha256'] == entry['request_image_sha256'], 'Stage input provenance differs')
        stage.update(replay_index=entry['index'],episode_id=entry['episode_id'],source_step=entry['step'],prompt_tokens=length,
            output_sha256=row['output_tokens_sha256'],normalized_payload_bytes_sha256=entry['normalized_payload_bytes_sha256'],
            source_image_sha256=entry['request_image_sha256'],client_gap_s=gap,
            first_seen_prompt_length=first,source_row_sha256=entry['source_row_sha256'])
        measured.append(stage); evidence_sha[relative(service/f'timing_{index:03d}.json')] = digest(service/f'timing_{index:03d}.json')
    if contract['vision_qmax_evidence']:
        for grid in contract['vision_qmax_evidence']['first_grid_coverage']:
            index = grid['request_index']; require(0 <= index < COUNT+2, 'QMAX first grid outside flushed requests')
            timing = read(service/f'timing_{index:03d}.json')
            require(timing['index'] == index and timing['plan'] == plan[index] and timing['steps'][0]['rows'] == grid['input_tokens'],
                    'QMAX first-grid actual prefill binding differs')
    summary = aggregate(measured); by_episode = {eid:aggregate([r for r in measured if r['episode_id'] == eid]) for eid in EPISODES}
    def check_http(saved,rows,computed):
        expected = dict(requests=len(rows),http_total_s=computed['sum'],http_p50_s=computed['p50'],http_p95_s=computed['p95'],
            min_s=computed['min'],max_s=computed['max'],prompt_tokens=sum(r['prompt_tokens'] for r in rows),completion_tokens=18*len(rows))
        require(all(close(v,saved[k]) if isinstance(v,float) else v == saved[k] for k,v in expected.items()), 'Controller HTTP summary differs')
    check_http(controller['complete_summary'],measured,summary['http_total_s'])
    for eid in EPISODES:
        check_http(controller['by_episode'][eid],[r for r in measured if r['episode_id'] == eid],by_episode[eid]['http_total_s'])
    for first in (False,True):
        chosen = [r for r in measured if r['first_seen_prompt_length'] is first]
        if chosen: check_http(controller['first_prompt_length_groups'][str(first)],chosen,stats(r['http_total_s'] for r in chosen))
    gaps = [r['client_gap_s'] for r in measured if r['client_gap_s'] is not None]
    sequence_wall = requests[-2]['client_perf_end_s']-requests[2]['client_perf_start_s']
    require(close(controller['sequence_wall_s'],sequence_wall) and close(sequence_wall,summary['http_total_s']['sum']+sum(gaps)),
            'Sequence wall time differs from HTTP intervals plus client gaps')
    require(controller['actual_decode_forwards'] == summary['actual_decode_forwards']['sum'] and
            close(controller['total_decode_model_ms'],summary['decode_model_total_ms']['sum']) and
            controller['replay_manifest_sha256'] == protocol['replay_manifest_sha256'] and controller['measured_requests'] == COUNT and
            controller['artificial_idle_seconds'] == 0 and controller['sequence_repetitions'] == 1 and controller['outputs_fed_back'] is False and
            controller['measured_http_includes_new_input_tensor_hashing'] is False, 'Controller counts/replay scope differ')
    for name in ('result.json','requests.json','stages.json','replay_responses.jsonl','fixed_evidence_binding.json','command.json','ownership.json','cleanup.json'):
        evidence_sha[relative(directory/name)] = digest(directory/name)
    for name in ('launch.json','execution_audit.json','repair_execution_audit.json','startup_graph_capture.json','request_plan.json','input_701.json','input_2815.json','timing_000.json','timing_001.json'):
        evidence_sha[relative(service/name)] = digest(service/name)
    return dict(status='audited',id=relative(directory),study=relative(run),label=session['label'],group=session['group'],
        engine=contract['effective_engine'],**contract,versions=launch['versions'],candidate_environment=env,source_verification=sources,
        actual_input_tensors=actual_inputs,model_weights_sha256=launch['model_weights_sha256'],
        model_config_sha256=launch['model_config_sha256'],
        frozen_launch_configuration_sha256=binding['frozen_launch_configuration_sha256'],
        manifest_sha256=protocol['replay_manifest_sha256'],http_mode='nonstream',idle_seconds=0,warmups=2,profile_enabled=False,
        measurements=measured,overall=summary,by_episode=by_episode,client_gaps_s=stats(gaps),sequence_wall_s=sequence_wall,
        source_contract='All 343 frozen payload/JPEG/text identities plus actual prompt token counts; tensor SHA only on two excluded smokes',
        evidence_sha256=evidence_sha,fixed_latency_used=False,navigation_quality_evaluated=False)


def system_family(session):
    engine = session['engine']; identity = session['execution_identity']
    if (engine['distributed_executor_backend'] == 'mp' and engine['prefill_interval'] == 16 and
        engine['enable_mp_engine'] is True and not identity['inprocess_requested'] and not identity['linear_ipc_requested']):
        return 'original_mp16'
    if (engine['distributed_executor_backend'] == 'uni' and engine['prefill_interval'] == 2 and
        engine['enable_mp_engine'] is False and identity['inprocess_requested'] and not identity['linear_ipc_requested']):
        return 'optimized_inprocess2'
    return 'other_execution_contract'


def paired_summary(rows):
    before = [r['old_http_s'] for r in rows]; after = [r['new_http_s'] for r in rows]
    return dict(n=len(rows),old_http_s=stats(before),new_http_s=stats(after),
        paired_http_saved_s=stats(r['http_saved_s'] for r in rows),
        paired_decode_forwards_saved=stats(r['decode_forwards_saved'] for r in rows),
        paired_prefill_model_ms_saved=stats(r['prefill_model_ms_saved'] for r in rows),
        paired_decode_model_ms_saved=stats(r['decode_model_ms_saved'] for r in rows),
        faster_requests=sum(r['http_saved_s'] > 0 for r in rows),slower_requests=sum(r['http_saved_s'] < 0 for r in rows),
        equal_http_requests=sum(r['http_saved_s'] == 0 for r in rows),
        observed_total_http_ratio=sum(after)/sum(before),observed_p50_ratio=statistics.median(after)/statistics.median(before),
        interpretation='Each delta is old minus new for the same frozen request index; positive means lower candidate latency. Percentiles are descriptive over heterogeneous requests, not repeated estimates of one request.')


def comparison_rows(sessions):
    comparisons = []
    for i,a in enumerate(sessions):
        for b in sessions[i+1:]:
            if a['study'] != b['study']:
                continue  # Repeated rounds remain separate; no cross-round percentile pooling.
            same_weight = a['model_weights_sha256'] == b['model_weights_sha256']
            if a['group'] == b['group']:
                families = {system_family(a),system_family(b)}
                if families != {'original_mp16','optimized_inprocess2'}:
                    continue
                before,after = (a,b) if system_family(a) == 'original_mp16' else (b,a)
                kind = 'same_weight_system_change'
                reasons = fixed.comparable(before,after,ignore_executor=True,ignore_inprocess=True,ignore_interval=True)
                if not same_weight: reasons.append('same group checkpoint weights differ')
                if before['frozen_launch_configuration_sha256'] != after['frozen_launch_configuration_sha256']:
                    reasons.append('native frozen kernel configurations differ')
            else:
                rank = {'fp16':0,'awq_w4a16':1,'gptq_w4a16':2}
                before,after = (a,b) if rank[a['group']] < rank[b['group']] else (b,a)
                kind = 'matched_quantization' if before['group'] == 'fp16' else 'matched_quantization_methods'
                reasons = fixed.comparable(before,after)
            if before['manifest_sha256'] != after['manifest_sha256']:
                reasons.append('frozen manifest differs')
            old,new = before['measurements'],after['measurements']
            require(len(old) == len(new) == COUNT, 'Only complete 343-request pairs may be compared')
            pairs, mismatches = [], []
            for x,y in zip(old,new):
                input_keys = ('replay_index','episode_id','source_step','prompt_tokens','normalized_payload_bytes_sha256',
                              'semantic_payload_sha256','source_image_sha256','source_row_sha256')
                if any(x[k] != y[k] for k in input_keys):
                    reasons.append('input/order differs at replay index '+str(x['replay_index']))
                if same_weight and x['output_tokens'] != y['output_tokens']:
                    positions = [j for j,(u,v) in enumerate(zip(x['output_tokens'],y['output_tokens'])) if u != v]
                    mismatches.append(dict(replay_index=x['replay_index'],episode_id=x['episode_id'],source_step=x['source_step'],
                        first_different_token_index=positions[0] if positions else min(len(x['output_tokens']),len(y['output_tokens'])),
                        different_token_positions=positions,old_token_ids=x['output_tokens'],new_token_ids=y['output_tokens']))
                pairs.append(dict(replay_index=x['replay_index'],episode_id=x['episode_id'],source_step=x['source_step'],
                    prompt_tokens=x['prompt_tokens'],normalized_payload_bytes_sha256=x['normalized_payload_bytes_sha256'],
                    old_http_s=x['http_total_s'],new_http_s=y['http_total_s'],http_saved_s=x['http_total_s']-y['http_total_s'],
                    old_actual_decode_forwards=x['actual_decode_forwards'],new_actual_decode_forwards=y['actual_decode_forwards'],
                    decode_forwards_saved=x['actual_decode_forwards']-y['actual_decode_forwards'],
                    prefill_model_ms_saved=x['prefill_model_ms']-y['prefill_model_ms'],
                    decode_model_ms_saved=x['decode_model_total_ms']-y['decode_model_total_ms'],
                    old_client_gap_s=x['client_gap_s'],new_client_gap_s=y['client_gap_s'],
                    output_tokens_equal=x['output_tokens'] == y['output_tokens'] if same_weight else None))
            if same_weight and mismatches:
                reasons.append('same-weight exact output gate failed on '+str(len(mismatches))+' of 343 requests')
            summary = paired_summary(pairs)
            comparisons.append(dict(kind=kind,study=before['study'],baseline=before['id'],candidate=after['id'],
                baseline_group=before['group'],candidate_group=after['group'],comparable=not reasons,reasons=sorted(set(reasons)),
                benefit_claim_allowed=not reasons,same_weights=same_weight,token_equality_required=same_weight,
                all_output_tokens_equal=not mismatches if same_weight else None,token_mismatches=mismatches,
                http_total_reduction_percent=100*(1-summary['observed_total_http_ratio']) if not reasons else None,
                http_p50_reduction_percent=100*(1-summary['observed_p50_ratio']) if not reasons else None,
                observed_paired_summary=summary,by_episode={eid:paired_summary([p for p in pairs if p['episode_id'] == eid]) for eid in EPISODES},
                pairs=pairs,quality_scope='Different checkpoint outputs are not required to match and are not scored as quality here. No feedback or navigation success rate was measured.',
                statistical_scope='One complete sequence per configuration, separately run; paired observations are descriptive, without significance or independence claims.'))
    return comparisons


def propagate_system_output_gates(sessions, comparisons):
    blocked = {}
    for session in sessions:
        if system_family(session) != 'optimized_inprocess2':
            session['same_weight_old_path_gate'] = dict(status='not_applicable_original_or_other_path')
            continue
        matching = [r for r in comparisons if r['kind'] == 'same_weight_system_change' and r['candidate'] == session['id']]
        passed = matching and all(r['comparable'] and r['all_output_tokens_equal'] is True for r in matching)
        gate = dict(status='verified' if passed else 'failed_or_missing_same_weight_original_pair',
                    reference_ids=[r['baseline'] for r in matching],
                    mismatching_requests=sum(len(r['token_mismatches']) for r in matching),
                    scope='All 343 exact 18-token outputs must match the same-weight original MP16 sequence before claiming optimized-path benefits.')
        session['same_weight_old_path_gate'] = gate
        if not passed: blocked[session['id']] = gate
    for row in comparisons:
        failures = {sid:blocked[sid] for sid in (row['baseline'],row['candidate']) if sid in blocked}
        if failures:
            row['comparable'] = row['benefit_claim_allowed'] = False
            row['system_output_gate_failures'] = failures
            row['http_total_reduction_percent'] = row['http_p50_reduction_percent'] = None
            reason = 'Optimized path lacks a complete successful same-weight original-output gate; retained paired observations cannot support benefits'
            if reason not in row['reasons']: row['reasons'].append(reason)


def audit_runs(paths):
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),studies=[],sessions=[],failures=[],unknowns=[],
        limitations=[
            'All measurements are one complete 343-request frozen historical sequence per configuration, without feedback; they do not measure navigation SR or quality.',
            'Only this same-manifest no-idle replay is paired. Fixed-input study latencies with one-second idle gaps are never pooled or substituted.',
            'Same-weight MP16/in-process2 changes require exact token equality for every request. Cross-precision token differences are not treated as quality failures or successes.',
            'Per-request HTTP intervals include serial service effects. CUDA-event stage intervals include dispatch gaps and post-response decode work; nested stages must not be added as exclusive wall time.',
            'Only the two excluded smoke requests have actual tensor SHA receipts. Every historical request is bound to frozen payload/JPEG/text hashes and observed prompt tokens, not a newly collected tensor SHA.',
            'Client gaps are reported separately from HTTP intervals and retained in sequence wall time. Payload decoding, audit checks, and stage-file reads occur outside the measured request loop.',
            'P50/P95 over different historical requests describe this sequence; one pass supplies no repeated-run uncertainty, significance, or independence claim.',
            'The original full-model quantization quality evidence remains separate. Forced 18-token replay output is not fed back or parsed into a navigation success metric.'])
    for value in paths:
        run = project_path(value); require(run.is_relative_to(ROOT/'logs'), 'Replay must be under project logs')
        try:
            state,protocol,replay_dir,manifest = protocol_audit(run)
        except Exception as error:
            output['failures'].append(dict(study=relative(run),audit_error=repr(error))); continue
        output['studies'].append(dict(directory=relative(run),status=state['status'],protocol_sha256=digest(run/'protocol.json'),
            manifest_sha256=protocol['replay_manifest_sha256'],request_count=COUNT,source_episode_order=EPISODES))
        if state['status'] == 'failed':
            output['failures'].append(dict(study=relative(run),failure=read(run/'failure.json') if (run/'failure.json').exists() else 'Missing failure details'))
        for number,session in enumerate(protocol['sessions']):
            directory = run/f"{number:02d}_{session['label']}"
            if not (directory/'result.json').is_file():
                output['unknowns'].append(dict(session=relative(directory),reason='No complete session result; partial sequence excluded from comparisons'))
                continue
            try:
                output['sessions'].append(audit_session(run,protocol,replay_dir,manifest,number,session))
            except Exception as error:
                output['failures'].append(dict(session=relative(directory),audit_error=repr(error)))
    output['comparisons'] = comparison_rows(output['sessions'])
    propagate_system_output_gates(output['sessions'],output['comparisons'])
    for session in output['sessions']:
        if session['same_weight_old_path_gate']['status'].startswith('failed'):
            output['unknowns'].append(dict(session=session['id'],reason='Optimized-path benefits blocked by missing/failed same-weight complete output gate',
                                           gate=session['same_weight_old_path_gate']))
    output['status'] = 'audited_with_failures_or_unverified_comparisons' if output['failures'] or output['unknowns'] else 'audited'
    output['auditor_sha256'] = digest(__file__)
    output['audit_dependencies_sha256'] = {name:digest(ROOT/'checks'/name) for name in (
        'summarize_navigation_latency_20261120.py','prepare_navigation_latency_replay_20261120.py','navigation_repair_output_gate.py')}
    return output


def markdown(result):
    lines = ['# 343条真实历史输入：独立配对时延审计','',
        '状态：`'+result['status']+'`。每配置完整单次回放，未反馈输出，不是导航成功率评测。','',
        '| 配置 | HTTP总计/s | Mean/s | P50/s | P95/s | 实际decode次数 | Client gap总计/s |','|---|---:|---:|---:|---:|---:|---:|']
    for s in result['sessions']:
        h = s['overall']['http_total_s']
        lines.append(f"| {s['label']} | {h['sum']:.3f} | {h['mean']:.4f} | {h['p50']:.4f} | {h['p95']:.4f} | {s['overall']['actual_decode_forwards']['sum']} | {s['client_gaps_s']['sum']:.3f} |")
    lines += ['','## 配对比较','',
        '| 类型/配置 | 可归因 | 总HTTP降低 | 同输入配对平均减少/ms | 配对P50减少/ms | 变快/变慢请求 | 同权重输出 |',
        '|---|---|---:|---:|---:|---:|---|']
    for c in result['comparisons']:
        p = c['observed_paired_summary']; allowed = c['benefit_claim_allowed']
        reduction = f"{c['http_total_reduction_percent']:.2f}%" if allowed else '禁用收益声明'
        mean = f"{1000*p['paired_http_saved_s']['mean']:.2f}" if allowed else '保留观测'
        median = f"{1000*p['paired_http_saved_s']['p50']:.2f}" if allowed else '保留观测'
        equality = str(c['all_output_tokens_equal']) if c['token_equality_required'] else '跨权重不作同等要求'
        lines.append(f"| {c['kind']}: {c['baseline'].split('/')[-1]} → {c['candidate'].split('/')[-1]} | {'是' if allowed else '否'} | {reduction} | {mean} | {median} | {p['faster_requests']}/{p['slower_requests']} | {equality} |")
    lines += ['','## 每个episode的HTTP统计','',
        '| 配置 | Episode | 请求 | Sum/s | Mean/s | P50/s | P95/s |','|---|---|---:|---:|---:|---:|---:|']
    for s in result['sessions']:
        for eid,summary in s['by_episode'].items():
            h = summary['http_total_s']; lines.append(f"| {s['label']} | {eid} | {h['n']} | {h['sum']:.3f} | {h['mean']:.4f} | {h['p50']:.4f} | {h['p95']:.4f} |")
    lines += ['','逐请求配对差、各episode配对差、prefill/vision/decode阶段和client gap完整数据保存在JSON。正配对差表示候选更快。','',
              '## 限制与未通过项','']
    lines += ['- '+x for x in result['limitations']]
    for c in result['comparisons']:
        if c['reasons']:
            lines.append('- '+c['baseline'].split('/')[-1]+' → '+c['candidate'].split('/')[-1]+': '+'；'.join(c['reasons']))
    lines += ['- '+canonical(v) for v in result['failures']+result['unknowns']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    output = project_path(args.output)
    require(output.suffix == '.json' and not output.exists() and not output.with_suffix('.md').exists(), 'Choose a fresh project JSON output path')
    result = audit_runs(args.runs); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    output.with_suffix('.md').write_text(markdown(result),encoding='utf-8')
    print(canonical(dict(status=result['status'],sessions=len(result['sessions']),comparisons=len(result['comparisons']),
                         failures=len(result['failures']),unknowns=len(result['unknowns']),output=str(output))))


if __name__ == '__main__':
    main()
