"""CPU-only1+8 inventory of real repair100 reports and the latest20 audit.

Example (omit --reports until real collector reports exist)::
  python -B checks/summarize_navigation_repair100.py \
    --screen-audit reports/repair20_latest.json \
    --reports reports/repair_fp16_100.json reports/repair_awq_w4a16_100.json \
    --output reports/repair100_comparison.json

Always emits FP16 plus eight requested quantization rows. Missing80 never
becomes100. Complete100 describes completed evidence, not a new quality pass
threshold. The20 acceptance thresholds are not extrapolated to100.

Every supplied100 report must come from collect_navigation_repair100.py and
retain its new20+80 raw sources. Non-model evidence hashes, raw case metrics,
real-request percentiles/tokens and actual forward counts are rechecked. Model
assets are never opened; their prior collector-certified SHA values are cited.
The only cross-case input comparison is the FIRST real navigation request.
Later trajectories legitimately diverge. Missing FP16100 or mismatching first
inputs prevents comparison conclusions. This script does not run experiments.

Development status: CPU in-memory boundary checks only until actual80/100
artifacts exist. A successful invocation on real reports performs the checks
below; implementation availability alone is not an experiment result.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT/'data/baseline/navigation_repair_20261112'
REPAIR_LOGS = ROOT/'logs/navigation_repair_20261112'
GROUPS = ['fp16'] + [algorithm+'_'+precision for algorithm in ('awq','gptq')
                    for precision in ('w4a16','w4a4','w4a8','w8a16')]
QUALITY_ERRORS = {'truncated_model_output', 'invalid_action_random_fallback_rejected'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    require(isinstance(actual, (int,float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8),
            label+': differs from raw evidence')


def p95(values):
    require(bool(values), 'Empty latency distribution')
    ordered = sorted(values)
    index = (len(ordered)-1)*.95
    lower = int(index)
    return ordered[lower]+(ordered[min(lower+1,len(ordered)-1)]-ordered[lower])*(index-lower)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Evidence:
    """Read small immutable evidence, never model assets or the private server file."""
    def __init__(self):
        self.verified = {}
        self.referenced_model_assets = {}
        self._stats = {}

    def path(self, source):
        path = Path(source)
        path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
        require(path.is_relative_to(ROOT), 'Evidence escapes project: '+str(source))
        require(path.name.lower() != 'server.md', 'Private server file is not permitted evidence')
        return path

    def read_bytes(self, source):
        path = self.path(source)
        require(not path.is_relative_to(ROOT/'models'), 'Aggregator must never open model assets')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        key = relative(path)
        require(key not in self.verified or self.verified[key] == digest, 'Evidence changed during aggregation: '+key)
        self.verified[key] = digest
        stat = path.stat()
        self._stats[key] = (stat.st_size,stat.st_mtime_ns)
        return data

    def load(self, source):
        return json.loads(self.read_bytes(source))

    def verify(self, source, expected):
        require(isinstance(expected,str) and len(expected)==64
                and all(c in '0123456789abcdef' for c in expected), 'Malformed SHA256')
        path = self.path(source)
        key = relative(path)
        if path.is_relative_to(ROOT/'models'):
            require(key not in self.referenced_model_assets or self.referenced_model_assets[key] == expected,
                    'Conflicting model SHA references: '+key)
            self.referenced_model_assets[key] = expected
            return
        if key not in self.verified:
            self.read_bytes(path)
        require(self.verified[key] == expected, 'Evidence SHA mismatch: '+key)

    def finish(self):
        for source, stamp in self._stats.items():
            stat = (ROOT/source).stat()
            require((stat.st_size,stat.st_mtime_ns) == stamp, 'Evidence changed while summarizing: '+source)


def first_input(row):
    require(row.get('model_called') is True and row.get('step') == 0, 'Case first row must be a real request')
    texts, images = row.get('request_text'), row.get('request_image_sha256')
    count = row.get('image_count')
    require(isinstance(texts,list) and texts and all(isinstance(t,str) for t in texts), 'Missing first-input text')
    require(type(count) is int and 2 <= count <= 9 and isinstance(images,list) and len(images)==count,
            'Missing first-input image identities')
    require(all(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in images),
            'Malformed first-input image SHA')
    return dict(request_text=texts, request_image_sha256=images, image_count=count,
                prompt_tokens=row['usage']['prompt_tokens'])


def request_timing(evidence, service, index, row, method):
    data = evidence.load(service/f'timing_{index:03d}.json')
    steps = data['steps']
    require(data['index']==index and data['group']==method and data['graph_requested']=='on', 'Timing identity mismatch')
    require(len(steps)>1 and len(steps)>=row['usage']['completion_tokens'], 'Actual forward evidence missing')
    prefill, decode = steps[0],steps[1:]
    require(prefill['phase']=='prefill' and prefill['rows']==row['usage']['prompt_tokens']
            and prefill['execution_mode']=='eager', 'Prefill/request mismatch')
    require(all(s['phase']=='decode' and s['rows']==1 and s['execution_mode']=='replay' for s in decode),
            'Decode includes unverified execution')
    pre = prefill['timers']['language']['cuda_interval_ms']
    dec = [s['timers']['model']['cuda_interval_ms'] for s in decode]
    require(all(math.isfinite(x) and x>=0 for x in [pre]+dec), 'Invalid stage interval')
    return dict(prefill_language_ms=pre, decode_forwards=len(decode), decode_total_ms=sum(dec))


def verify_case(evidence, folder, episode, service, timing_index, method):
    eid = str(episode['episode_id'])
    path = folder/('episode_'+eid)/'steps.jsonl'
    rows = [json.loads(line) for line in evidence.read_bytes(path).decode().splitlines()]
    require(0<len(rows)==episode['steps']<=400, 'Invalid raw case length')
    require(evidence.load(folder/('episode_'+eid)/'result.json')==episode, 'Episode/summary mismatch')
    signature = first_input(rows[0])
    requests, stages = [],[]
    path_length = 0.0
    for number,row in enumerate(rows):
        require(row['step']==number and str(row['episode_id'])==eid, 'Raw step identity/order differs')
        require(type(row.get('model_called')) is bool, 'Missing real-request flag')
        error = row.get('error')
        require(error is None or error in QUALITY_ERRORS, 'Infrastructure/unclassified failure prevents complete100 audit')
        if error:
            require(number==len(rows)-1 and row['action']=='STOP' and row['model_called'], 'Quality failure not retained as terminal failed STOP')
        before,after = row['before']['position'],row['after']['position']
        require(len(before)==len(after)==3 and all(math.isfinite(x) for x in before+after), 'Invalid positions')
        if number:
            require(all(math.isclose(a,b,abs_tol=1e-7) for a,b in zip(before,rows[number-1]['after']['position'])), 'Discontinuous trajectory')
        path_length += math.sqrt(sum((a-b)**2 for a,b in zip(before,after)))
        if row['model_called']:
            usage = row['usage']
            require(type(usage['prompt_tokens']) is int and 0<usage['prompt_tokens']<=3072
                    and type(usage['completion_tokens']) is int and 0<usage['completion_tokens']<=512,
                    'Invalid actual token counts')
            require(math.isfinite(row['model_request_s']) and row['model_request_s']>0, 'Invalid HTTP duration')
            require(row['finish_reason'] in ('stop','length'), 'Unexpected output termination')
            require((row['finish_reason']=='length')==(error=='truncated_model_output'), 'Truncation classification differs')
            requests.append(row)
            stages.append(request_timing(evidence,service,timing_index,row,method))
            timing_index += 1
        else:
            require(not error and row['model_request_s']==0 and row['usage'] is None, 'Queued action has request metrics')
    last = rows[-1]
    source = ('error_fallback' if last.get('error') else 'model') if last['action']=='STOP' else 'step_limit'
    require(episode['terminal_stop_source']==source and (last['action']=='STOP' or len(rows)==400), 'Terminal-source mismatch')
    require(last['metrics']==episode['metrics'], 'Last-step metrics differ')
    equal(episode['path_length_m'],path_length,'Path length')
    distance = episode['start_distance_m']
    require(math.isfinite(distance) and distance>0, 'Invalid initial distance')
    success = int(last['action']=='STOP' and source=='model' and not last.get('error') and last['metrics']['distance_to_goal']<3.)
    calculated = dict(episode_id=eid,trajectory_id=str(episode['trajectory_id']),success=success,
        spl=success*distance/max(distance,path_length),steps=len(rows),actual_model_calls=len(requests),
        errors=sum(bool(r.get('error')) for r in rows),
        collisions=sum(bool(r['metrics']['collisions']['is_collision']) for r in rows),terminal_stop_source=source)
    return calculated,requests,stages,signature,timing_index


def recompute(episodes, requests, stages, memory):
    require(len(episodes)==100 and requests and memory, 'Incomplete100 distributions')
    latencies = [r['model_request_s'] for r in requests]
    pre = [s['prefill_language_ms'] for s in stages]
    dec_means = [s['decode_total_ms']/s['decode_forwards'] for s in stages]
    forwards = sum(s['decode_forwards'] for s in stages)
    errors = Counter(r['error'] for r in requests if r.get('error'))
    return dict(case_count=100,success_count=sum(e['success'] for e in episodes),
        sr=sum(e['success'] for e in episodes)/100,spl=sum(e['spl'] for e in episodes)/100,
        actual_model_calls=len(requests),prompt_tokens=sum(r['usage']['prompt_tokens'] for r in requests),
        completion_tokens=sum(r['usage']['completion_tokens'] for r in requests),
        request_median_s=statistics.median(latencies),request_p95_s=p95(latencies),request_total_s=sum(latencies),
        sampled_whole_card_peak_mib=max(memory),error_count=sum(errors.values()),infrastructure_errors=0,
        quality_errors=dict(errors),collisions=sum(e['collisions'] for e in episodes),
        actual_prefill_forwards=len(stages),actual_decode_forwards=forwards,actual_total_forwards=len(stages)+forwards,
        stage_summary=dict(requests=len(stages),prefill_language_p50_ms=statistics.median(pre),
            prefill_language_p95_ms=p95(pre),actual_decode_forwards=forwards,
            decode_ms_per_actual_forward=sum(s['decode_total_ms'] for s in stages)/forwards,
            decode_request_mean_p50_ms=statistics.median(dec_means),decode_request_mean_p95_ms=p95(dec_means)))


def numeric_map(saved, computed, label):
    for key,value in computed.items():
        if isinstance(value,dict):
            require(isinstance(saved[key],dict), 'Missing numeric dictionary: '+label+'.'+key)
            if key=='quality_errors':
                require(saved[key]==value, 'Quality classification totals differ')
            else:
                numeric_map(saved[key],value,label+'.'+key)
        else:
            equal(saved[key],value,label+'.'+key)


def verify_100(evidence, path, original, split_manifests):
    report = evidence.load(path)
    group = report['group']
    require(group in GROUPS and report['status']=='complete' and report['case_count']==100, 'Only completed collector100 reports are inputs')
    require(report['quality_failures_retained_in_denominator'] is True and report['infrastructure_errors']==0, 'Unsupported100 error policy')
    collector = 'checks/collect_navigation_repair100.py'
    require(collector in report['evidence_files_sha256'], 'Input is missing collector source provenance')
    for source,digest in report['evidence_files_sha256'].items():
        evidence.verify(source,digest)
    require(len(report['batches'])==2, 'Expected new20 plus new80, not historical reused cohorts')
    acceptance = report['screen_acceptance']
    evidence.verify(acceptance['path'],acceptance['sha256'])
    accepted = evidence.load(acceptance['path'])
    verdict = next((r for r in accepted['table'] if r['group']==group),None)
    require(verdict==acceptance['verdict'] and verdict['passed'] is True and verdict['status']=='passed', 'Original20 acceptance was not passed')
    all_episodes,requests,stages,memory = [],[],[],[]
    signatures,execution_order,source_steps = {},[],{}
    excluded_warmup_errors = []
    raw_contracts = []
    for i,(batch,expected_count) in enumerate(zip(report['batches'],(20,80))):
        directory = evidence.path(batch['directory'])
        require(directory.is_relative_to(REPAIR_LOGS) and directory.name==group, 'Historical/cancelled or wrong-group source')
        result_path = directory/'result.json'
        evidence.verify(result_path,batch['result_sha256'])
        result = evidence.load(result_path)
        require(result['status']=='complete' and result['case_count']==expected_count and result['group']==group, 'Incomplete raw batch')
        require(batch['source_checks'] and all(check['passed'] for check in batch['source_checks']),
                'Collector recorded missing/failing source evidence')
        if i==0:
            require(accepted['groups'][group]['directory']==batch['directory']
                    and accepted['groups'][group]['result_sha256']==batch['result_sha256'], 'Accepted20 points elsewhere')
        service,folder = directory/'service',directory/'navigation'
        summary = evidence.load(folder/'summary.json')
        manifest = evidence.load(folder/'manifest.json')
        require(manifest==evidence.load(directory/'case_manifest.json'), 'Raw evaluator manifest differs')
        frozen = split_manifests[i]
        require(manifest['episodes']==frozen['episodes'] and manifest['trajectory_ids']==frozen['trajectory_ids']
                and manifest['version']=='navigation_repair_20261112', 'Wrong20/80 source roster')
        require(summary['status']=='complete' and summary['episode_count']==expected_count
                and summary['warmup_requests_excluded']==2, 'Incomplete summary/warmup evidence')
        order = [str(e['episode_id']) for e in summary['episodes']]
        require(len(order)==len(set(order))==expected_count and set(order)==set(map(str,frozen['episodes'])), 'Batch membership differs')
        require(order==batch['execution_episode_order'], 'Recorded execution order differs')
        require(batch['timing_start']==result['timing_start'] and batch['timing_end_exclusive']==result['timing_end_exclusive'], 'Batch timing offsets differ')
        index = result['timing_start']
        warmups = evidence.load(folder/'warmups.json')
        require(len(warmups)==2, 'Missing excluded navigation warmups')
        launch = evidence.load(service/'launch.json')
        require(launch['group']==group and result['source_hashes']==launch['sources_sha256'], 'Raw runtime identity differs')
        # Bind the comparison contract to original artifacts, not a report's
        # self-consistent but otherwise unverified fingerprint.
        import yaml
        from collect_navigation_repair100 import normalized_manifest, launch_semantics
        provenance = evidence.load(folder/'provenance.json')
        repair = evidence.load(service/'repair_execution_audit.json')
        capture = evidence.load(service/'startup_graph_capture.json')
        habitat = yaml.safe_load(evidence.read_bytes(folder/'habitat_config.yaml'))
        require(list(map(str, habitat['DATASET'].pop('EPISODES_ALLOWED'))) == list(map(str, manifest['episodes'])),
                'Raw Habitat roster differs')
        habitat['SIMULATOR'].pop('SCENE')
        evidence.verify(folder/'system_prompt.txt', provenance['system_prompt_sha256'])
        raw_contract = dict(launch=launch,
            execution=dict(weight_bits=repair['weight_bits'],activation_bits=repair['activation_bits'],
                extension=repair['extension'],launch_table=launch_semantics(repair.get('frozen_launch_table')),
                graph_keys=capture['graph_keys']),
            task_manifest=normalized_manifest(manifest),habitat=habitat,
            environment_evidence=dict(inference_versions=launch['versions'],renderer=summary['renderer'],
                habitat_source_sha256=provenance['scripts']['simplevln_env.py'],
                verification_scope='Saved engine versions, complete Habitat settings and environment source; historical Habitat binary/library digests were not recorded.'),
            system_prompt_sha256=provenance['system_prompt_sha256'])
        require(batch['sources']['evaluator_sources_sha256']==provenance['scripts'], 'Reported evaluator differs from raw provenance')
        if raw_contracts:
            require(raw_contract == raw_contracts[0], 'Raw20/80 runtime conditions differ')
        raw_contracts.append(raw_contract)
        for row in warmups:
            require(row.get('model_called') is True and row.get('error') in QUALITY_ERRORS|{None}, 'Invalid warmup')
            if row.get('error'):
                excluded_warmup_errors.append(row['error'])
            request_timing(evidence,service,index,row,launch['recipe']['method'])
            index += 1
        for ep in summary['episodes']:
            calculated,calls,event_rows,signature,index = verify_case(evidence,folder,ep,service,index,launch['recipe']['method'])
            eid = calculated['episode_id']
            require(eid not in signatures, '20/80 overlap')
            signatures[eid] = signature
            source_steps[eid] = relative(folder/('episode_'+eid)/'steps.jsonl')
            raw_saved = next(e for e in result['episodes'] if str(e['episode_id'])==eid)
            final_saved = next(e for e in report['episodes'] if str(e['episode_id'])==eid)
            for saved in (raw_saved,final_saved):
                for key,value in calculated.items():
                    if isinstance(value,(int,float)):
                        equal(saved[key],value,'Case '+eid+'.'+key)
                    else:
                        require(saved[key]==value, 'Case metadata differs: '+eid+'.'+key)
            all_episodes.append(calculated)
            requests.extend(calls)
            stages.extend(event_rows)
        require(index==result['timing_end_exclusive'], 'Raw requests/timing count differs')
        require(sorted(int(p.stem.split('_')[1]) for p in service.glob('timing_*.json'))==list(range(index)), 'Timing gap/extra file')
        memory_data = evidence.read_bytes(folder/'gpu_samples.csv').decode()
        samples = [float(line.split(',')[1]) for line in memory_data.splitlines() if len(line.split(','))>=3]
        require(samples and all(math.isfinite(v) and v>0 for v in samples), 'Invalid memory distribution')
        memory.extend(samples)
        execution_order.extend(order)
    batch_prefixes = tuple(batch['directory'].rstrip('/')+'/' for batch in report['batches'])
    for source,digest in evidence.verified.items():
        if source.startswith(batch_prefixes):
            require(report['evidence_files_sha256'].get(source)==digest,
                    'Raw source is not bound to the collector evidence inventory: '+source)
    identities = {e['episode_id']:e['trajectory_id'] for e in all_episodes}
    expected = dict(zip(map(str,original['episodes']),map(str,original['trajectory_ids'])))
    require(identities==expected and len(all_episodes)==100, 'Union is not the original100')
    require(report['execution_episode_order']==execution_order
            and report['display_episode_order']==list(map(str,original['episodes'])), 'Execution/display order conflated')
    require([str(e['episode_id']) for e in report['episodes']]==report['display_episode_order'], 'Collector display rows reordered')
    computed = recompute(all_episodes,requests,stages,memory)
    numeric_map(report,computed,'Collector100')
    require(report['passed_stability']==(not computed['error_count'] and not excluded_warmup_errors), 'Stability label differs')
    contract = report['runtime_contract']
    require(contract == raw_contracts[0], 'Reported runtime contract differs from original artifacts')
    require(report['same_conditions_sha256']==json_sha(dict(launch=contract['launch'],execution=contract['execution'],
        task=contract['task_manifest'],habitat=contract['habitat'])), 'Collector condition fingerprint differs')
    return dict(group=group,report_path=relative(path),report_sha256=evidence.verified[relative(path)],
        metrics={key:report[key] for key in computed},
        episodes=report['episodes'],initial_inputs=signatures,step_sources=source_steps,
        original20_directory=report['batches'][0]['directory'],original20_result_sha256=report['batches'][0]['result_sha256'],
        excluded_warmup_errors=excluded_warmup_errors,execution_episode_order=execution_order,
        same_conditions_sha256=report['same_conditions_sha256'],
        common_conditions=dict(task=contract['task_manifest'],habitat=contract['habitat'],
            system_prompt_sha256=contract['system_prompt_sha256'],
            versions=contract['launch']['versions'],
            engine={key:value for key,value in contract['launch']['engine'].items()
                    if key not in ('custom_module_map','port','server_port','model_name')},
            evaluator_sources_sha256=report['batches'][0]['sources']['evaluator_sources_sha256']),
        original20_qualification=report['screen_acceptance'],raw_verification='passed',
        field_sources=dict(metrics100=dict(path=relative(path),sha256=evidence.verified[relative(path)],
            scope='Collector values, independently checked against pooled raw steps/timings; no percentile averaging.'),
            initial_inputs='First model_called step0 request in each listed raw steps.jsonl.'))


def screen_status(row, details):
    if row.get('passed') is True:
        return 'passed_20_waiting_80'
    status = row.get('status','pending')
    if status=='review_required':
        return 'completed_20_review_required'
    if status in ('failed','audit_failed','run_failed','fixed_speed_gate_failed','validation_failed'):
        return 'failed_20' if row.get('case_count',0)==20 else 'failed_precheck_or_execution'
    if status in ('fixed_input_only_complete',):
        return 'fixed_input_only'
    if row.get('case_count')==20 or details.get('metrics',{}):
        return 'completed_20_pending_acceptance'
    if status=='pending':
        reason = details.get('error') or row.get('reason', '')
        if reason.startswith('Started but no complete result.json'):
            return 'running_or_incomplete'
        if reason in ('Group directory not created; not run', 'Not run in this audit'):
            return 'not_run'
        directory = details.get('directory')
        if directory:
            path = Evidence().path(directory)
            if path.is_dir():
                return 'running_or_incomplete'
        return 'not_run'
    return status


def build_rows(completed, latest):
    table = latest['table']
    require(len(table)==9 and {r['group'] for r in table}==set(GROUPS), 'Latest20 audit must retain FP16+8 unique rows')
    baseline = completed.get('fp16')
    baseline_screen = latest.get('groups',{}).get('fp16',{})
    baseline_current = bool(baseline and (not baseline_screen.get('result_sha256') or
        (baseline_screen['result_sha256']==baseline['original20_result_sha256']
         and baseline_screen.get('directory')==baseline['original20_directory'])))
    rows = []
    for group in GROUPS:
        screen = next(r for r in table if r['group']==group)
        detail = latest.get('groups',{}).get(group,{})
        item = completed.get(group)
        row = dict(group=group,status=screen_status(screen,detail),
            latest20=dict(verdict=screen,pipeline_status=detail.get('pipeline_status'),
                directory=detail.get('directory'),result_sha256=detail.get('result_sha256'),
                quality_errors=detail.get('quality_errors'),error=detail.get('error')),
            metrics100=None,comparison=dict(status='pending_comparison',reason='Requires both this group and FP16 complete100 evidence.'),
            final100_quality_threshold=None,
            final100_quality_statement='No preregistered100 quality pass threshold;20 gates are not extrapolated.')
        if item:
            row['status'] = 'complete_100_with_quality_errors' if item['metrics']['error_count'] else 'complete_100'
            row['metrics100'] = item['metrics']
            row['report100'] = dict(path=item['report_path'],sha256=item['report_sha256'])
            row['field_sources'] = item['field_sources']
            row['original20_qualification'] = item['original20_qualification']
            row['latest20_matches_accumulated20'] = (not detail.get('result_sha256') or
                (detail['result_sha256']==item['original20_result_sha256'] and detail.get('directory')==item['original20_directory']))
            row['excluded_warmup_errors'] = item['excluded_warmup_errors']
            if baseline:
                mismatches = [eid for eid in baseline['initial_inputs']
                    if item['initial_inputs'].get(eid)!=baseline['initial_inputs'][eid]]
                if not baseline_current:
                    row['comparison'] = dict(status='pending_comparison',reason='FP16100 belongs to a different20 implementation/run than the latest FP16 screen.')
                elif item['common_conditions'] != baseline['common_conditions']:
                    row['comparison'] = dict(status='blocked_common_condition_mismatch',
                        reason='Task/engine/generation/environment/evaluator conditions differ from FP16100; precision module-map filenames are the only implementation-path exception.')
                elif mismatches:
                    row['comparison'] = dict(status='blocked_initial_input_mismatch',mismatched_episode_ids=mismatches,
                        reason='Original first-request text/images/count/prompt-token identity differs from FP16.')
                elif not row['latest20_matches_accumulated20']:
                    row['comparison'] = dict(status='pending_comparison',reason='Latest20 belongs to a different implementation/run than this completed100.')
                else:
                    m,ref = item['metrics'],baseline['metrics']
                    row['comparison'] = dict(status='baseline' if group=='fp16' else 'descriptive_comparison',
                        baseline_report=dict(path=baseline['report_path'],sha256=baseline['report_sha256']),
                        initial_input_identity_cases=100,initial_input_mismatches=[],
                        success_count_delta=m['success_count']-ref['success_count'],sr_delta=m['sr']-ref['sr'],
                        spl_delta=m['spl']-ref['spl'],request_p50_ratio=m['request_median_s']/ref['request_median_s'],
                        request_p95_ratio=m['request_p95_s']/ref['request_p95_s'],
                        scope='Observed closed-loop distributions; different subsequent trajectories/output lengths prevent pure algorithm attribution.')
        rows.append(row)
    return rows


STATUS_ZH = {'complete_100':'100条完成','complete_100_with_quality_errors':'100条完成，含质量错误',
    'passed_20_waiting_80':'仅20条通过，待80条','failed_20':'20条未通过',
    'failed_precheck_or_execution':'预检/执行未通过','completed_20_review_required':'20条需复核',
    'completed_20_pending_acceptance':'20条完成，待验收','fixed_input_only':'仅固定输入测量','not_run':'未运行',
    'running_or_incomplete':'已启动，未完成','natural_output_gate_failed':'自然输出格式预检未通过'}


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reports',nargs='*',type=Path,default=[])
    parser.add_argument('--screen-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT/'reports') and output.suffix=='.json'
            and not output.exists() and not output.with_suffix('.md').exists(), 'Use fresh reports/.json and.md paths')
    evidence = Evidence()
    latest = evidence.load(args.screen_audit)
    original = evidence.load(ROSTER/'original100.json')
    manifests = [evidence.load(ROSTER/name) for name in ('subset20.json','remaining80.json')]
    roster = evidence.load(ROSTER/'manifest.json')
    for name,digest in roster['artifact_files_sha256'].items():
        evidence.verify(ROSTER/name,digest)
    completed = {}
    for path in args.reports:
        item = verify_100(evidence,path.resolve(),original,manifests)
        require(item['group'] not in completed, 'Duplicate100 group; choose the intended explicit run')
        completed[item['group']] = item
    rows = build_rows(completed,latest)
    source = relative(args.screen_audit)
    for row in rows:
        row.setdefault('field_sources',{})['latest20'] = dict(path=source,sha256=evidence.verified[source],
            pointer='table[group='+row['group']+']; groups['+row['group']+']')
    evidence.read_bytes(Path(__file__))
    evidence.finish()
    report = dict(status='complete_inventory',created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Fixed1+8 inventory; completed100 and20 qualification are separate statuses.',
        completed100_groups=list(completed),all_nine_100_complete=len(completed)==9,
        all_eight_quantized_100_comparisons_available=all(
            row['comparison']['status']=='descriptive_comparison' for row in rows if row['group']!='fp16'),
        baseline100_available='fp16' in completed,table=rows,
        initial_input_comparison_scope='Only each case first real request; subsequent policy trajectories may differ.',
        latest20_source=dict(path=source,sha256=evidence.verified[source],created_utc=latest.get('created_utc')),
        raw100_verification={group:{k:v for k,v in item.items() if k not in ('metrics','episodes','original20_qualification')}
                             for group,item in completed.items()},
        verified_non_model_evidence_sha256=evidence.verified,
        model_assets_sha256_referenced_from_collectors=evidence.referenced_model_assets,
        validation_status=('Supplied complete100 reports independently checked against their raw evidence at invocation.'
            if completed else 'No100 reports supplied; this is only the latest20 status inventory. No100 raw-evidence validation was performed.'),
        limitations=['Complete100 is completion, not a quality pass; no100 threshold is invented from the20 gates.',
            'Absent80 leaves100 fields empty. Latest20 failures/review requirements remain visible even if an earlier100 exists.',
            'SR/SPL use model-origin STOP, no error, distance<3m; quality failures stay in the100 denominator.',
            'HTTP quantiles are checked from all real requests and displayed from collector values; no batch/episode percentile averaging.',
            'Closed-loop HTTP distributions have different subsequent trajectories/token lengths and do not isolate quantization algorithm speed.',
            'Without matched FP16100 there is no comparative conclusion; differing first inputs block comparison.',
            'Model assets/weights are not opened here; their SHA identity is referenced from the preceding collector verification.',
            'Whole-card memory includes desktop/prewarm and200ms sampling; decode uses actual forwards, including post-response work.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    lines=['# 量化修复同100条：FP16＋八组状态','',
        '100条完成状态与20条验收资格分开记录；空白结果表示尚无完整100条证据。','',
        '| 配置 | 100/执行状态 | 最新20资格 | 比较状态 | 成功/100 | SR | SPL | ΔSR | ΔSPL | HTTP P50/s | HTTP P95/s | 质量错误 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m,comparison=row['metrics100'],row['comparison']
        values=([str(m['success_count']),f"{m['sr']:.3f}",f"{m['spl']:.4f}",
                 f"{comparison['sr_delta']:+.3f}" if 'sr_delta' in comparison else '—',
                 f"{comparison['spl_delta']:+.4f}" if 'spl_delta' in comparison else '—',
                 f"{m['request_median_s']:.3f}",f"{m['request_p95_s']:.3f}",str(m['error_count'])]
                if m else ['—']*8)
        lines.append('| '+ ' | '.join([row['group'],STATUS_ZH.get(row['status'],row['status']),
            str(row['latest20']['verdict'].get('status','pending')),comparison['status']]+values)+' |')
    lines += ['', '100条没有额外“质量达标”阈值；只展示实际SR/SPL及同100 FP16差值。缺少FP16100时比较状态为 `pending_comparison`。初始输入不一致会阻止比较。', '',
        'HTTP来自各自闭环轨迹的真实请求分布，不能纯归因于量化算法。后续轨迹不同是允许的；逐案例输入核对只针对第一次请求。', '',
        f"完整状态、逐字段来源、初始请求核对、失败信息及证据SHA见[{output.name}]({output.name})。模型权重未在汇总步骤重新读取，引用此前累计器的校验结果。",'']
    with output.with_suffix('.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines))
    print(json.dumps(dict(output=relative(output),completed100_groups=list(completed),baseline100_available='fp16' in completed),ensure_ascii=False))


if __name__=='__main__':
    main()
