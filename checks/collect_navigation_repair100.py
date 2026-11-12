"""CPU-only audit of an accepted NEW repair20 plus its unchanged remaining80.

Example (all paths are new repair runs, never historical precision runs)::
  python -B checks/collect_navigation_repair100.py \
    --screen-dir logs/navigation_repair_20261112/screen/fp16 \
    --remaining-dir logs/navigation_repair_20261112/remaining/fp16 \
    --screen-audit reports/repair20_audit.json --output reports/repair_fp16_100.json

The 20-case audit must have accepted this exact group/result before the 80 run
started. Both batches must retain raw steps, full timing, source snapshots,
manifests and provenance. Runtime/model/policy/evaluator/engine/generation and
the selected kernel launch table must be unchanged. Different orchestration
controllers are allowed only with independently verifiable source receipts;
their identity is retained and no measured runtime difference is ignored.

This does not authorize or launch jobs. Missing/changed evidence fails closed:
rerun the affected20 instead of combining incompatible data. Quality failures
in the remaining80 are retained in the 100 denominator. Infrastructure errors
or incomplete requests make the evidence unauditable and produce no report.
Only standard-library modules and the independent CPU audit helper are used;
PyYAML is imported solely to compare the saved Habitat configuration.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

import audit_navigation_repair as audit


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / 'data/baseline/navigation_repair_20261112'
REPAIR_LOGS = ROOT / 'logs/navigation_repair_20261112'
QUALITY_ERRORS = audit.QUALITY_ERRORS
require, equal_number = audit.require, audit.equal_number
read, sha, relative = audit.read, audit.sha, audit.relative
MANIFEST_IDENTITY_FIELDS = {
    'episodes', 'trajectory_ids', 'scope', 'repair_roster_sha256',
    'model', 'dtype', 'backend', 'repair_group', 'precision_metadata',
    'repair_recipe', 'source_precision',
}


def utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Unqualified timestamp: ' + value)
    return result.astimezone(timezone.utc)


def json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalized_manifest(value):
    return {k:v for k,v in value.items() if k not in MANIFEST_IDENTITY_FIELDS}


def normalized_engine(value):
    # These are transport labels, never model execution parameters.
    return {k:v for k,v in value.items() if k not in ('port', 'server_port', 'model_name')}


def launch_semantics(receipt):
    value = copy.deepcopy(receipt)
    if value is None:
        return None
    # Retain selected configurations/geometry/precision, remove observation cost.
    for key in ('prepare_seconds', 'preparation_seconds', 'compiled_entries',
                'autotuner_cache_entries', 'cache_entries', 'frozen_cache_entries',
                'kernel_cache_entries', 'kernel_cache_sizes', 'selection_mode', 'frozen_table_source'):
        value.pop(key, None)
    return value


def habitat_contract(evidence, folder, manifest):
    import yaml
    path = folder / 'habitat_config.yaml'
    evidence.hashes[relative(path)] = sha(path)
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    allowed = list(map(str, config['DATASET'].pop('EPISODES_ALLOWED')))
    require(allowed == list(map(str, manifest['episodes'])), 'Habitat episode list differs from manifest')
    # Habitat initializes the simulator in the first selected scene. That scene
    # necessarily differs between the disjoint20/80 manifests.
    initial_scene = config['SIMULATOR'].pop('SCENE')
    valid_scenes = [str(r['path']).replace('\\', '/') for r in manifest['scene_files'] if str(r['path']).endswith('.glb')]
    require(any(str(initial_scene).replace('\\', '/').endswith(scene) for scene in valid_scenes),
            'Initial Habitat scene is outside the frozen scene set')
    require(config['SEED'] == 7 and config['ENVIRONMENT']['MAX_EPISODE_STEPS'] == 400,
            'Habitat seed/step budget changed')
    require(config['SIMULATOR']['HABITAT_SIM_V0']['GPU_DEVICE_ID'] == -1,
            'Habitat renderer no longer uses the frozen CPU device')
    return dict(config=config, initial_scene=initial_scene)


def source_identity(evidence, protocol, provenance, launch):
    for name, expected in launch['sources_sha256'].items():
        evidence.source('checks/' + name, expected)
    for name, expected in provenance['scripts'].items():
        evidence.source('checks/' + name, expected)
    for key, name in [('navigation_policy_sha256', 'checks/simplevln_policy.py'),
                      ('evaluator_sha256', 'checks/run_navigation_precision_diagnostic.py')]:
        evidence.source(name, protocol[key])
        require(protocol[key] == provenance['scripts'][Path(name).name], 'Protocol/evaluator source disagreement: ' + key)
    controller = audit.controller_source(protocol, evidence)
    evidence.source(controller, protocol['controller_sha256'])
    if protocol.get('source_snapshot_sha256'):
        require(protocol['source_snapshot_sha256'] == sha(evidence.directory.parent / 'source_snapshot.json'),
                'Controller snapshot manifest changed')
    # Include evaluator helpers absent from its four-file provenance receipt.
    extras = {}
    for source in ('checks/navigation_precision_quality.py', 'scripts/run_habitat_task_python.sh'):
        if source in evidence.snapshots:
            extras[source] = evidence.snapshots[source]['sha256']
            evidence.source(source, extras[source])
        elif source == 'checks/navigation_precision_quality.py':
            raise ValueError('Missing frozen quality-classification helper: ' + source)
    return dict(controller_source=controller, controller_sha256=protocol['controller_sha256'],
        runtime_sources_sha256=launch['sources_sha256'], evaluator_sources_sha256=provenance['scripts'],
        additional_sources_sha256=extras)


def execution_contract(evidence, service, launch):
    group, recipe = launch['group'], launch['recipe']
    expected = (16, 16) if group == 'fp16' else tuple(map(int, group.split('_w')[1].split('a')))
    require((recipe['weight_bits'], recipe['activation_bits']) == expected, 'Wrong declared precision')
    execution = evidence.load(service / 'execution_audit.json')
    repair = evidence.load(service / 'repair_execution_audit.json')
    capture = evidence.load(service / 'startup_graph_capture.json')
    require(execution['first_native_forward_verified'] and execution['native_forward_preserved'],
            'Native model execution proof missing')
    require(execution['group'] == recipe['method'] and execution['graph_requested'] == 'on', 'Execution group/graph mismatch')
    require((repair['weight_bits'], repair['activation_bits']) == expected and repair['native_forward'],
            'Actual repair precision differs')
    require(capture['actual_capture_observed'], 'No actual CUDA graph capture')
    if group != 'fp16':
        packed = execution['native_awq_receipt']
        require(packed['verified'] and packed['module_count'] == 144 and packed['loaded_quantized_tensor_count'] == 756,
                'Incomplete packed weight loading/execution')
        table = repair['frozen_launch_table']
        require(table['frozen'], 'Unfrozen precision kernel')
        if expected == (4, 16):
            require(not repair['extension'] and table['loaded_module_count'] == 144
                    and len(table['shapes']) == 4 and table['actual_m_preserved']
                    and table['generic_m_specialization_disabled']
                    and not table['runtime_autotuning_allowed'] and not table['runtime_cold_compilation_allowed'],
                    'W4A16 launch guard evidence missing')
            origin = table.get('frozen_table_source')
            if origin:
                path = Path(origin['path']).resolve()
                require(path.is_relative_to(ROOT) and sha(path) == origin['sha256'], 'Frozen table origin SHA changed')
                evidence.hashes[relative(path)] = origin['sha256']
                if origin.get('identity_source'):
                    identity = origin['identity_source']
                    identity_path = Path(identity['path']).resolve()
                    require(identity_path.is_relative_to(ROOT) and sha(identity_path) == identity['sha256'], 'Frozen table identity origin changed')
                    evidence.hashes[relative(identity_path)] = identity['sha256']
        else:
            calls = repair['actual_extension_calls']
            require(repair['extension'] and len(calls) == 144 and set(calls.values()) == {1},
                    'Requested precision did not actually execute every projection')
            require(table['prepared_projection_count'] == 4 and table['private_cold_compile_guard_installed']
                    and table['cache_unchanged_since_freeze'] and table['allowed_dynamic_rows'] == [1, 3072],
                    'Experimental precision launch guard evidence missing')
    return dict(weight_bits=expected[0], activation_bits=expected[1], extension=repair['extension'],
        launch_table=launch_semantics(repair.get('frozen_launch_table')), graph_keys=capture['graph_keys'])


def summarize(episodes, requests, timing_rows, memory):
    require(episodes and requests and memory, 'Cannot summarize missing cases/requests/memory')
    latencies = [r['model_request_s'] for r in requests]
    errors = Counter(r.get('error') for r in requests if r.get('error'))
    count = len(episodes)
    stage = audit.stages(timing_rows)
    return dict(case_count=count, success_count=sum(e['success'] for e in episodes),
        sr=sum(e['success'] for e in episodes)/count, spl=sum(e['spl'] for e in episodes)/count,
        actual_model_calls=len(requests), prompt_tokens=sum(r['usage']['prompt_tokens'] for r in requests),
        completion_tokens=sum(r['usage']['completion_tokens'] for r in requests),
        request_median_s=statistics.median(latencies), request_p95_s=audit.p95(latencies),
        request_total_s=sum(latencies), sampled_whole_card_peak_mib=max(memory),
        error_count=sum(errors.values()), infrastructure_errors=0,
        quality_errors=dict(errors), collisions=sum(e['collisions'] for e in episodes),
        actual_prefill_forwards=len(timing_rows), actual_decode_forwards=stage['actual_decode_forwards'],
        actual_total_forwards=len(timing_rows)+stage['actual_decode_forwards'], stage_summary=stage)


def audit_batch(directory, base_path, expected_count):
    require(directory.resolve().is_relative_to(REPAIR_LOGS), 'Historical/cancelled runs cannot be accumulated')
    evidence = audit.Evidence(directory)
    result = evidence.load(directory / 'result.json')
    require(result['status'] == 'complete' and result['case_count'] == expected_count, 'Incomplete/wrong-sized result')
    group = result['group']
    require(group in audit.GROUPS and directory.name == group, 'Group directory/result mismatch')
    service, folder = directory / 'service', directory / 'navigation'
    launch = evidence.load(service / 'launch.json')
    require(launch['group'] == group and launch['model_name'] == 'navigation-repair-' + group, 'Wrong service identity')
    require(result['source_hashes'] == launch['sources_sha256'], 'Result/runtime source receipt mismatch')
    protocol = evidence.load(directory.parent / 'protocol.json')
    base = evidence.load(base_path)
    require(protocol['manifest_sha256'] == sha(base_path), 'Run did not use the designated frozen20/80 manifest')
    manifest = evidence.load(folder / 'manifest.json')
    require(manifest == evidence.load(directory / 'case_manifest.json'), 'Evaluator used another manifest')
    require(manifest['version'] == base['version'] == 'navigation_repair_20261112', 'Old run identity rejected')
    require(normalized_manifest(manifest) == normalized_manifest(base), 'Frozen prompt/input/generation/task manifest changed')
    for key in ('episodes', 'trajectory_ids', 'repair_roster_sha256'):
        require(manifest[key] == base[key], 'Frozen roster changed: ' + key)
    ids = dict(zip(map(str, manifest['episodes']), map(str, manifest['trajectory_ids'])))
    require(len(ids) == len(manifest['episodes']) == expected_count, 'Duplicate or missing cases')
    summary = evidence.load(folder / 'summary.json')
    require(summary['status'] == 'complete' and summary['episode_count'] == expected_count
            and summary['warmup_requests_excluded'] == 2, 'Incomplete evaluator summary/warmup convention')
    require(summary['model'] == manifest['model'] == launch['model_name'], 'Wrong model in navigation')
    dtype = 'FP16' if group == 'fp16' else group.split('_', 1)[1].upper()
    require(manifest['dtype'].upper() == dtype, 'Manifest precision mislabels actual group')
    provenance = evidence.load(folder / 'provenance.json')
    require(provenance['manifest_sha256'] == sha(directory / 'case_manifest.json'), 'Manifest provenance mismatch')
    prompt = folder / 'system_prompt.txt'
    evidence.hashes[relative(prompt)] = sha(prompt)
    require(provenance['system_prompt_sha256'] == sha(prompt)
            and prompt.read_text() == 'You are a visual language navigation model.', 'System prompt changed')
    sources = source_identity(evidence, protocol, provenance, launch)
    execution = execution_contract(evidence, service, launch)
    habitat = habitat_contract(evidence, folder, manifest)
    model = ROOT / launch['recipe']['model']
    evidence.source(relative(model / 'config.json'), launch['model_config_sha256'])
    evidence.source(relative(model / 'model.safetensors'), launch['model_weights_sha256'])
    contract = evidence.load(ROOT / 'setup/navigation-stage2-execution-20261109.json')
    for name in ('chat_template.jinja', 'generation_config.json', 'preprocessor_config.json',
                 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'):
        evidence.source(relative(model / name), contract['reference_files_sha256'][
            'models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy/' + name])
    evidence.source('tmp/habitat_setup/navigation_source_review/eval/vln/eval.py', manifest['author_policy_sha256'])
    fixed, fixed_metrics, tensor_inputs, fixed_gate = audit.audit_fixed(evidence, directory, result, launch)
    index = len(fixed)
    if (directory / 'fixed_flush_request.json').exists():
        tail = evidence.load(directory / 'fixed_flush_request.json')
        require(tail['response']['model'] == launch['model_name'], 'Wrong fixed tail model')
        audit.timing(evidence, service, index, tail['response']['usage'], launch['recipe']['method'])
        index += 1
    require(result['timing_start'] == index, 'Result navigation timing start does not follow fixed warmups')
    timing_start = index
    warmups = evidence.load(folder / 'warmups.json')
    require(len(warmups) == 2, 'Expected exactly two excluded navigation warmups')
    warmup_errors = []
    for row in warmups:
        error = audit.verify_record(row, 'excluded warmup')
        require(row['model_called'], 'Warmup has no real request')
        if error:
            warmup_errors.append(error)
        audit.timing(evidence, service, index, row['usage'], launch['recipe']['method'])
        index += 1
    order = [str(e['episode_id']) for e in summary['episodes']]
    require(len(order) == len(set(order)) == expected_count and set(order) == set(ids), 'Summary roster mismatch')
    require(set(str(e['episode_id']) for e in result['episodes']) == set(ids)
            and len(result['episodes']) == expected_count, 'Saved result roster mismatch')
    episodes, requests, timing_rows = [], [], []
    all_rows, step_hashes = [], {}
    for episode in summary['episodes']:
        eid = str(episode['episode_id'])
        require(str(episode['trajectory_id']) == ids[eid], 'Trajectory ID mismatch')
        require(evidence.load(folder / ('episode_' + eid) / 'result.json') == episode, 'Episode result/summary mismatch')
        path = folder / ('episode_' + eid) / 'steps.jsonl'
        evidence.hashes[relative(path)] = sha(path)
        step_hashes[relative(path)] = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(0 < len(rows) == episode['steps'] <= 400 and rows[0]['model_called'], 'Bad step count/initial call')
        path_length = 0.0
        for number, row in enumerate(rows):
            require(row['step'] == number and str(row['episode_id']) == eid, 'Step order/identity changed')
            error = audit.verify_record(row, eid + ':' + str(number))
            if error:
                require(number == len(rows)-1 and episode['terminal_stop_source'] == 'error_fallback',
                        'Quality failure did not terminate as failed STOP')
            before, after = row['before']['position'], row['after']['position']
            require(len(before) == len(after) == 3 and all(math.isfinite(v) for v in before+after), 'Invalid positions')
            if number:
                require(all(math.isclose(a,b,abs_tol=1e-7) for a,b in zip(before, rows[number-1]['after']['position'])),
                        'Saved path is discontinuous')
            path_length += math.sqrt(sum((a-b)**2 for a,b in zip(before, after)))
            if row['model_called']:
                requests.append(row)
                timing_rows.append(audit.timing(evidence, service, index, row['usage'], launch['recipe']['method']))
                index += 1
        all_rows.extend(rows)
        equal_number(episode['path_length_m'], path_length, 'Saved path length')
        require(episode['metrics'] == rows[-1]['metrics'], 'Final metrics/last step mismatch')
        last = rows[-1]
        expected_stop_source = ('error_fallback' if last.get('error') else 'model') if last['action'] == 'STOP' else 'step_limit'
        require(episode['terminal_stop_source'] == expected_stop_source, 'Incorrect terminal STOP source')
        require(last['action'] == 'STOP' or len(rows) == 400, 'Nonterminal short episode')
        success = int(last['action'] == 'STOP' and expected_stop_source == 'model'
                      and not last.get('error') and last['metrics']['distance_to_goal'] < 3.)
        shortest = episode['start_distance_m']
        require(math.isfinite(shortest) and shortest > 0, 'Invalid starting distance')
        spl = success*shortest/max(shortest, path_length)
        equal_number(episode['metrics']['spl'], episode['metrics']['success']*shortest/max(shortest,path_length), 'Habitat SPL')
        calculated = dict(episode_id=eid, trajectory_id=ids[eid], success=success, spl=spl,
            steps=len(rows), actual_model_calls=sum(r['model_called'] for r in rows),
            errors=sum(bool(r.get('error')) for r in rows),
            collisions=sum(bool(r['metrics']['collisions']['is_collision']) for r in rows),
            terminal_stop_source=expected_stop_source)
        saved = next(e for e in result['episodes'] if str(e['episode_id']) == eid)
        for key, value in calculated.items():
            if isinstance(value, (int,float)):
                equal_number(saved[key], value, eid + '.' + key)
            else:
                require(saved[key] == value, 'Episode result differs: ' + key)
        episodes.append(calculated)
    require(result['step_files_sha256'] == step_hashes, 'Step file hash inventory differs')
    require(result['timing_end_exclusive'] == index, 'Navigation timing end mismatch')
    files = sorted(int(p.stem.split('_')[1]) for p in service.glob('timing_*.json'))
    require(files == list(range(index)), 'Timing files have gaps/extras; only final unflushed tail may be absent')
    tail = evidence.load(directory / 'flush_request.json')
    require(tail['response']['model'] == launch['model_name']
            and tail['response']['usage']['completion_tokens'] == 18, 'Missing final excluded tail refresh')
    memory_path = folder / 'gpu_samples.csv'
    evidence.hashes[relative(memory_path)] = sha(memory_path)
    memory = [float(line.split(',')[1]) for line in memory_path.read_text().splitlines() if len(line.split(',')) >= 3]
    require(memory and all(math.isfinite(v) and v > 0 for v in memory), 'Invalid memory samples')
    metrics = summarize(episodes, requests, timing_rows, memory)
    for key in ('case_count','success_count','sr','spl','actual_model_calls','prompt_tokens','completion_tokens',
                'request_median_s','request_p95_s','request_total_s','sampled_whole_card_peak_mib','error_count','infrastructure_errors'):
        equal_number(result[key], metrics[key], 'Batch result.' + key)
    audit.check_numeric_map(result['stage_summary'], metrics['stage_summary'], 'Batch stage metrics')
    require(result['quality_errors'] == metrics['quality_errors'] and result['warmup_errors'] == warmup_errors,
            'Saved quality error classification differs')
    require(result['passed_stability'] == (not metrics['error_count'] and not warmup_errors), 'Saved stability verdict differs')
    require(result['manifest_sha256'] == sha(folder / 'manifest.json'), 'Saved manifest SHA mismatch')
    equal_number(summary['decision_count'], len(all_rows), 'Evaluator decision count')
    equal_number(summary['error_count'], metrics['error_count'], 'Evaluator errors')
    equal_number(summary['sr'], sum(e['metrics']['success'] for e in summary['episodes'])/expected_count, 'Evaluator raw SR')
    equal_number(summary['spl'], sum(e['metrics']['spl'] for e in summary['episodes'])/expected_count, 'Evaluator raw SPL')
    equal_number(summary['model_stop_success_count'], metrics['success_count'], 'Evaluator model STOP successes')
    equal_number(summary['fallback_stop_success_count'], sum(e['metrics']['success'] == 1 and e['terminal_stop_source'] == 'error_fallback'
                 for e in summary['episodes']), 'Evaluator fallback successes')
    # Historical evaluator summary includes queued zeroes; verify it but never
    # use it as the real-request latency distribution.
    raw_latencies = [r['model_request_s'] for r in all_rows]
    for key, expected in [('median',statistics.median(raw_latencies)), ('p95',audit.p95(raw_latencies)),
                          ('mean',statistics.mean(raw_latencies))]:
        equal_number(summary['model_request_latency_s'][key], expected, 'Evaluator all-action latency.'+key)
    require(summary['gpu_memory']['interval_ms'] == 200, 'GPU memory sampling interval changed')
    require(summary['renderer'] == 'software Mesa llvmpipe; simulated movement time separately recorded', 'Renderer contract changed')
    equal_number(summary['gpu_memory']['sampled_peak_mib'], max(memory), 'Evaluator memory peak')
    require(not evidence.failures, 'Source evidence failed: ' + repr(evidence.failures))
    return dict(group=group, directory=relative(directory), result_sha256=sha(directory/'result.json'),
        protocol=protocol, provenance=provenance, launch=launch, sources=sources,
        manifest=manifest, habitat=habitat, execution=execution, episodes=episodes,
        environment_evidence=dict(inference_versions=launch['versions'], renderer=summary['renderer'],
            habitat_source_sha256=provenance['scripts']['simplevln_env.py'],
            verification_scope='Saved engine versions, complete Habitat settings and environment source; historical Habitat binary/library digests were not recorded.'),
        requests=requests, timing_rows=timing_rows, memory=memory, metrics=metrics,
        execution_episode_order=order, warmup_errors=warmup_errors,
        timing_start=timing_start, timing_end_exclusive=index, fixed_input=fixed_metrics,
        fixed_input_tensors=tensor_inputs, evidence_files_sha256=evidence.hashes,
        source_checks=evidence.source_checks)


def check_same_conditions(screen, remaining):
    require(screen['group'] == remaining['group'], 'Cannot combine different groups')
    for key in ('recipe','versions','model_weights_sha256','model_config_sha256','sources_sha256'):
        require(screen['launch'][key] == remaining['launch'][key], 'Changed execution condition; rerun20: launch.' + key)
    require(normalized_engine(screen['launch']['engine']) == normalized_engine(remaining['launch']['engine']),
            'Changed inference engine; rerun20')
    require(normalized_manifest(screen['manifest']) == normalized_manifest(remaining['manifest']), 'Changed task/prompt/generation; rerun20')
    require(screen['manifest']['dtype'].upper() == remaining['manifest']['dtype'].upper(), 'Changed precision label')
    require(screen['sources']['evaluator_sources_sha256'] == remaining['sources']['evaluator_sources_sha256'],
            'Changed evaluator/environment source; rerun20')
    before_helpers, after_helpers = (item['sources']['additional_sources_sha256'] for item in (screen, remaining))
    require(before_helpers['checks/navigation_precision_quality.py'] == after_helpers['checks/navigation_precision_quality.py'],
            'Changed output quality classification; rerun20')
    for source in set(before_helpers) & set(after_helpers):
        require(before_helpers[source] == after_helpers[source], 'Changed recorded environment helper: '+source)
    require(screen['provenance']['system_prompt_sha256'] == remaining['provenance']['system_prompt_sha256'], 'Changed system prompt')
    require(screen['habitat']['config'] == remaining['habitat']['config'], 'Changed Habitat execution settings; rerun20')
    require(screen['environment_evidence'] == remaining['environment_evidence'], 'Changed recorded execution environment')
    require(screen['execution'] == remaining['execution'], 'Changed actual precision/kernel launch table; rerun20')
    require(screen['fixed_input_tensors'] == remaining['fixed_input_tensors'], 'Changed actual fixed input tensors')
    require(screen['protocol']['navigation'] == remaining['protocol']['navigation'], 'Changed navigation timing/scheduling contract')


def verify_membership(screen_episodes, remaining_episodes, original):
    original_ids = dict(zip(map(str, original['episodes']), map(str, original['trajectory_ids'])))
    left = {e['episode_id']:e['trajectory_id'] for e in screen_episodes}
    right = {e['episode_id']:e['trajectory_id'] for e in remaining_episodes}
    require(len(left) == len(screen_episodes) == 20 and len(right) == len(remaining_episodes) == 80,
            'Expected unique20 and80 case identities')
    require(not (set(left)&set(right)) and dict(left,**right) == original_ids
            and len(original_ids) == len(original['episodes']) == len(original['trajectory_ids']) == 100,
            '20/80 are not an exact disjoint union of the original100 identities')
    by_id = {e['episode_id']:e for e in screen_episodes+remaining_episodes}
    return [by_id[str(eid)] for eid in original['episodes']]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--remaining-dir', type=Path, required=True)
    parser.add_argument('--screen-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    screen_dir, remaining_dir = args.screen_dir.resolve(), args.remaining_dir.resolve()
    require(screen_dir != remaining_dir, '20 and80 must be separate disjoint runs')
    output = args.output.resolve()
    require(output.suffix == '.json' and output.is_relative_to(ROOT/'reports')
            and not output.exists() and not output.with_suffix('.md').exists(), 'Choose fresh .json/.md paths inside reports/')
    accepted = read(args.screen_audit)
    require(accepted['audit_source_sha256'] == sha(Path(audit.__file__)), '20 audit implementation changed; regenerate the audit first')
    group = screen_dir.name
    accepted_row = next((r for r in accepted['table'] if r['group'] == group), None)
    require(accepted_row and accepted_row['passed'] is True and accepted_row['status'] == 'passed', 'This exact group has no passed20 acceptance')
    saved = accepted['groups'][group]
    require(saved['directory'] == relative(screen_dir) and saved['result_sha256'] == sha(screen_dir/'result.json'),
            'Passed20 audit refers to a different run/result')
    for source, expected in saved['evidence_files_sha256'].items():
        path = (ROOT/source).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha(path) == expected, 'Accepted20 evidence changed: '+source)
    roster = read(ROSTER/'manifest.json')
    evidence_hashes = {relative(args.screen_audit):sha(args.screen_audit),
                      relative(Path(__file__)):sha(Path(__file__)), relative(Path(audit.__file__)):sha(Path(audit.__file__)),
                      relative(ROSTER/'manifest.json'):sha(ROSTER/'manifest.json')}
    for source, expected in roster['sources_sha256'].items():
        require(sha(ROOT/source) == expected, 'Frozen roster source changed: '+source)
        evidence_hashes[source] = expected
    for source, expected in roster['artifact_files_sha256'].items():
        require(sha(ROSTER/source) == expected, 'Frozen roster artifact changed: '+source)
        evidence_hashes[relative(ROSTER/source)] = expected
    screen = audit_batch(screen_dir, ROSTER/'subset20.json', 20)
    remaining = audit_batch(remaining_dir, ROSTER/'remaining80.json', 80)
    check_same_conditions(screen, remaining)
    require(utc(screen['provenance']['started_utc']) < utc(accepted['created_utc'])
            <= utc(remaining['protocol']['created_utc']) <= utc(remaining['provenance']['started_utc']),
            'Remaining80 started before accepted20 audit or timestamp sequence is inconsistent')
    # Bind fresh recomputation to the accepted20 metric record.
    for key, value in saved['metrics'].items():
        equal_number(screen['metrics'][key], value, 'Accepted20 recomputation.'+key)
    original = read(ROSTER/'original100.json')
    episodes = verify_membership(screen['episodes'], remaining['episodes'], original)
    metrics = summarize(episodes, screen['requests']+remaining['requests'],
                        screen['timing_rows']+remaining['timing_rows'], screen['memory']+remaining['memory'])
    batches = []
    for item in (screen, remaining):
        evidence_hashes.update(item['evidence_files_sha256'])
        batches.append({key:item[key] for key in ('directory','result_sha256','execution_episode_order',
            'timing_start','timing_end_exclusive','warmup_errors','metrics','sources','source_checks')})
    report = dict(status='complete', group=group, created_utc=datetime.now(timezone.utc).isoformat(),
        accumulation='New accepted20 followed by unchanged remaining80; historical cancelled runs excluded.',
        screen_acceptance=dict(path=relative(args.screen_audit), sha256=sha(args.screen_audit),
                               created_utc=accepted['created_utc'], verdict=accepted_row),
        **metrics, episodes=episodes, batches=batches,
        execution_episode_order=screen['execution_episode_order']+remaining['execution_episode_order'],
        display_episode_order=list(map(str,original['episodes'])),
        runtime_contract=dict(launch=screen['launch'], execution=screen['execution'],
            task_manifest=normalized_manifest(screen['manifest']), habitat=screen['habitat']['config'],
            environment_evidence=screen['environment_evidence'],
            system_prompt_sha256=screen['provenance']['system_prompt_sha256']),
        same_conditions_sha256=json_sha(dict(launch=screen['launch'], execution=screen['execution'],
            task=normalized_manifest(screen['manifest']), habitat=screen['habitat']['config'])),
        quality_failures_retained_in_denominator=True,
        passed_stability=not metrics['error_count'] and not screen['warmup_errors'] and not remaining['warmup_errors'],
        excluded_requests=dict(navigation_warmups=4, fixed_input_and_tail_requests='All excluded; retained in each batch evidence.'),
        evidence_files_sha256=evidence_hashes,
        limitations=['100 fixed development cases, not independent generalization or statistical equivalence.',
            'Execution order is selected20 then remaining80; display order is original100.',
            'HTTP P50/P95 are recomputed from every real request; queued zeroes, warmups and tails are excluded.',
            'Decode denominator is actual forwards including work after HTTP completion; CUDA intervals include dispatch/idle.',
            'Memory is the maximum of all200ms whole-card samples including desktop and prewarm; not a model-only instantaneous peak.',
            'Evidence SHA inventory covers every artifact used by this audit; original PNG frame files are not independently rehashed.',
            'Environment equivalence is checked against recorded versions/configuration/source; historical Habitat interpreter/library digests were not recorded and cannot be reconstructed.',
            'Different controller SHA is permitted only as orchestration provenance; runtime/evaluator/source/selected-launch changes reject accumulation.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    markdown = '\n'.join([
        '# 新修复配置同100条累计复核', '',
        f"配置 `{group}`：通过验收的新20条＋相同冻结条件的剩余80条；全部100条已逐步重算。", '',
        '| 配置 | 成功/100 | SR | SPL | 真实请求 | HTTP P50/s | HTTP P95/s | 整卡峰值/MiB | 质量错误 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        f"| {group} | {metrics['success_count']} | {metrics['sr']:.3f} | {metrics['spl']:.4f} | {metrics['actual_model_calls']} | {metrics['request_median_s']:.3f} | {metrics['request_p95_s']:.3f} | {metrics['sampled_whole_card_peak_mib']:.0f} | {metrics['error_count']} |", '',
        f"输入token {metrics['prompt_tokens']}；返回token {metrics['completion_tokens']}；实际prefill {metrics['actual_prefill_forwards']}次、decode {metrics['actual_decode_forwards']}次。", '',
        f"20条来源：`{screen['directory']}`；80条来源：`{remaining['directory']}`。执行顺序为20后80，完整案例顺序、配置、审计记录和全部已用证据SHA保存在[{output.name}]({output.name})。", '',
        'P50/P95直接合并全部真实HTTP请求重算，未平均两批分位。排队动作、4次导航预热、固定输入测量和尾请求均排除。质量失败保留在100分母。显存为200ms整卡采样峰值。', '',
        '该结果仅代表这一组的同100条复测；不能据此宣称八组全部完成，也不能把20条验收当作统计等价证明。', ''])
    with output.with_suffix('.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(markdown)
    print(json.dumps(dict(output=relative(output), group=group, **metrics), ensure_ascii=False))


if __name__ == '__main__':
    main()
