"""Independent, CPU-only audit and pre-registered gates for repair same20.

Example: python -B checks/audit_navigation_repair.py --directories
logs/navigation_repair_20261112/screen_fp16_v1/fp16 --output reports/repair_audit.json
The supplied directories contain result.json, service/, and navigation/.
Unrun groups stay pending. Pipeline completion alone never grants acceptance.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / 'data/baseline/navigation_repair_20261112'
GROUPS = ['fp16'] + [a + '_' + p for a in ('awq', 'gptq') for p in ('w4a16', 'w4a4', 'w4a8', 'w8a16')]
QUALITY_ERRORS = {'truncated_model_output', 'invalid_action_random_fallback_rejected'}
CELLS = [(s, t) for s in (0, 1) for t in (18, 64)]
HASH_CACHE = {}
TENSOR_HASH_CACHE = {}
ACTIVATION_RECIPES = {
    4: 'dynamic_per_token_per_group128_symmetric_maxabs_narrow_range',
    8: 'dynamic_per_token_symmetric_maxabs_narrow_range', 16: 'none'}
FUSED = {'qkv': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
         'o': ['self_attn.o_proj'], 'gate_up': ['mlp.gate_proj', 'mlp.up_proj'], 'down': ['mlp.down_proj']}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(block)
        HASH_CACHE[key] = value.hexdigest()
    return HASH_CACHE[key]


def require(value, message):
    if not value:
        raise ValueError(message)


def equal_number(actual, expected, label):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), label + ': not a number')
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8),
            f'{label}: {actual!r} != independently computed {expected!r}')


def p95(values):
    ordered = sorted(values)
    index = (len(ordered) - 1) * .95
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def normalize(value):
    return str(value).replace('\\', '/')


class Evidence:
    def __init__(self, directory):
        self.directory = directory
        self.hashes = {}
        self.failures = []
        self.source_checks = []
        self.snapshots = {}
        for folder in (directory.parent, directory):
            path = folder / 'source_snapshot.json'
            if path.exists():
                record = self.load(path)
                for source, item in record['files'].items():
                    self.snapshots[normalize(source)] = item

    def load(self, path):
        self.hashes[relative(path)] = sha(path)
        return read(path)

    def source(self, source, expected):
        source = normalize(source)
        current = ROOT / source
        actual = sha(current) if current.is_file() else None
        chosen = source
        if actual != expected and source in self.snapshots:
            item = self.snapshots[source]
            snapshot = ROOT / normalize(item['snapshot'])
            require(snapshot.resolve().is_relative_to(ROOT), 'Snapshot escapes project')
            actual = sha(snapshot) if snapshot.is_file() else None
            chosen = relative(snapshot)
            if item['sha256'] != expected:
                actual = None
        passed = actual == expected
        self.source_checks.append(dict(source=source, expected_sha256=expected,
            evidence_path=chosen, actual_sha256=actual, passed=passed))
        if passed:
            self.hashes[chosen] = actual
        else:
            self.failures.append('Missing/mismatching frozen source: ' + source)


def verify_record(row, label):
    require(type(row.get('model_called')) is bool, label + ': missing actual-call flag')
    error = row.get('error')
    require(error is None or error in QUALITY_ERRORS, label + ': infrastructure/unclassified error ' + str(error))
    if error:
        require(row['model_called'] and row['action'] == 'STOP', label + ': error must be a real-call failed STOP')
    if row['model_called']:
        require(isinstance(row.get('raw'), str), label + ': missing raw output')
        require(row.get('finish_reason') in ('stop', 'length'), label + ': unexpected finish_reason')
        usage = row.get('usage')
        require(isinstance(usage, dict), label + ': missing usage')
        for key, maximum in [('prompt_tokens', 3072), ('completion_tokens', 512)]:
            require(type(usage.get(key)) is int and 0 < usage[key] <= maximum, label + ': invalid ' + key)
        require(2 <= row.get('image_count', 0) <= 9, label + ': image count outside frozen budget')
        latency = row['model_request_s']
        require(math.isfinite(latency) and latency > 0, label + ': nonpositive/nonfinite actual latency')
        if row['finish_reason'] == 'length':
            require(error == 'truncated_model_output', label + ': length output not classified as truncation')
        if error == 'truncated_model_output':
            require(row['finish_reason'] == 'length', label + ': false truncation classification')
    else:
        require(not error and row['model_request_s'] == 0 and row.get('usage') is None,
                label + ': queued action has request data/error')
    return error


def timing(evidence, service, index, usage, method):
    value = evidence.load(service / f'timing_{index:03d}.json')
    require(value['index'] == index and value['group'] == method, 'Timing identity mismatch')
    require(value['graph_requested'] == 'on' and value['capture_replay_observed_outside_graph'],
            'Missing actual graph timing evidence')
    steps = value['steps']
    require(len(steps) >= usage['completion_tokens'] and len(steps) > 1, 'Missing actual forwards')
    prefill, decode = steps[0], steps[1:]
    require(prefill['phase'] == 'prefill' and prefill['execution_mode'] == 'eager'
            and prefill['rows'] == usage['prompt_tokens'] <= 3072, 'Prefill/request association mismatch')
    require(all(r['phase'] == 'decode' and r['rows'] == 1 and r['execution_mode'] == 'replay' for r in decode),
            'Decode not entirely real graph replay')
    require(value['logical_forward_counts'] == dict(Counter(r['phase'] for r in steps)), 'Forward counts disagree')
    require(value['actual_execution_modes'] == dict(Counter(r['execution_mode'] for r in steps)), 'Execution counts disagree')
    for row in steps:
        for item in row['timers'].values():
            require(all(math.isfinite(item[k]) and item[k] >= 0 for k in ('cuda_interval_ms', 'host_wall_ms')),
                    'Invalid CUDA/host interval')
    return dict(index=index, prefill_language_ms=prefill['timers']['language']['cuda_interval_ms'],
        prefill_vision_ms=prefill['timers']['vision']['cuda_interval_ms'],
        prefill_model_ms=prefill['timers']['model']['cuda_interval_ms'],
        decode_forwards=len(decode), output_tokens=usage['completion_tokens'],
        decode_model_total_ms=sum(r['timers']['model']['cuda_interval_ms'] for r in decode))


def stages(rows):
    require(rows, 'No stage samples')
    prefill = [r['prefill_language_ms'] for r in rows]
    decode = [r['decode_model_total_ms'] / r['decode_forwards'] for r in rows]
    count = sum(r['decode_forwards'] for r in rows)
    return dict(requests=len(rows), prefill_language_p50_ms=statistics.median(prefill),
        prefill_language_p95_ms=p95(prefill), actual_decode_forwards=count,
        decode_ms_per_actual_forward=sum(r['decode_model_total_ms'] for r in rows) / count,
        decode_request_mean_p50_ms=statistics.median(decode), decode_request_mean_p95_ms=p95(decode))


def check_numeric_map(saved, calculated, label):
    for name, value in calculated.items():
        equal_number(saved[name], value, label + '.' + name)


def controller_source(protocol, evidence):
    if protocol.get('controller_source'):
        return normalize(protocol['controller_source'])
    candidates = ['checks/run_navigation_repair.py', 'checks/run_navigation_repair_checked.py']
    matches = []
    for candidate in candidates:
        snapshot = evidence.snapshots.get(candidate)
        if snapshot and snapshot['sha256'] == protocol['controller_sha256']:
            matches.append(candidate)
        elif (ROOT / candidate).exists() and sha(ROOT / candidate) == protocol['controller_sha256']:
            matches.append(candidate)
    require(len(matches) == 1, 'Cannot resolve the frozen controller identity')
    return matches[0]


def audit_fixed(evidence, directory, result, launch):
    service = directory / 'service'
    fixed = evidence.load(directory / 'fixed_requests.json')
    require([r['index'] for r in fixed] == list(range(len(fixed))), 'Fixed request index gap')
    require([(r['sample'], r['phase']) for r in fixed[:2]] == [(0, 'smoke'), (1, 'smoke')], 'First2 are not explicit SHA smokes')
    fixed_stages = {}
    for row in fixed:
        response = row['response']
        require(response['model'] == launch['model_name'], 'Fixed response model mismatch')
        require(response['usage']['prompt_tokens'] == (701 if row['sample'] == 0 else 2815), 'Fixed input token count changed')
        require(math.isfinite(row['http_total_s']) and row['http_total_s'] > 0, 'Invalid fixed HTTP latency')
        choice = response['choices'][0]
        if row['phase'] == 'smoke':
            require(choice['finish_reason'] == 'stop', 'Natural EOS smoke failed')
        else:
            require(row['phase'] in ('warmup', 'measured') and row['tokens'] in (18, 64), 'Unexpected fixed protocol cell')
            require(response['usage']['completion_tokens'] == row['tokens'] and choice['finish_reason'] == 'length',
                    'Fixed output budget not actually reached')
        fixed_stages[row['index']] = timing(evidence, service, row['index'], response['usage'], launch['recipe']['method'])
    metrics = {}
    for sample, tokens in CELLS:
        cell = [r for r in fixed if r['sample'] == sample and r.get('tokens') == tokens]
        measured = [r for r in cell if r['phase'] == 'measured']
        require(sum(r['phase'] == 'warmup' for r in cell) == 2 and len(measured) >= 10, 'Insufficient fixed repetitions')
        first_measured = min(r['index'] for r in measured)
        require(all(r['index'] < first_measured for r in cell if r['phase'] == 'warmup'), 'Warmup follows measurement')
        values = [r['http_total_s'] for r in measured]
        calculated = dict(n=len(measured), http_p50_s=statistics.median(values), http_p95_s=p95(values))
        saved = result['fixed_input'][f'{sample}_{tokens}']
        check_numeric_map(saved, calculated, 'fixed_' + str((sample, tokens)))
        calculated['stage_summary'] = stages([fixed_stages[r['index']] for r in measured])
        check_numeric_map(saved['stage_summary'], calculated['stage_summary'], 'fixed_stage')
        metrics[f'{sample}_{tokens}'] = calculated
    tensor_inputs = {}
    require(len(list(service.glob('input_*.json'))) == 2, 'Formal variable-length input SHA work detected')
    for sample, tokens in ((0, 701), (1, 2815)):
        item = evidence.load(service / f'input_{tokens}.json')
        require(item['request_index'] == sample and item['input_ids']['shape'] == [1, tokens], 'Fixed tensor SHA association mismatch')
        tensor_inputs[str(sample)] = {key: item[key] for key in ('input_ids', 'pixel_values')}
    if (directory / 'fixed_result.json').exists():
        require(evidence.load(directory / 'fixed_result.json') == metrics, 'Separate fixed_result differs from recomputation')
    gate = evidence.load(directory / 'fixed_speed_gate.json') if (directory / 'fixed_speed_gate.json').exists() else None
    return fixed, metrics, tensor_inputs, gate


def audit_fixed_only(directory, evidence, result, launch, expected_bits):
    require(result['case_count'] == 0 and result['success_count'] is None, 'Fixed-only result must not claim navigation cases')
    require(not (directory / 'navigation').exists(), 'Fixed-only group nevertheless started navigation')
    protocol = evidence.load(directory.parent / 'protocol.json')
    require(protocol['manifest_sha256'] == sha(ROSTER / 'subset20.json'), 'Speed failure used a different20 source')
    original = evidence.load(ROSTER / 'manifest.json')
    for name, expected in original['sources_sha256'].items():
        evidence.source(name, expected)
    for name, expected in original['artifact_files_sha256'].items():
        evidence.source(relative(ROSTER / name), expected)
    contract = evidence.load(ROOT / 'setup/navigation-stage2-execution-20261109.json')
    for name in ('checks/simplevln_policy.py', 'checks/simplevln_env.py', 'checks/run_closed_loop_baseline.py',
                 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'):
        evidence.source(name, contract['reference_files_sha256'][name])
    for key, source in [('controller_sha256', controller_source(protocol, evidence)),
                        ('navigation_policy_sha256', 'checks/simplevln_policy.py'),
                        ('evaluator_sha256', 'checks/run_navigation_precision_diagnostic.py')]:
        evidence.source(source, protocol[key])
    if protocol.get('source_snapshot_sha256'):
        require(protocol['source_snapshot_sha256'] == sha(directory.parent / 'source_snapshot.json'), 'Snapshot manifest hash mismatch')
    for source, snapshot in evidence.snapshots.items():
        evidence.source(source, snapshot['sha256'])
    fixed, metrics, inputs, gate = audit_fixed(evidence, directory, result, launch)
    if result['status'] == 'fixed_speed_gate_failed':
        require(gate is not None and gate == result['gate'] and gate['passed'] is False and gate['max_ratio'] == 1.10,
                'Missing retained failed speed gate')
    else:
        require(result['status'] == 'fixed_input_only_complete', 'Unexpected fixed-only status')
    tail = evidence.load(directory / 'fixed_flush_request.json')
    require(tail['response']['model'] == launch['model_name'] and tail['response']['usage']['completion_tokens'] == 18,
            'Missing fixed-input tail refresh')
    files = list((directory / 'service').glob('timing_*.json'))
    require(sorted(int(p.stem.split('_')[1]) for p in files) == list(range(len(fixed))),
            'Fixed-only timing count differs; final unflushed tail must be excluded')
    return dict(group=result['group'], directory=relative(directory), pipeline_status=result['status'],
        integrity_passed=not evidence.failures, integrity_failures=evidence.failures,
        metrics=None, case_count=0, fixed_input=metrics, input_tensors=inputs, saved_speed_gate=gate,
        result_sha256=sha(directory / 'result.json'),
        fixed_input_plan=[(r['sample'], r['phase'], r['tokens']) for r in fixed],
        actual_recipe=launch['recipe'], precision_metadata_consistent=None,
        actual_execution_receipt=evidence.load(directory / 'service' / 'repair_execution_audit.json'),
        service_sources_sha256=launch['sources_sha256'], model_config_sha256=launch['model_config_sha256'],
        model_weights_sha256=launch['model_weights_sha256'],
        timing_files=len(files), final_tail_request_excluded=True, source_checks=evidence.source_checks,
        evidence_files_sha256=evidence.hashes)


def audit_group(directory):
    evidence = Evidence(directory)
    run = directory.parent
    result = evidence.load(directory / 'result.json')
    group = result['group']
    require(group in GROUPS, 'Unknown group: ' + group)
    service, folder = directory / 'service', directory / 'navigation'
    launch = evidence.load(service / 'launch.json')
    recipe = launch['recipe']
    expected_bits = (16, 16) if group == 'fp16' else tuple(map(int, group.split('_w')[1].split('a')))
    require((recipe['weight_bits'], recipe['activation_bits']) == expected_bits, 'Launch precision/group mismatch')
    require(launch['group'] == group and launch['model_name'] == 'navigation-repair-' + group, 'Launch group mismatch')
    require(launch['versions'] == dict(lmdeploy='0.11.1', torch='2.5.1+cu121', transformers='4.57.6', triton='3.1.0'),
            'Inference library versions changed')
    engine = launch['engine']
    for key, expected in dict(dtype='float16', tp=1, dp=1, session_len=4096, max_batch_size=1,
        max_prefill_token_num=3072, num_gpu_blocks=64, block_size=64, eager_mode=False,
        enable_prefix_caching=False, enable_microbatch=False, quant_policy=0).items():
        require(engine[key] == expected, 'Engine condition changed: ' + key)
    for name, expected in launch['sources_sha256'].items():
        evidence.source('checks/' + name, expected)
    require(result['source_hashes'] == launch['sources_sha256'], 'Result source hashes differ from launch')
    model = ROOT / recipe['model']
    evidence.source(relative(model / 'config.json'), launch['model_config_sha256'])
    evidence.source(relative(model / 'model.safetensors'), launch['model_weights_sha256'])
    contract = evidence.load(ROOT / 'setup/navigation-stage2-execution-20261109.json')
    for name in ('chat_template.jinja', 'generation_config.json', 'preprocessor_config.json',
                 'processor_config.json', 'tokenizer.json', 'tokenizer_config.json'):
        expected = contract['reference_files_sha256']['models/SimpleVLN_Qwen3VL_4B_RL_lmdeploy/' + name]
        evidence.source(relative(model / name), expected)
    if group == 'fp16':
        require(launch['model_weights_sha256'] == 'b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b',
                'FP16 source identity changed')
    else:
        require(evidence.load(model / 'conversion_audit.json')['sha256'] == launch['model_weights_sha256'],
                'Quantized checkpoint conversion identity changed')
    execution = evidence.load(service / 'execution_audit.json')
    repair = evidence.load(service / 'repair_execution_audit.json')
    require(execution['first_native_forward_verified'] and execution['native_forward_preserved'], 'Missing native execution proof')
    require((repair['weight_bits'], repair['activation_bits']) == expected_bits and repair['native_forward'],
            'Repair precision execution mismatch')
    if group != 'fp16':
        receipt = execution['native_awq_receipt']
        require(receipt['verified'] and receipt['module_count'] == 144
                and receipt['loaded_quantized_tensor_count'] == 756, 'Missing actual packed projections')
        if expected_bits != (4, 16):
            calls = repair['actual_extension_calls']
            require(repair['extension'] and len(calls) == 144 and set(calls.values()) == {1},
                    'Missing actual precision-extension execution')
            initializations = list(service.glob('runtime_init_*.json'))
            require(len(initializations) == 1, 'Expected one precision runtime initialization')
            runtime = evidence.load(initializations[0])
            wb, ab = expected_bits
            require(runtime['runtime_class'] == 'PrecisionRepairModel' and runtime['activation_recipe'] == ACTIVATION_RECIPES[ab]
                    and runtime['kernel_variant'] == 'balanced' and runtime['stock_backend'] is False,
                    'Missing explicit new precision recipe/runtime')
            require(recipe['activation_recipe'] == ACTIVATION_RECIPES[ab] and recipe['kernel_variant'] == 'balanced'
                    and recipe['compute'] == ('int8_mma' if ab < 16 else 'fp16_mma')
                    and recipe['activation_storage_bits'] == ab and recipe['native_int4_mma'] is False,
                    'Service recipe differs from the actual packed activation computation')
            require(runtime['activation_storage_bits'] == ab and runtime['weight_bits'] == wb
                    and runtime['activation_bits'] == ab and runtime['intermediate_dtype'] == 'float16',
                    'Runtime precision storage mismatch')
            if ab < 16:
                require(runtime['integer_compute_bits'] == 8, 'Low-bit activations did not declare INT8 MMA')
            frozen = repair['frozen_launch_table']
            require(frozen['frozen'] and frozen['prepared_projection_count'] == 4
                    and frozen['private_cold_compile_guard_installed'] and frozen['cache_unchanged_since_freeze']
                    and frozen['allowed_dynamic_rows'] == [1, 3072], 'Extension prepared/frozen execution proof missing')
            require(len(frozen['prepared_projections']) == 4 and all(
                (p['weight_bits'], p['activation_bits'], p['variant']) == (wb, ab, 'balanced')
                and p['prepared_rows'] == [1, 16, 17, 701, 2815, 2816]
                for p in frozen['prepared_projections']), 'Extension launch geometry/variant mismatch')
        else:
            frozen = repair['frozen_launch_table']
            require(not repair['extension'] and frozen['frozen'] and frozen['actual_m_preserved']
                    and not frozen['runtime_autotuning_allowed'] and not frozen['runtime_cold_compilation_allowed']
                    and frozen['loaded_module_count'] == 144 and len(frozen['shapes']) == 4
                    and frozen['generic_m_specialization_disabled'], 'W4A16 actual frozen launch table incomplete')
    require(evidence.load(service / 'startup_graph_capture.json')['actual_capture_observed'], 'Missing graph capture')
    if result['status'] in ('fixed_speed_gate_failed', 'fixed_input_only_complete'):
        return audit_fixed_only(directory, evidence, result, launch, expected_bits)
    manifest = evidence.load(folder / 'manifest.json')
    case_manifest = evidence.load(directory / 'case_manifest.json')
    require(manifest == case_manifest, 'Evaluator manifest differs from launched manifest')
    summary = evidence.load(folder / 'summary.json')
    protocol = evidence.load(run / 'protocol.json')
    original = evidence.load(ROSTER / 'manifest.json')
    base_manifest = evidence.load(ROSTER / 'subset20.json')
    for name, expected in original['sources_sha256'].items():
        evidence.source(name, expected)
    for name, expected in original['artifact_files_sha256'].items():
        evidence.source(relative(ROSTER / name), expected)
    original_contract = evidence.load(ROOT / 'setup/navigation-stage2-execution-20261109.json')
    for name in ('checks/simplevln_policy.py', 'checks/simplevln_env.py', 'checks/run_closed_loop_baseline.py',
                 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'):
        evidence.source(name, original_contract['reference_files_sha256'][name])
    require(protocol['manifest_sha256'] == sha(ROSTER / 'subset20.json'), 'Run did not use the frozen20 source manifest')
    require(manifest['episodes'] == base_manifest['episodes'] == original['selected20']
            and manifest['trajectory_ids'] == base_manifest['trajectory_ids'], 'Different20 membership/order')
    require(len(manifest['episodes']) == len(set(manifest['episodes'])) == 20, 'Expected20 unique cases')
    for key in ('temperature', 'top_p', 'max_tokens', 'seed', 'max_episode_steps', 'success_distance_m',
                'model_inputs', 'output_actions', 'action_parser', 'dataset_path', 'split', 'author_policy_sha256'):
        require(manifest[key] == base_manifest[key], 'Frozen navigation setting changed: ' + key)
    require(manifest['model'] == summary['model'] == launch['model_name'], 'Navigation model identity mismatch')
    expected_dtype = 'FP16' if expected_bits == (16, 16) else 'W%dA%d' % expected_bits
    if manifest['dtype'].upper() != expected_dtype:
        evidence.failures.append('Manifest dtype=' + manifest['dtype'] + ' disagrees with actual launch ' + expected_dtype)
    provenance = evidence.load(folder / 'provenance.json')
    evidence.hashes[relative(folder / 'habitat_config.yaml')] = sha(folder / 'habitat_config.yaml')
    require(provenance['manifest_sha256'] == sha(directory / 'case_manifest.json'), 'Manifest provenance changed')
    require(provenance['system_prompt_sha256'] == sha(folder / 'system_prompt.txt'), 'System prompt hash mismatch')
    require((folder / 'system_prompt.txt').read_text() == 'You are a visual language navigation model.', 'Original system prompt changed')
    for name, expected in provenance['scripts'].items():
        evidence.source('checks/' + name, expected)
    for key, source in [('controller_sha256', controller_source(protocol, evidence)),
                        ('navigation_policy_sha256', 'checks/simplevln_policy.py'),
                        ('evaluator_sha256', 'checks/run_navigation_precision_diagnostic.py')]:
        evidence.source(source, protocol[key])
    if protocol.get('source_snapshot_sha256'):
        require(protocol['source_snapshot_sha256'] == sha(run / 'source_snapshot.json'), 'Snapshot manifest hash mismatch')
    evidence.source('tmp/habitat_setup/navigation_source_review/eval/vln/eval.py', manifest['author_policy_sha256'])
    for name in ('simplevln_env.py', 'simplevln_policy.py', 'run_navigation_precision_diagnostic.py', 'run_closed_loop_baseline.py'):
        require(name in provenance['scripts'], 'Missing evaluator source provenance: ' + name)
    for name in ('navigation_precision_quality.py', 'check_navigation_awq_dart_request.py'):
        source = 'checks/' + name
        if source in evidence.snapshots:
            evidence.source(source, evidence.snapshots[source]['sha256'])
        else:
            evidence.failures.append('Missing frozen auxiliary source: ' + source)
    fixed, fixed_metrics, tensor_inputs, saved_gate = audit_fixed(evidence, directory, result, launch)
    require(summary['status'] == result['status'] == 'complete' and summary['episode_count'] == 20, 'Incomplete navigation cohort')
    require(summary['warmup_requests_excluded'] == 2, 'Navigation warmup convention changed')
    warmups = evidence.load(folder / 'warmups.json')
    require(len(warmups) == 2, 'Missing navigation warmups')
    index = len(fixed)
    if (directory / 'fixed_flush_request.json').exists():
        fixed_tail = evidence.load(directory / 'fixed_flush_request.json')
        require(fixed_tail['response']['model'] == launch['model_name']
                and fixed_tail['response']['usage']['completion_tokens'] == 18, 'Invalid excluded fixed tail')
        timing(evidence, service, index, fixed_tail['response']['usage'], recipe['method'])
        index += 1
    navigation_start = index
    warmup_errors = []
    for row in warmups:
        error = verify_record(row, 'warmup')
        require(row['model_called'], 'Warmup was queued')
        if error:
            warmup_errors.append(error)
        timing(evidence, service, index, row['usage'], recipe['method'])
        index += 1
    membership = dict(zip(manifest['episodes'], manifest['trajectory_ids']))
    observed_ids = [str(ep['episode_id']) for ep in summary['episodes']]
    require(len(observed_ids) == len(set(observed_ids)) == 20 and set(observed_ids) == set(membership), 'Summary membership mismatch')
    episodes, requests, navigation_stages = [], [], []
    first_inputs = {}
    errors = Counter()
    for ep in summary['episodes']:
        eid = str(ep['episode_id'])
        require(str(ep['trajectory_id']) == str(membership[eid]), 'Trajectory identity changed')
        require(evidence.load(folder / ('episode_' + eid) / 'result.json') == ep, 'Episode result differs from summary')
        path = folder / ('episode_' + eid) / 'steps.jsonl'
        evidence.hashes[relative(path)] = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require(0 < len(rows) == ep['steps'] <= 400, 'Invalid episode step count')
        require(rows[0]['model_called'], 'First episode action was not a real request')
        first_inputs[eid] = dict(request_text=rows[0]['request_text'],
            request_image_sha256=rows[0]['request_image_sha256'], image_count=rows[0]['image_count'],
            prompt_tokens=rows[0]['usage']['prompt_tokens'])
        path_length = 0.
        for number, row in enumerate(rows):
            require(row['step'] == number and str(row['episode_id']) == eid, 'Step index/identity mismatch')
            error = verify_record(row, eid + ':' + str(number))
            if error:
                errors[error] += 1
                require(number == len(rows) - 1 and ep['terminal_stop_source'] == 'error_fallback', 'Quality error did not terminate case')
            before, after = row['before']['position'], row['after']['position']
            if number:
                require(all(math.isclose(a, b, abs_tol=1e-7) for a, b in zip(before, rows[number - 1]['after']['position'])),
                        'Noncontiguous saved trajectory')
            path_length += math.sqrt(sum((a - b) ** 2 for a, b in zip(before, after)))
            if row['model_called']:
                requests.append(row)
                navigation_stages.append(timing(evidence, service, index, row['usage'], recipe['method']))
                index += 1
        equal_number(ep['path_length_m'], path_length, 'Saved trajectory length')
        require(ep['metrics'] == rows[-1]['metrics'], 'Final metrics differ from last step')
        last = rows[-1]
        success = int(last['action'] == 'STOP' and ep['terminal_stop_source'] == 'model'
                      and not last.get('error') and last['metrics']['distance_to_goal'] < 3.)
        shortest = ep['start_distance_m']
        require(math.isfinite(shortest) and shortest > 0, 'Invalid starting geodesic distance')
        spl = success * shortest / max(shortest, path_length)
        # Habitat accumulates its internal distance in FP32. Preserve the
        # original evaluator's 1e-5 consistency tolerance only for this check;
        # saved-pose recomputation and reported result comparisons stay strict.
        habitat_spl = ep['metrics']['spl']
        habitat_expected = ep['metrics']['success'] * shortest / max(shortest, path_length)
        require(math.isfinite(habitat_spl) and math.isclose(habitat_spl, habitat_expected,
                rel_tol=1e-9, abs_tol=1e-5), 'Habitat SPL consistency exceeds original1e-5 tolerance')
        calculated = dict(episode_id=eid, trajectory_id=str(membership[eid]), success=success, spl=spl,
            steps=len(rows), actual_model_calls=sum(r['model_called'] for r in rows),
            errors=sum(bool(r.get('error')) for r in rows),
            collisions=sum(bool(r['metrics']['collisions']['is_collision']) for r in rows),
            terminal_stop_source=ep['terminal_stop_source'])
        saved = next(r for r in result['episodes'] if str(r['episode_id']) == eid)
        for key, value in calculated.items():
            if isinstance(value, (int, float)):
                equal_number(saved[key], value, 'Episode ' + eid + '.' + key)
            else:
                require(saved[key] == value, 'Episode metadata mismatch: ' + key)
        episodes.append(calculated)
        require(result['step_files_sha256'][relative(path)] == sha(path), 'Saved step SHA mismatch')
    require(index == len(list(service.glob('timing_*.json'))), 'Unexpected timing file count; only final unflushed tail may be absent')
    require(sorted(int(p.stem.split('_')[1]) for p in service.glob('timing_*.json')) == list(range(index)), 'Timing index gap/extra file')
    require(result['timing_start'] == navigation_start and result['timing_end_exclusive'] == index, 'Result timing bounds mismatch')
    flush = evidence.load(directory / 'flush_request.json')
    require(flush['response']['model'] == launch['model_name'] and flush['response']['usage']['completion_tokens'] == 18,
            'Missing independent tail refresh')
    memory_path = folder / 'gpu_samples.csv'
    evidence.hashes[relative(memory_path)] = sha(memory_path)
    memory = [float(line.split(',')[1]) for line in memory_path.read_text().splitlines() if len(line.split(',')) >= 3]
    require(memory and all(math.isfinite(v) and v > 0 for v in memory), 'Invalid GPU memory samples')
    latencies = [r['model_request_s'] for r in requests]
    metrics = dict(case_count=20, success_count=sum(r['success'] for r in episodes),
        sr=sum(r['success'] for r in episodes) / 20, spl=sum(r['spl'] for r in episodes) / 20,
        actual_model_calls=len(requests), completion_tokens=sum(r['usage']['completion_tokens'] for r in requests),
        prompt_tokens=sum(r['usage']['prompt_tokens'] for r in requests), request_median_s=statistics.median(latencies),
        request_p95_s=p95(latencies), request_total_s=sum(latencies), sampled_whole_card_peak_mib=max(memory),
        error_count=sum(errors.values()), infrastructure_errors=0)
    check_numeric_map(result, metrics, 'Navigation result')
    require(result['quality_errors'] == dict(errors) and result['warmup_errors'] == warmup_errors, 'Error counts differ')
    require(summary['error_count'] == sum(errors.values()), 'Evaluator error count differs')
    require(result['manifest_sha256'] == sha(folder / 'manifest.json'), 'Result manifest SHA differs')
    stage_metrics = stages(navigation_stages)
    check_numeric_map(result['stage_summary'], stage_metrics, 'Navigation stages')
    stability = not errors and not warmup_errors
    require(result['passed_stability'] == stability, 'Saved stability label differs')
    return dict(group=group, directory=relative(directory), pipeline_status='complete',
        integrity_passed=not evidence.failures, integrity_failures=evidence.failures,
        stability_passed=stability, metrics=metrics, quality_errors=dict(errors), warmup_errors=warmup_errors,
        episodes=episodes, execution_episode_order=observed_ids, stage_summary=stage_metrics,
        episode_first_input_signatures=first_inputs,
        fixed_input=fixed_metrics, fixed_input_plan=[(r['sample'], r['phase'], r['tokens']) for r in fixed],
        saved_speed_gate=saved_gate, result_sha256=sha(directory / 'result.json'),
        input_tensors=tensor_inputs, actual_recipe=recipe, manifest_dtype=manifest['dtype'],
        actual_execution_receipt=repair,
        service_sources_sha256=launch['sources_sha256'], model_config_sha256=launch['model_config_sha256'],
        model_weights_sha256=launch['model_weights_sha256'],
        precision_metadata_consistent=manifest['dtype'].upper() == expected_dtype,
        timing_files=index, final_tail_request_excluded=True, source_checks=evidence.source_checks,
        evidence_files_sha256=evidence.hashes)


def combine(audits):
    baseline = audits.get('fp16')
    baseline_ready = bool(baseline and baseline.get('integrity_passed') and baseline.get('stability_passed'))
    table = []
    for group in GROUPS:
        item = audits.get(group)
        if item is None:
            table.append(dict(group=group, status='pending', passed=False, reason='Not run in this audit'))
            continue
        if not item.get('metrics') and not item.get('fixed_input'):
            table.append(dict(group=group, status=item.get('pipeline_status', 'audit_failed'), passed=False,
                              reason=item.get('error', 'No completed result')))
            continue
        quality = speed = same20 = tensor_match = first_inputs_match = None
        speed_cells = {}
        if baseline and baseline.get('metrics'):
            tensor_match = item['input_tensors'] == baseline['input_tensors']
            plan_match = item['fixed_input_plan'] == baseline['fixed_input_plan']
            if item.get('metrics'):
                same20 = ([(r['episode_id'], r['trajectory_id']) for r in item['episodes']] ==
                          [(r['episode_id'], r['trajectory_id']) for r in baseline['episodes']])
                first_inputs_match = item['episode_first_input_signatures'] == baseline['episode_first_input_signatures']
                quality = (item['metrics']['success_count'] >= baseline['metrics']['success_count'] - 2
                           and item['metrics']['spl'] >= baseline['metrics']['spl'] - .10 - 1e-12)
            for cell, values in item['fixed_input'].items():
                ref = baseline['fixed_input'][cell]
                p50ratio, p95ratio = values['http_p50_s'] / ref['http_p50_s'], values['http_p95_s'] / ref['http_p95_s']
                speed_cells[cell] = dict(http_p50_ratio_to_fp16=p50ratio, http_p95_ratio_to_fp16=p95ratio,
                                        passed=p50ratio <= 1.10 and p95ratio <= 1.10)
            speed = plan_match and all(r['passed'] for r in speed_cells.values())
            gate = item.get('saved_speed_gate')
            if gate:
                try:
                    require(gate['baseline_sha256'] == baseline['result_sha256'], 'Saved gate used a different baseline result')
                    require(gate['max_ratio'] == 1.10, 'Saved speed threshold changed')
                    for cell, values in speed_cells.items():
                        equal_number(gate['ratios'][cell]['http_p50_s'], values['http_p50_ratio_to_fp16'], 'Saved P50 ratio')
                        equal_number(gate['ratios'][cell]['http_p95_s'], values['http_p95_ratio_to_fp16'], 'Saved P95 ratio')
                    require(gate['passed'] == all(r['passed'] for r in speed_cells.values()), 'Saved speed verdict differs')
                except Exception as error:
                    item['integrity_passed'] = False
                    item['integrity_failures'].append(str(error))
        if not item.get('metrics'):
            table.append(dict(group=group, status=item['pipeline_status'], passed=False, case_count=0,
                success_count=None, sr=None, spl=None, baseline_accepted=baseline_ready,
                integrity_passed=item['integrity_passed'], integrity_failures=item['integrity_failures'],
                fixed_input_tensor_sha_matches_fp16=tensor_match, speed_gate_passed=speed,
                fixed_speed_cells=speed_cells, quality_gate_passed=None,
                reason='Fixed-input-only result; no20-case navigation result exists'))
            continue
        passed = bool(baseline_ready and item['integrity_passed'] and item['stability_passed']
                      and same20 and tensor_match and first_inputs_match and quality and speed)
        failed_locally = not item['integrity_passed'] or not item['stability_passed']
        row = dict(group=group, status='passed' if passed else 'failed' if baseline_ready or failed_locally else 'pending_baseline',
            passed=passed, baseline_accepted=baseline_ready, integrity_passed=item['integrity_passed'],
            stability_passed=item['stability_passed'], same20_and_execution_order=same20,
            fixed_input_tensor_sha_matches_fp16=tensor_match, quality_gate_passed=quality, speed_gate_passed=speed,
            episode_initial_input_matches_fp16=first_inputs_match,
            fixed_speed_cells=speed_cells, integrity_failures=item['integrity_failures'], **item['metrics'])
        table.append(row)
    return table


def audit_repeat(path, baseline):
    directory = path.parent if path.name == 'result.json' else path
    directory = directory.resolve()
    require(directory.is_relative_to(ROOT / 'logs'), 'FP16 repeat must be inside logs/')
    if not (directory / 'result.json').exists():
        return dict(status='pending', speed_review_required=True, directory=relative(directory),
                    reason='Requested FP16 repeat has no complete result yet')
    try:
        repeat = audit_group(directory)
        require(repeat['group'] == 'fp16' and repeat['pipeline_status'] == 'fixed_input_only_complete',
                'FP16 drift repeat must be the independent fixed-only FP16 run')
        require(repeat['integrity_passed'], 'FP16 repeat source/execution integrity failed: ' + repr(repeat['integrity_failures']))
        require(baseline and baseline.get('metrics'), 'Original full20 FP16 baseline is missing')
        require(repeat['input_tensors'] == baseline['input_tensors'], 'FP16 repeat actual input tensors changed')
        require(repeat['fixed_input_plan'] == baseline['fixed_input_plan'], 'FP16 repeat measurement plan changed')
        cells = {}
        for cell, values in repeat['fixed_input'].items():
            reference = baseline['fixed_input'][cell]
            ratios = {key: values[key] / reference[key] for key in ('http_p50_s', 'http_p95_s')}
            cells[cell] = dict(ratios_to_original_fp16=ratios,
                drift_fraction={key: ratio - 1 for key, ratio in ratios.items()},
                exceeds_10_percent=any(abs(ratio - 1) > .10 + 1e-12 for ratio in ratios.values()))
        review = any(value['exceeds_10_percent'] for value in cells.values())
        return dict(status='review_required' if review else 'within_10_percent', speed_review_required=review,
            comparison='Repeat/original FP16; symmetric absolute relative drift; original baseline remains the only speed denominator.',
            cells=cells, audit=repeat)
    except Exception as error:
        return dict(status='audit_failed', speed_review_required=True, directory=relative(directory),
                    reason=type(error).__name__ + ': ' + str(error))


def apply_drift_review(table, audits, repeat):
    if not repeat or not repeat['speed_review_required']:
        return
    for row in table:
        group = row['group']
        if group == 'fp16':
            row['performance_drift_review_required'] = True
        elif audits.get(group, {}).get('fixed_input'):
            row['raw_speed_gate_passed_against_original_fp16'] = row.get('speed_gate_passed')
            row['speed_gate_passed'] = None
            row['speed_conclusion'] = 'review_required'
            row['passed'] = False
            if row['status'] in ('passed', 'fixed_speed_gate_failed', 'fixed_input_only_complete'):
                row['status'] = 'review_required'


def fused_checkpoint_hashes(model_path, layer, projection):
    """Hash exactly the fused packed tensors, using only standard-library CPU IO."""
    path = ROOT / model_path / 'model.safetensors'
    require(path.resolve().is_relative_to(ROOT / 'models'), 'Kernel tensor path escapes models/')
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, layer, projection)
    if key in TENSOR_HASH_CACHE:
        return TENSOR_HASH_CACHE[key]
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        output = {}
        for suffix in ('qweight', 'qzeros', 'scales'):
            pieces, row_count = [], None
            for part in FUSED[projection]:
                item = header[f'model.language_model.layers.{layer}.{part}.{suffix}']
                require(len(item['shape']) == 2 and item['dtype'] in ('I32', 'F16'), 'Unexpected packed tensor layout')
                rows, columns = item['shape']
                require(row_count in (None, rows), 'Cannot concatenate projection tensors along columns')
                row_count = rows
                begin, end = item['data_offsets']
                width = columns * (4 if item['dtype'] == 'I32' else 2)
                require(end - begin == rows * width, 'Packed tensor byte length mismatch')
                stream.seek(data_start + begin)
                raw = stream.read(end - begin)
                require(len(raw) == end - begin, 'Truncated checkpoint tensor')
                pieces.append((memoryview(raw), width))
            value = hashlib.sha256()
            for row in range(row_count):
                for raw, width in pieces:
                    value.update(raw[row * width:(row + 1) * width])
            output[suffix] = value.hexdigest()
    TENSOR_HASH_CACHE[key] = output
    return output


def numerical_error(value, tolerance, label):
    require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value < tolerance,
            label + ': numerical reference error violates gate')


def audit_kernel_gate(path):
    path = path.resolve()
    require(path.is_relative_to(ROOT) and path.name == 'result.json', 'Kernel gate must be a project result.json')
    evidence = Evidence(path.parent)
    result = evidence.load(path)
    protocol = result['protocol']
    require(evidence.load(path.parent / 'protocol.json') == protocol, 'Kernel result/protocol mismatch')
    for name, expected in protocol['sources_sha256'].items():
        source = normalize(name)
        evidence.source(source if '/' in source else 'checks/' + source, expected)
    require(not evidence.failures, 'Kernel probe source integrity failed: ' + repr(evidence.failures))
    require(evidence.load(path.parent / 'measurements.json') == result['measurements'], 'Kernel measurement artifact differs')
    coverage = {}
    if 'navigation_w4a16_repair_kernels.py' in protocol['sources_sha256']:
        kernel = 'navigation_w4a16_repair_kernels.py'
        require(result['status'] == 'passed' and result['numerical_gate_passed'], 'W4A16 numerical gate did not pass')
        require(result['runtime_autotuner_cache_unchanged'] and result['runtime_compiled_cache_unchanged'],
                'W4A16 gate allowed runtime tuning/compilation')
        frozen = result['launch_table']
        require(frozen['frozen'] and not frozen['runtime_autotuning_allowed']
                and not frozen['runtime_cold_compilation_allowed'] and len(frozen['shapes']) == 4,
                'W4A16 probe launch table was not frozen')
        require(evidence.load(path.parent / 'launch_table.json') == frozen, 'W4A16 launch-table artifact differs')
        tolerance = protocol['relative_l2_limit']
        require(0 < tolerance <= .005 and set(protocol['fused_projections']) == set(FUSED), 'W4A16 reference protocol weakened')
        require({1, 701, 2815}.issubset(protocol['row_counts']) and 0 in protocol['layers'], 'Insufficient W4A16 numeric shapes')
        expected = {(g, layer, projection, rows) for g in protocol['groups'] for layer in protocol['layers']
                    for projection in protocol['fused_projections'] for rows in protocol['row_counts']}
        observed = [(r['group'], r['layer'], r['projection'], r['m']) for r in result['measurements']]
        require(len(observed) == len(set(observed)) and set(observed) == expected, 'W4A16 declared numerical coverage incomplete')
        for row in result['measurements']:
            require(set(row['variants']) == {'phase_tuned', 'frozen_launch'}, 'Missing W4A16 before/after reference')
            for variant, values in row['variants'].items():
                numerical_error(values['relative_l2'], tolerance, variant)
            numerical_error(row['frozen_vs_phase_relative_l2'], tolerance, 'W4A16 frozen/phase regression')
        for group in protocol['groups']:
            require(group in ('awq', 'gptq'), 'Unknown W4A16 probe group')
            model = protocol['model_files'][group]
            require((ROOT / model['path']).resolve().is_relative_to(ROOT / 'models'), 'Kernel model path escapes models/')
            evidence.source(model['path'] + '/config.json', model['config_sha256'])
            evidence.source(model['path'] + '/model.safetensors', model['weights_sha256'])
            rows = [r for r in result['measurements'] if r['group'] == group]
            coverage[group + '_w4a16'] = dict(numerical_cases=len(rows), layers=protocol['layers'],
                projections=list(protocol['fused_projections']), rows=protocol['row_counts'],
                max_reference_relative_l2=max(r['variants']['frozen_launch']['relative_l2'] for r in rows),
                model=model, kernel_variant='frozen_launch')
    elif 'navigation_precision_repair_kernels.py' in protocol['sources_sha256']:
        kernel = 'navigation_precision_repair_kernels.py'
        require(result['status'] == 'complete' and result['numerical_gate'] == 'passed', 'Precision numerical gate did not pass')
        tolerance = protocol['relative_l2_tolerance']
        require(0 < tolerance <= .005, 'Precision numerical tolerance weakened')
        args = protocol['args']
        require({'qkv', 'o', 'gate_up', 'down'} == set(args['projections'])
                and {0, 18, 35}.issubset(args['layers']) and {1, 701, 2815}.issubset(args['rows'])
                and 'balanced' in args['variants'], 'Precision gate lacks actual selected variant/layer/shape coverage')
        for bits, recipe in ACTIVATION_RECIPES.items():
            require(protocol['activation_recipes'][str(bits)] == recipe, 'Numerical gate activation definition changed')
        evidence.source('setup/navigation-precision-20261110.json', protocol['config_sha256'])
        for filename, field in [('synthetic.json', 'synthetic'), ('preparation.json', 'preparation'),
                                ('cache_checks.json', 'unseen_shape_cache_checks')]:
            require(evidence.load(path.parent / filename) == result[field], 'Precision probe artifact differs: ' + filename)
        frozen = result['final_kernel_snapshot']
        require(frozen['frozen'] and frozen['private_cold_compile_guard_installed']
                and frozen['cache_unchanged_since_freeze'], 'Precision probe was not frozen')
        rejection = result['preparation']['rejection_checks']
        require(all(rejection.get(key) is True for key in ('misaligned_pointer_rejected', 'unknown_geometry_rejected',
                    'unprepared_stride_rejected', 'private_cold_compile_rejected_before_compilation')),
                'Missing bad-input/cold-compilation rejection checks')
        expected = {(g, layer, projection, rows) for g in args['groups'] for layer in args['layers']
                    for projection in args['projections'] for rows in args['rows']}
        observed = [(r['group'], r['layer'], r['projection'], r['rows']) for r in result['measurements']]
        require(len(observed) == len(set(observed)) and set(observed) == expected, 'Precision declared numerical coverage incomplete')
        modes = {tuple(map(int, g.split('_w')[1].split('a'))) for g in args['groups']}
        expected_synthetic = {(wb, ab, m, v) for wb, ab in modes for m in (1, 17, 65) for v in args['variants']}
        observed_synthetic = [(r['weight_bits'], r['activation_bits'], r['rows'], r['variant']) for r in result['synthetic']]
        require(len(observed_synthetic) == len(set(observed_synthetic)) and set(observed_synthetic) == expected_synthetic,
                'Precision declared synthetic coverage incomplete')
        for row in result['synthetic'] + result['measurements']:
            group = row.get('group')
            bits = tuple(map(int, group.split('_w')[1].split('a'))) if group else (row['weight_bits'], row['activation_bits'])
            wb, ab = bits
            required_instruction = '.s8.s8.' if ab < 16 else '.f16.f16.'
            variants = row['variants'] if group else {row['variant']: row}
            if group:
                require(set(variants) == set(args['variants']), 'Missing declared numerical candidate variant')
            for variant, values in variants.items():
                numerical_error(values['relative_l2'], tolerance, 'Precision ' + variant)
                ptx = values['ptx']
                require(ptx['verified'] and ptx['required_instruction'] == required_instruction
                        and len(ptx['ptx_sha256']) == 64 and any(required_instruction in line for line in ptx['mma_instructions']),
                        'Missing actual selected-precision MMA instructions')
            activation = row['activation']
            if ab < 16:
                require(activation['packing'] == activation['scales'] == 'exact', 'Activation packing/scales differed from reference')
            else:
                require(activation['packing'] == 'not_applicable', 'Unexpected W8A16 activation quantization')
            if group:
                model = protocol['model_files'][group]
                require(row['tensors_sha256'] == fused_checkpoint_hashes(model['path'], row['layer'], row['projection']),
                        'Probe packed matrix differs from the frozen real checkpoint')
        require(result['synthetic'], 'Missing synthetic tail/zero/outlier numeric cases')
        for group in args['groups']:
            require(group in GROUPS and not group.endswith('w4a16') and group != 'fp16', 'Unknown extension probe group')
            wb, ab = tuple(map(int, group.split('_w')[1].split('a')))
            for m in (1, 17, 65):
                require(any((r['weight_bits'], r['activation_bits'], r['rows'], r['variant']) == (wb, ab, m, 'balanced')
                            for r in result['synthetic']), 'Missing selected-precision synthetic edge case')
            for projection in args['projections']:
                for m in args['cache_check_rows']:
                    require(any(tuple(map(int, r['group'].split('_w')[1].split('a'))) == (wb, ab)
                                and r['projection'] == projection and r['rows'] == m and r['variant'] == 'balanced'
                                and r['finite_output'] and r['cache_growth'] == 0
                                for r in result['unseen_shape_cache_checks']), 'Missing unseen-shape frozen cache check')
            model = protocol['model_files'][group]
            require((ROOT / model['path'] / 'model.safetensors').stat().st_size == model['safetensors_size'], 'Probe model size changed')
            evidence.source(model['path'] + '/config.json', model['config_sha256'])
            rows = [r for r in result['measurements'] if r['group'] == group]
            coverage[group] = dict(numerical_cases=len(rows), layers=args['layers'], projections=args['projections'],
                rows=args['rows'], max_reference_relative_l2=max(r['variants']['balanced']['relative_l2'] for r in rows),
                model=model, kernel_variant='balanced', activation_recipe=ACTIVATION_RECIPES[ab],
                actual_checkpoint_fused_tensor_hashes_verified=True)
    else:
        raise ValueError('Unrecognized numerical kernel probe')
    require(not evidence.failures, 'Kernel model/source hashes changed: ' + repr(evidence.failures))
    return dict(status='passed', result=relative(path), result_sha256=sha(path), kernel_source=kernel,
        kernel_sha256=protocol['sources_sha256'][kernel], coverage=coverage,
        source_checks=evidence.source_checks, evidence_files_sha256=evidence.hashes,
        scope='Reference arithmetic, packing, real packed matrices and execution/cache checks with synthetic inputs only.',
        quality_conclusion='None: random/synthetic activation errors do not prove activation fidelity or navigation quality.')


def apply_kernel_gates(table, audits, paths):
    gates = []
    for path in paths:
        try:
            gates.append(audit_kernel_gate(path))
        except Exception as error:
            gates.append(dict(status='audit_failed', result=str(path), error=type(error).__name__ + ': ' + str(error)))
    for row in table:
        group = row['group']
        if group == 'fp16':
            row['technical_gate_status'] = 'not_required_for_unquantized_baseline'
            continue
        item = audits.get(group)
        if not item or not item.get('actual_recipe'):
            continue
        matches, mismatches = [], []
        for gate in gates:
            if gate['status'] != 'passed' or group not in gate['coverage']:
                continue
            coverage = gate['coverage'][group]
            if item['service_sources_sha256'].get(gate['kernel_source']) != gate['kernel_sha256']:
                mismatches.append(dict(result=gate['result'], reason='Probe/service kernel source SHA differs'))
                continue
            model = coverage['model']
            if model['path'] != item['actual_recipe']['model'] or model['config_sha256'] != item['model_config_sha256']:
                mismatches.append(dict(result=gate['result'], reason='Probe/service model path or config SHA differs'))
                continue
            if model.get('weights_sha256') and model['weights_sha256'] != item['model_weights_sha256']:
                mismatches.append(dict(result=gate['result'], reason='Probe/service checkpoint SHA differs'))
                continue
            if coverage.get('activation_recipe') and coverage['activation_recipe'] != item['actual_recipe']['activation_recipe']:
                mismatches.append(dict(result=gate['result'], reason='Probe/service activation recipe differs'))
                continue
            matches.append(dict(result=gate['result'], result_sha256=gate['result_sha256'],
                kernel_sha256=gate['kernel_sha256'], numerical_cases=coverage['numerical_cases'],
                max_reference_relative_l2=coverage['max_reference_relative_l2']))
        row['technical_gate_status'] = 'passed' if matches else 'technical_evidence_pending'
        row['technical_gate_matches'] = matches
        row['technical_gate_mismatches'] = mismatches
        if not matches:
            row['passed'] = False
            if row['status'] == 'passed':
                row['status'] = 'technical_evidence_pending'
    return gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directories', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fp16-repeat', type=Path, help='Optional fixed-only FP16 group directory or its result.json; does not add another quality row.')
    parser.add_argument('--kernel-gates', nargs='*', default=[], type=Path,
                        help='Numerical probe result.json files; quantized groups cannot pass without matching kernel/model evidence.')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(ROOT / 'reports') and not output.exists(), 'Choose a fresh report path inside reports/')
    audits = {}
    for supplied in args.directories:
        directory = supplied.resolve()
        require(directory.is_relative_to(ROOT / 'logs'), 'Group directory must be under logs/')
        group = directory.name
        require(group in GROUPS and group not in audits, 'Unknown/duplicate group directory: ' + group)
        if not (directory / 'result.json').exists():
            failure_path = directory.parent / 'failure.json'
            audits[group] = dict(group=group, directory=relative(directory),
                pipeline_status='run_failed' if failure_path.exists() else 'pending',
                error=read(failure_path).get('error', 'Run failed before result') if failure_path.exists()
                      else 'No complete result.json; no acceptance inferred')
            continue
        try:
            audits[group] = audit_group(directory)
            require(audits[group]['group'] == group, 'Directory/result group mismatch')
        except Exception as error:
            audits[group] = dict(group=group, directory=relative(directory), pipeline_status='audit_failed',
                                 error=type(error).__name__ + ': ' + str(error))
    table = combine(audits)
    repeat = audit_repeat(args.fp16_repeat, audits.get('fp16')) if args.fp16_repeat else None
    apply_drift_review(table, audits, repeat)
    kernel_gates = apply_kernel_gates(table, audits, args.kernel_gates)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        audit_source_sha256=sha(Path(__file__)), protocol_sha256=sha(ROOT / 'reports/navigation_repair_protocol_20261112.md'),
        scope='Independent raw-step/timing audit of fixed20; not statistical equivalence or completed same100.',
        gates=dict(infrastructure_errors=0, format_or_truncation_errors=0, max_success_count_drop=2,
            max_spl_drop=.10, fixed_http_p50_p95_max_ratio=1.10, fixed_repetitions_per_cell_min=10,
            all_four_cells_required=True, actual_input_tensor_sha_equality_required=True),
        all_eight_quantized_20_gates_passed=all(r['passed'] for r in table if r['group'] != 'fp16'),
        formal100_status='Not assessed by this fixed20 audit',
        table=table, groups=audits, fp16_repeat=repeat, kernel_gates=kernel_gates,
        limitations=['20 cases screen engineering defects and obvious collapse; no statistical quality-equivalence claim.',
            'Fixed-input speed uses matching inputs/budgets; navigation request distributions may differ by trajectory.',
            'CUDA events include dispatch/idle; decode denominator is actual forwards including post-response work.',
            'Memory is whole-card 200ms sampling and includes desktop/prewarm usage.',
            'Kernel numerical gates validate reference arithmetic on synthetic inputs; they do not establish activation fidelity or navigation quality.',
            'A passed20 gate authorizes the next formal run; it does not mean100 cases are complete.'])
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(report=relative(output), table=table), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
