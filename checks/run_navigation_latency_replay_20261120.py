"""Single-pass HTTP replay of frozen, complete real navigation input histories.

Run prepare_navigation_latency_replay_20261120.py --prepare-only first. This
runner never decodes PNGs and never feeds outputs back into the recorded path.
It requires matching audited fixed-study evidence and owns only its serial
services. All 343 historical calls run once, greedily for 18 tokens, without
artificial idle. This is input-matched latency evidence, not navigation SR.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import time
import traceback
from types import SimpleNamespace
from urllib.request import Request


ROOT = Path(__file__).resolve().parents[1]
EPISODES = ['1250', '159', '1413', '558', '1123', '384']
EXPECTED_REQUESTS = 343
H = None


def require(value, message):
    if not value:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_helpers():
    # These are controller/audit helpers, never the service entry or a model.
    from navigation_stage2_common import read, write, digest
    from run_navigation_latency_navigation_20261120 import normalize_sessions, session_key, fixed_evidence
    from run_navigation_latency_study_20261120 import (PYTHON, OPENER, stop_owned, gpu, request,
                                                     inspect_timing, quantile)
    from summarize_navigation_latency_20261120 import execution_contract, wide_execution_audit
    from check_navigation_awq_dart_request import prepare_input
    from navigation_repair_output_gate import inspect_answer
    from prepare_navigation_latency_replay_20261120 import load_manifest, load_request
    return SimpleNamespace(**locals())


def post_replay(port, model, spec, encoded, proof, entry):
    """Only HTTP and response parsing are timed; body preparation is complete."""
    started = time.perf_counter()
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=encoded,
                  headers={'Content-Type': 'application/json'}, method='POST')
    with H.OPENER.open(req, timeout=300) as response:
        body = response.read()
        answer = json.loads(body)
    ended = time.perf_counter()
    require(answer.get('model') == model and len(answer.get('choices', [])) == 1,
            'Unexpected replay response model or choice count')
    choice = answer['choices'][0]
    tokens = choice['message'].get('gen_tokens') or []
    usage = answer['usage']
    require(usage['prompt_tokens'] == entry['source_prompt_tokens'], 'Historical prompt token count changed')
    require(len(tokens) == usage['completion_tokens'] == 18 and choice['finish_reason'] == 'length',
            'Replay must return exactly 18 tokens with length finish')
    require(all(isinstance(t, int) for t in tokens), 'Token IDs absent or invalid')
    text = choice['message']['content'] or ''
    return dict(spec=spec, http_mode='nonstream', request_sha256=proof['wire_bytes_sha256'],
        semantic_payload_sha256=entry['semantic_payload_sha256'], input_proof=proof,
        frozen_replay_index=entry['index'], episode_id=entry['episode_id'], source_step=entry['step'],
        episode_start=entry['is_episode_start'], episode_end=entry['is_episode_end'],
        source_image_sha256=entry['request_image_sha256'], usage=usage, response=answer,
        response_body_utf8=body.decode('utf-8'), response_bytes_sha256=hashlib.sha256(body).hexdigest(),
        output_tokens=tokens, output_tokens_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(),
        output_text_sha256=hashlib.sha256(json.dumps(text, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
        http_total_s=ended-started, client_perf_start_s=started, client_perf_end_s=ended,
        ttft_s=None, stream_decode_ms_per_token=None, token_arrival_s=[], idle_seconds_outside_timer=0)


def build_plan(entries):
    plan = [dict(phase='input_warmup', sample=i, tokens=0, repeat=0) for i in (0, 1)]
    plan += [dict(phase='measured', sample=e['index'], tokens=18, repeat=0,
                  replay_index=e['index'], episode_id=e['episode_id'], source_step=e['step']) for e in entries]
    plan.append(dict(phase='flush', sample=0, tokens=18, repeat=0))
    return plan


def summary_rows(rows):
    require(rows, 'Empty replay summary group')
    values = [r['http_total_s'] for r in rows]
    return dict(requests=len(rows), http_total_s=sum(values), http_p50_s=statistics.median(values),
                http_p95_s=H.quantile(values, .95), min_s=min(values), max_s=max(values),
                prompt_tokens=sum(r['usage']['prompt_tokens'] for r in rows),
                completion_tokens=sum(r['usage']['completion_tokens'] for r in rows))


def match_launch(service, evidence):
    launch = H.read(service/'launch.json')
    require(launch['engine'] == evidence['requested_engine'], 'Requested engine differs from fixed evidence')
    require(launch['model_weights_sha256'] == evidence['model_weights_sha256'], 'Model weights differ from fixed evidence')
    require(launch['sources_sha256'] == evidence['source_sha256'], 'Service source bytes differ from fixed evidence')
    require(H.read(service/'execution_audit.json')['first_native_forward_verified'], 'Native execution audit absent')
    require(H.read(service/'startup_graph_capture.json')['actual_capture_observed'], 'Decode graph capture absent')
    return launch


def match_execution(service, launch, evidence):
    contract = H.execution_contract(service, launch,
        {name:dict(sha256=sha) for name, sha in launch['sources_sha256'].items()})
    require(contract['effective_engine'] == evidence['engine'], 'Effective engine differs from fixed evidence')
    require(contract['execution_identity'] == evidence['execution_identity'], 'Execution receipt identity differs')
    require(contract['effective_engine']['enable_prefix_caching'] is False, 'Prefix KV caching must remain off')
    repair = H.read(service/'repair_execution_audit.json')
    require(not repair['extension'] and repair['activation_bits'] == 16 and repair['native_forward'],
            'Replay requires unchanged FP16/W4A16 native repair precision')
    require(repair['input_sha_scope'] == 'Explicit first two fixed smoke requests only',
            'Runtime input SHA scope differs from the audited two warmups')
    method = launch['recipe']['method']
    if method != 'fp16':
        native = H.read(service/'execution_audit.json')['native_awq_receipt']
        require(native['verified'] and native['module_count'] == 144
                and native['loaded_quantized_tensor_count'] == 756, 'Actual quantized module/weight execution is unverified')
        table = repair['frozen_launch_table']
        from summarize_navigation_latency_20261120 import sha_object
        require(table['frozen'] and table['actual_m_preserved'] and table['generic_m_specialization_disabled']
                and not table['runtime_autotuning_allowed'] and not table['runtime_cold_compilation_allowed'],
                'Actual native quantized launch table is not frozen')
        require(table['model_weights_sha256'] == evidence['model_weights_sha256'] and table['algorithm'] == method,
                'Actual launch table weight/algorithm identity differs')
        actual = sha_object(dict(shapes=table['shapes'], representatives=table['representatives'], max_rows=table['max_rows']))
        require(actual == evidence['frozen_launch_configuration_sha256'], 'Actual native launch table differs from fixed study')
        command = H.read(service.parent/'command.json')['configuration']
        if command.get('frozen_table_sha256'):
            require(table['selection_mode'] == 'replayed_frozen_table'
                    and table['frozen_table_source']['sha256'] == command['frozen_table_sha256'],
                    'Actual runtime did not replay the configured frozen source table')
        if contract['wide_evidence']:
            require(table['selection_mode'] == 'replayed_frozen_table'
                    and table['frozen_table_source']['sha256'] == contract['wide_evidence']['numerical_probe']['baseline_table_sha256'],
                    'Wide native fallback differs from its numerical proof')
    else:
        require(evidence['frozen_launch_configuration_sha256'] is None, 'FP16 fixed evidence has an unexpected W4 launch table')
    return contract


def run_session(run, number, session, replay_dir, manifest, evidence, port, timeout):
    directory = run/f"{number:02d}_{session['label']}"
    service = directory/'service'; service.mkdir(parents=True)
    entries = manifest['requests']
    plan = build_plan(entries)
    H.write(service/'request_plan.json', plan)
    # All payload parsing, hashing and model-name replacement precede GPU service
    # startup. The timed sequence reads only these already encoded HTTP bodies.
    prepared = []
    for entry in entries:
        wire, proof = H.load_request(replay_dir, entry, session['model_name'])
        require(proof['model_only_substitution'] is True, 'Replay payload changed beyond model name')
        require(proof['normalized_payload_bytes_sha256'] == entry['normalized_payload_bytes_sha256']
                and proof['semantic_payload_sha256'] == entry['semantic_payload_sha256'],
                'Prepared input proof differs from frozen manifest')
        require(hashlib.sha256(wire).hexdigest() == proof['wire_bytes_sha256'], 'Prepared wire SHA mismatch')
        prepared.append((wire, proof))
    payloads = {s:json.loads(H.prepare_input(1., s, 'navigation')[-1]) for s in (0, 1)}
    env = {k:v for k,v in os.environ.items() if not k.startswith('PROJECT2_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    env.update(session.get('env', {}))
    if session.get('frozen_table'):
        env['PROJECT2_W4A16_REPAIR_TABLE'] = str((ROOT/session['frozen_table']).resolve())
    command = [H.PYTHON, '-B', '-u', str((ROOT/session['service_entry']).resolve()),
               '--group', session['group'], '--run-dir', str(service), '--port', str(port)] + session.get('extra_args', [])
    H.write(directory/'fixed_evidence_binding.json', evidence)
    H.write(directory/'command.json', dict(argv=command, configuration=H.session_key(session)))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    child = None
    rows = []
    try:
        with (directory/'service_console.log').open('x') as console:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        H.write(directory/'ownership.json', dict(service_pid=child.pid, service_process_group=child.pid,
            controller_pid=os.getpid(), port=port, created_utc=utc()))
        deadline = time.monotonic()+timeout
        while True:
            require(child.poll() is None, 'Service exited during startup')
            try:
                with H.OPENER.open(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    require(json.load(response)['data'][0]['id'] == session['model_name'], 'Served model differs')
                break
            except OSError:
                require(time.monotonic() < deadline, 'Service startup timed out')
                time.sleep(2)
        launch = match_launch(service, evidence)
        for index in (0, 1):
            row = H.request(port, session['model_name'], plan[index], payloads, 'nonstream', 0.)
            row['index'] = index; rows.append(row)
            gate = H.inspect_answer(row['response'])
            H.write(directory/f'natural_output_gate_{index}.json', gate)
            require(gate['passed'], 'Natural input warmup failed')
        H.write(directory/'input_warmups.json', rows)
        input_receipts = {}
        for length in (701, 2815):
            value = H.read(service/f'input_{length}.json')
            require({k:value[k] for k in ('input_ids', 'pixel_values')} == evidence['actual_input_tensors'][str(length)],
                    'Two fixed warmup input tensors differ from audited fixed study')
            require(value['request_index'] == (0 if length == 701 else 1), 'Input tensor SHA is not from its excluded warmup')
            name = f'input_{length}.json'
            input_receipts[name] = dict(path='service/'+name, sha256=H.digest(service/name),
                first_request_index=value['request_index'], input_ids=value['input_ids'], pixel_values=value['pixel_values'],
                scope='Excluded fixed input warmup only; not a tensor SHA of any historical replay request')
        match_execution(service, launch, evidence)
        seen_lengths = {r['usage']['prompt_tokens'] for r in rows}
        previous_end = None
        with (directory/'replay_responses.jsonl').open('x', encoding='utf-8') as stream:
            for entry, (wire, proof) in zip(entries, prepared):
                index = entry['index']+2
                row = post_replay(port, session['model_name'], plan[index], wire, proof, entry)
                row['index'] = index
                row['first_seen_prompt_length_after_explicit_warmups'] = row['usage']['prompt_tokens'] not in seen_lengths
                seen_lengths.add(row['usage']['prompt_tokens'])
                row['client_gap_since_previous_replay_s'] = None if previous_end is None else row['client_perf_start_s']-previous_end
                previous_end = row['client_perf_end_s']
                rows.append(row)
                # Buffered tiny response records only; no images, source hashes,
                # stage inspection, explicit sleep or cache action in this loop.
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        tail = H.request(port, session['model_name'], plan[-1], payloads, 'nonstream', 0.)
        tail['index'] = len(plan)-1; rows.append(tail)
        H.write(directory/'requests.json', rows)
        H.write(directory/'tail_request.json', tail)
        require(len(rows) == len(plan) == EXPECTED_REQUESTS+3, 'Replay request sequence incomplete')
        require({p.name for p in service.glob('input_*.json')} == set(input_receipts),
                'Unexpected additional input tensor SHA receipt; actual repair override should limit these to two warmups')
        measurements = []
        for index, row in enumerate(rows[:-1]):
            stage = H.inspect_timing(service, index, row)
            if index < 2:
                continue
            entry = entries[index-2]
            stage.update(replay_index=entry['index'], episode_id=entry['episode_id'], source_step=entry['step'],
                first_seen_prompt_length_after_explicit_warmups=row['first_seen_prompt_length_after_explicit_warmups'],
                normalized_payload_bytes_sha256=entry['normalized_payload_bytes_sha256'],
                source_image_sha256=entry['request_image_sha256'],
                measured_request_tensor_sha_collected=False,
                input_identity_basis='Frozen payload/JPEG/text and actual prompt tokens; native tensor SHA is only in excluded warmups')
            measurements.append(stage)
        contract = match_execution(service, launch, evidence)
        if contract['vision_qmax_evidence']:
            for grid in contract['vision_qmax_evidence']['first_grid_coverage']:
                index = grid['request_index']
                require(0 <= index < len(rows)-1, 'QMAX first-grid receipt outside flushed sequence')
                timing = H.read(service/f'timing_{index:03d}.json')
                pre = [step for step in timing['steps'] if step['phase'] == 'prefill']
                require(timing['plan'] == plan[index] and len(pre) == 1 and pre[0]['rows'] == grid['input_tokens'],
                        'QMAX first-grid receipt differs from exact plan/prefill')
        wide_coverage = []
        if contract['wide_evidence']:
            native = H.read(service/'execution_audit.json')['native_awq_receipt']
            for index in range(len(rows)-1):
                timing = H.read(service/f'timing_{index:03d}.json')
                wide_coverage.append(H.wide_execution_audit(service, index, timing, contract, native))
        measured = rows[2:-1]
        H.write(directory/'stages.json', measurements)
        result = dict(status='complete', scope='Single-pass real-history fixed-18 HTTP replay; not navigation SR',
            session=session, fixed_evidence_binding=evidence, execution_contract=contract,
            replay_manifest_sha256=H.digest(replay_dir/'manifest.json'), measured_requests=len(measured),
            complete_summary=summary_rows(measured),
            sequence_wall_s=measured[-1]['client_perf_end_s']-measured[0]['client_perf_start_s'],
            by_episode={eid:summary_rows([r for r in measured if r['episode_id'] == eid]) for eid in EPISODES},
            first_prompt_length_groups={str(first):summary_rows(chosen) for first in (False, True)
                if (chosen := [r for r in measured if r['first_seen_prompt_length_after_explicit_warmups'] is first])},
            actual_decode_forwards=sum(s['actual_decode_forwards'] for s in measurements),
            total_decode_model_ms=sum(s['decode_model_total_ms'] for s in measurements),
            input_tensor_receipts=input_receipts, wide_request_coverage=wide_coverage,
            source_sha256=launch['sources_sha256'], measured_http_includes_new_input_tensor_hashing=False,
            artificial_idle_seconds=0, sequence_repetitions=1, outputs_fed_back=False,
            episode_boundaries='Recorded only; no cache implementation/reset action',
            input_identity='Exact frozen normalized payload/JPEG hashes and per-request prompt tokens; native tensor SHA only for two excluded fixed input warmups',
            first_prompt_length_group_scope='Descriptive input-length novelty only; repair override disables additional tensor hashing',
            qmax_scope='First-grid execution receipts plus only the two excluded numerical warmups; not per-request GPU counters',
            comparison_scope='Compare only same frozen replay/no-idle protocol; do not pool with fixed studies having idle gaps')
        for name, sha in launch['sources_sha256'].items():
            require(H.digest(ROOT/'checks'/name) == sha, 'Service source changed during replay: '+name)
        H.write(directory/'result.json', result)
        return result
    except BaseException:
        if rows and not (directory/'requests.json').exists():
            H.write(directory/'requests_partial.json', rows)
        raise
    finally:
        H.stop_owned(child)
        H.write(directory/'cleanup.json', dict(completed_utc=utc(), service_pid=child.pid if child else None))


def main():
    global H
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path, help='Completed --prepare-only output directory')
    parser.add_argument('--session-config', required=True, type=Path)
    parser.add_argument('--fixed-study', nargs='+', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--service-entry', default='checks/serve_navigation_latency_20261120.py')
    parser.add_argument('--base-port', type=int, default=24510)
    parser.add_argument('--startup-timeout', type=int, default=1200)
    parser.add_argument('--check-only', action='store_true',
        help='Audit configuration without services; wide evidence may mmap packed weights, so use an idle window')
    args = parser.parse_args()
    require(os.name == 'posix' or args.check_only, 'Run the service controller with the project WSL Python')
    H = load_helpers()
    replay_dir = args.replay.resolve()
    manifest = H.load_manifest(replay_dir)
    require(len(manifest['requests']) == EXPECTED_REQUESTS, 'Only the complete frozen 343-request sequence is permitted')
    require(list(dict.fromkeys(e['episode_id'] for e in manifest['requests'])) == EPISODES, 'Frozen episode order changed')
    require(manifest['cache_contract']['vision_cache'] is False and manifest['cache_contract']['prefix_kv_cache'] is False,
            'This runner does not implement cache reset; cache-enabled replay is not permitted')
    sessions = H.normalize_sessions(args.session_config, args.service_entry)
    session_keys = [json.dumps(H.session_key(s), sort_keys=True) for s in sessions]
    require(len(set(session_keys)) == len(sessions), 'Each model/configuration may replay the full sequence only once per run')
    for session in sessions:
        require(session.get('env', {}).get('PROJECT2_LATENCY_PROFILE') != '1'
                and session.get('env', {}).get('PROJECT2_LATENCY_SCHEDULE_TRACE') != '1',
                'Diagnostic frontend/schedule tracing cannot enter replay timing')
        require(not any('CACHE' in k and v not in ('0', 'off', 'false') for k, v in session.get('env', {}).items()),
                'Cache-related overrides are unsupported by this uncached replay protocol')
    if args.check_only:
        evidence = [H.fixed_evidence(args.fixed_study, session) for session in sessions]
        print(json.dumps(dict(status='configuration_checked', requests=EXPECTED_REQUESTS,
            episodes=EPISODES, fixed_bindings=len(evidence), gpu_started=False, images_decoded=False)))
        return
    require(args.output is not None, 'A fresh output directory is required')
    require(1024 <= args.base_port < 65536-len(sessions), 'Invalid port range')
    run = args.output.resolve()
    require(run.is_relative_to(ROOT/'logs') and not run.exists(), 'Use a fresh project logs directory')
    import fcntl
    locks = []
    completed = []
    try:
        for path in (ROOT/'logs/navigation_latency_study_20261120.lock', ROOT/'tmp/navigation_repair_gpu.lock'):
            lock = path.open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        evidence = [H.fixed_evidence(args.fixed_study, session) for session in sessions]
        run.mkdir(parents=True)
        names = ['checks/run_navigation_latency_replay_20261120.py', 'checks/prepare_navigation_latency_replay_20261120.py',
            'checks/run_navigation_latency_navigation_20261120.py', 'checks/run_navigation_latency_study_20261120.py',
            'checks/summarize_navigation_latency_20261120.py', 'checks/navigation_stage2_common.py',
            'checks/check_navigation_awq_dart_request.py', 'checks/navigation_repair_output_gate.py']
        names += ['checks/'+name for item in evidence for name in item['source_sha256']]
        sources = {}
        for name in sorted(set(names)):
            target = run/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target); sources[name] = H.digest(target)
        # Existing execution_contract looks up runtime modules directly under
        # source_snapshot; retain that layout as well as full-path provenance.
        for module in sorted({name for item in evidence for name in item['source_sha256']}):
            target = run/'source_snapshot'/module
            shutil.copyfile(ROOT/'checks'/module, target)
            require(H.digest(target) == sources['checks/'+module], 'Runtime snapshot layouts disagree')
        H.write(run/'replay_manifest_binding.json', dict(path=str(replay_dir/'manifest.json'),
            sha256=H.digest(replay_dir/'manifest.json'), status='complete', requests=EXPECTED_REQUESTS,
            episodes=EPISODES, source_files_sha256=manifest['source_files_sha256']))
        H.write(run/'protocol.json', dict(created_utc=utc(), scope=__doc__, sessions=sessions,
            replay_manifest_sha256=H.digest(replay_dir/'manifest.json'), source_sha256=sources,
            fixed_study_bindings=evidence, gpu_before=H.gpu(), prepared_all_images_before_service=True,
            output_contract='18 tokens, greedy top_k1/top_p1/temperature0, seed7, min_new_tokens18, ignore_eosTrue',
            measured_requests=EXPECTED_REQUESTS, artificial_idle_seconds=0, sequence_repetitions=1,
            excludes='Two natural input SHA warmups and one final timing flush; no historical request excluded',
            input_tensor_sha_scope='Two excluded input warmups only; verified active repair override suppresses additional per-length hashing',
            outputs_fed_back=False, navigation_success_claim=False, vision_cache=False, prefix_kv_cache=False))
        for number, session in enumerate(sessions):
            H.write(run/'status.json', dict(status='running', updated_utc=utc(), current_session=session,
                completed_sessions=len(completed)), exclusive=False)
            result = run_session(run, number, session, replay_dir, manifest, evidence[number],
                                 args.base_port+number, args.startup_timeout)
            completed.append(result)
            H.write(run/'completed.json', completed, exclusive=False)
            for name, sha in sources.items():
                require(H.digest(ROOT/name) == sha, 'Controller/runtime source changed during replay: '+name)
        H.write(run/'result.json', dict(status='complete', completed_utc=utc(), sessions=completed,
            scope='Descriptive complete replay data; matched counterpart audit required before improvement claims'))
        H.write(run/'status.json', dict(status='complete', updated_utc=utc(), completed_sessions=len(completed)), exclusive=False)
    except BaseException as error:
        if run.exists():
            H.write(run/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), at_utc=utc()))
            H.write(run/'status.json', dict(status='failed', updated_utc=utc(), completed_sessions=len(completed)), exclusive=False)
        raise
    finally:
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == '__main__':
    main()
