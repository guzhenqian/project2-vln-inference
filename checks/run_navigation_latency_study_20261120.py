"""Matched fixed-input latency study; starts only its own serial services.

This controller does not change model weights, old runtimes or old reports.
Run it with the project WSL Python. Session configuration is a JSON list, e.g.:
[{"group":"fp16","label":"fp16_baseline","env":{"PROJECT2_LATENCY_VARIANT":"baseline"}},
 {"group":"awq_w4a16","label":"awq_baseline","frozen_table":"logs/.../table.json",
  "env":{"PROJECT2_LATENCY_VARIANT":"baseline"}}]

Per-session service_entry, model_name and extra_args override their defaults.
Stream and nonstream runs have different timing contracts and are never pooled.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import traceback
from urllib.request import Request, build_opener, ProxyHandler

from navigation_stage2_common import ROOT, GROUPS, digest, read, write
from check_navigation_awq_dart_request import prepare_input

PYTHON = str(Path.home()/'.venvs/project2/bin/python')
OPENER = build_opener(ProxyHandler({}))
FROZEN = {
    'fp16': 'b72a10c2b89a78f43c3c8bb05c4495b7d76dee85ec79f7125d9d795302aaa54b',
    'awq': '31d1c12dba3cbaace283cccf5be23337555f0383a3c4390e6270d2a9230414db',
    'gptq': '2479144ad32b5209cfb7f3b0927d863492e2894cc742603a2567c2802e99260f',
}

def utc():
    return datetime.now(timezone.utc).isoformat()


def sha_object(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def quantile(values, q):
    ordered = sorted(values)
    assert ordered and 0 <= q <= 1
    pos = (len(ordered)-1)*q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)


def gpu():
    binary = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    return subprocess.check_output([binary,
        '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()


def stop_owned(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


def resolve_project_path(value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    assert path.is_relative_to(ROOT), path
    return path


def parse_cells(values):
    cells = []
    for value in values:
        sample, tokens = map(int, value.split('_'))
        assert sample in (0, 1) and tokens in (18, 64), value
        assert (sample, tokens) not in cells, value
        cells.append((sample, tokens))
    return cells


def build_plan(cells, warmups, repeats):
    # The repair runtime hashes only the first two actual image requests.
    # Both shapes must occur here, including when screening a single cell.
    plan = [dict(phase='input_warmup', sample=s, tokens=0, repeat=0) for s in (0, 1)]
    plan += [dict(phase='warmup', sample=s, tokens=t, repeat=r)
             for r in range(warmups) for s, t in cells]
    plan += [dict(phase='measured', sample=s, tokens=t, repeat=r)
             for r in range(repeats) for s, t in cells]
    # The next prefill flushes the final measured request's CUDA events.
    plan.append(dict(phase='flush', sample=0, tokens=18, repeat=0))
    return plan


def request(port, model_name, spec, payloads, http_mode, idle_seconds):
    payload = dict(payloads[spec['sample']], model=model_name, seed=7,
                   stream=http_mode == 'stream', return_token_ids=True)
    count = spec['tokens']
    if count:
        payload.update(max_tokens=count, min_new_tokens=count, ignore_eos=True,
                       temperature=0, top_p=1, top_k=1)
    if http_mode == 'stream':
        payload['stream_options'] = {'include_usage': True}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    semantic_payload = dict(payload)
    semantic_payload.pop('model')
    arrivals, tokens, texts, reasons, usage = [], [], [], [], None
    time.sleep(idle_seconds)
    start = time.perf_counter()
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=encoded,
                  headers={'Content-Type': 'application/json'}, method='POST')
    with OPENER.open(req, timeout=300) as response:
        if http_mode == 'nonstream':
            answer = json.load(response)
            assert answer['model'] == model_name and len(answer['choices']) == 1
            choice = answer['choices'][0]
            tokens = choice['message'].get('gen_tokens') or []
            texts = [choice['message']['content'] or '']
            reasons = [choice['finish_reason']]
            usage = answer['usage']
        else:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                value = line[6:].strip()
                if value == b'[DONE]':
                    break
                chunk = json.loads(value)
                assert chunk.get('model', model_name) == model_name
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta') or {}
                    new = delta.get('gen_tokens') or []
                    if new:
                        arrivals.extend([time.perf_counter()-start]*len(new))
                        tokens.extend(new)
                    texts.append(delta.get('content') or '')
                    if choice.get('finish_reason'):
                        reasons.append(choice['finish_reason'])
            answer = dict(model=model_name, usage=usage, choices=[dict(index=0,
                message=dict(role='assistant', content=''.join(texts), gen_tokens=tokens),
                finish_reason=reasons[0] if len(reasons) == 1 else None)])
    elapsed = time.perf_counter()-start
    assert usage and usage['prompt_tokens'] == (701 if spec['sample'] == 0 else 2815), usage
    assert len(reasons) == 1 and usage['completion_tokens'] > 0, (reasons, usage)
    assert len(tokens) == usage['completion_tokens'], (len(tokens), usage)
    if count:
        assert len(tokens) == count and reasons == ['length'], (len(tokens), reasons)
    return dict(spec=spec, http_mode=http_mode, request_sha256=hashlib.sha256(encoded).hexdigest(),
        semantic_payload_sha256=sha_object(semantic_payload), usage=usage, response=answer,
        ttft_s=arrivals[0] if arrivals else None,
        stream_decode_ms_per_token=1000*(arrivals[-1]-arrivals[0])/(len(tokens)-1)
            if len(arrivals) > 1 else None,
        http_total_s=elapsed, output_tokens=tokens, output_tokens_sha256=sha_object(tokens),
        output_text_sha256=sha_object(''.join(texts)), token_arrival_s=arrivals,
        idle_seconds_outside_timer=idle_seconds)


def inspect_timing(service, index, row):
    timing = read(service/f'timing_{index:03d}.json')
    assert timing['index'] == index and timing['plan'] == row['spec']
    steps = timing['steps']
    assert steps[0]['phase'] == 'prefill'
    assert steps[0]['rows'] == row['usage']['prompt_tokens'] <= 3072
    dec = steps[1:]
    assert dec and all(x['phase'] == 'decode' and x['rows'] == 1 for x in dec)
    modes = Counter(x['execution_mode'] for x in dec)
    assert modes == {'replay': len(dec)}, modes
    assert len(steps) >= row['usage']['completion_tokens']
    pre = steps[0]
    result = dict(index=index, sample=row['spec']['sample'], tokens=row['spec']['tokens'],
        repeat=row['spec']['repeat'], http_total_s=row['http_total_s'], ttft_s=row['ttft_s'],
        stream_decode_ms_per_token=row['stream_decode_ms_per_token'],
        prefill_language_ms=pre['timers']['language']['cuda_interval_ms'],
        prefill_vision_ms=pre['timers']['vision']['cuda_interval_ms'],
        prefill_model_ms=pre['timers']['model']['cuda_interval_ms'],
        prefill_model_host_ms=pre['timers']['model']['host_wall_ms'],
        decode_model_ms_per_actual_forward=statistics.mean(x['timers']['model']['cuda_interval_ms'] for x in dec),
        decode_model_total_ms=sum(x['timers']['model']['cuda_interval_ms'] for x in dec),
        decode_through_logits_ms_per_actual_forward=statistics.mean(x['model_through_logits_ms'] for x in dec),
        actual_decode_forwards=len(dec), execution_modes=dict(modes),
        output_tokens_sha256=row['output_tokens_sha256'], output_tokens=row['output_tokens'],
        semantic_payload_sha256=row['semantic_payload_sha256'])
    return result


def summarize(directory, session, cells, repeats):
    requests = read(directory/'requests.json')
    measurements = []
    for index, row in enumerate(requests[:-1]):
        timing = inspect_timing(directory/'service', index, row)
        if row['spec']['phase'] == 'measured':
            measurements.append(timing)
    assert len(measurements) == len(cells)*repeats
    summary = {}
    for sample, tokens in cells:
        chosen = [r for r in measurements if (r['sample'], r['tokens']) == (sample, tokens)]
        assert len(chosen) == repeats
        metrics = {}
        for key in chosen[0]:
            if key.endswith(('_s', '_ms', '_per_token', '_per_actual_forward')):
                values = [r[key] for r in chosen if r[key] is not None]
                metrics[key] = dict(p50=statistics.median(values), p95=quantile(values, .95),
                                    min=min(values), max=max(values)) if values else None
        summary[f'{sample}_{tokens}'] = dict(n=len(chosen), metrics=metrics,
            actual_decode_forwards=sum(r['actual_decode_forwards'] for r in chosen),
            decode_ms_per_actual_forward=sum(r['decode_model_total_ms'] for r in chosen)/
                                         sum(r['actual_decode_forwards'] for r in chosen),
            output_sequences_identical=len({r['output_tokens_sha256'] for r in chosen}) == 1,
            output_tokens_sha256=chosen[0]['output_tokens_sha256'], output_tokens=chosen[0]['output_tokens'],
            semantic_payload_sha256=chosen[0]['semantic_payload_sha256'])
    return dict(session=session, cells=summary, measurements=measurements)


def comparisons(results):
    baseline = {}
    same_group = {}
    comparisons = []
    for result in results:
        group = result['session']['group']
        label = result['session']['label']
        for cell, current in result['cells'].items():
            if group == 'fp16' and cell not in baseline:
                baseline[cell] = (label, current)
            if cell in baseline:
                base_label, reference = baseline[cell]
                assert current['semantic_payload_sha256'] == reference['semantic_payload_sha256']
                comparisons.append(dict(label=label, cell=cell, reference_label=base_label,
                    kind='relative_to_first_fp16',
                    http_p50_ratio=current['metrics']['http_total_s']['p50']/reference['metrics']['http_total_s']['p50'],
                    http_p95_ratio=current['metrics']['http_total_s']['p95']/reference['metrics']['http_total_s']['p95']))
            key = (group, cell)
            if key not in same_group:
                same_group[key] = (label, current)
            else:
                base_label, reference = same_group[key]
                comparisons.append(dict(label=label, cell=cell, reference_label=base_label,
                    kind='same_weight_execution_change',
                    exact_output_tokens_equal=current['output_tokens_sha256'] == reference['output_tokens_sha256'],
                    both_repeats_identical=current['output_sequences_identical'] and reference['output_sequences_identical'],
                    http_p50_ratio=current['metrics']['http_total_s']['p50']/reference['metrics']['http_total_s']['p50'],
                    http_p95_ratio=current['metrics']['http_total_s']['p95']/reference['metrics']['http_total_s']['p95']))
    return comparisons


def run_session(run, number, session, args, payloads, cells, expected_tensors):
    directory = run/f"{number:02d}_{session['label']}"
    service = directory/'service'
    service.mkdir(parents=True)
    plan = build_plan(cells, args.warmups, args.repeats)
    write(service/'request_plan.json', plan)
    env = {k:v for k,v in os.environ.items() if not k.startswith('PROJECT2_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    env.update(session.get('env', {}))
    if session.get('frozen_table'):
        table = resolve_project_path(session['frozen_table'])
        assert table.is_file() and session['group'].endswith('_w4a16')
        env['PROJECT2_W4A16_REPAIR_TABLE'] = str(table)
    port = args.base_port+number
    command = [PYTHON, '-B', '-u', str(resolve_project_path(session['service_entry'])),
        '--group', session['group'], '--run-dir', str(service), '--port', str(port)]
    command += session.get('extra_args', [])
    write(directory/'command.json', dict(argv=command, environment_overrides=session.get('env', {}),
        frozen_table=session.get('frozen_table'), frozen_table_sha256=
        digest(resolve_project_path(session['frozen_table'])) if session.get('frozen_table') else None))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
    child = None
    samples, stopped = [], threading.Event()
    sampler = None
    try:
        with (directory/'console.log').open('x') as console:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write(directory/'ownership.json', dict(controller_pid=os.getpid(), service_pid=child.pid,
            service_process_group=child.pid, port=port, created_utc=utc()))
        deadline = time.monotonic()+args.startup_timeout
        while True:
            assert child.poll() is None, 'Service exited; see retained console'
            try:
                with OPENER.open(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    assert json.load(response)['data'][0]['id'] == session['model_name']
                break
            except OSError:
                assert time.monotonic() < deadline, 'Service startup timeout'
                time.sleep(2)
        audit = read(service/'execution_audit.json')
        assert audit['first_native_forward_verified']
        assert read(service/'startup_graph_capture.json')['actual_capture_observed']
        launch = read(service/'launch.json')
        method = session['group'].split('_')[0]
        assert launch['model_weights_sha256'] == FROZEN[method]
        if method != 'fp16':
            receipt = audit['native_awq_receipt']
            assert receipt['verified'] and receipt['module_count'] == 144
            assert receipt['loaded_quantized_tensor_count'] == 756

        def sample_gpu():
            while not stopped.is_set():
                try:
                    samples.append(dict(time=time.time(), value=gpu()))
                except Exception as error:
                    samples.append(dict(time=time.time(), error=repr(error)))
                stopped.wait(1)

        sampler = threading.Thread(target=sample_gpu, daemon=True)
        sampler.start()
        rows = []
        for index, spec in enumerate(plan):
            assert child.poll() is None, 'Service exited during requests'
            row = request(port, session['model_name'], spec, payloads, args.http_mode, args.idle_seconds)
            row['index'] = index
            rows.append(row)
            write(directory/'requests.json', rows, exclusive=False)
            if spec['phase'] == 'input_warmup':
                from navigation_repair_output_gate import inspect_answer
                gate = inspect_answer(row['response'])
                write(directory/f"natural_output_gate_{spec['sample']}.json", gate)
                assert gate['passed'], gate
            print('REQUEST', session['label'], spec['phase'], spec['sample'], spec['tokens'],
                  round(row['http_total_s'], 5), flush=True)
        for length in (701, 2815):
            evidence = read(service/f'input_{length}.json')
            actual = {k:evidence[k] for k in ('input_ids', 'pixel_values')}
            if length in expected_tensors:
                assert actual == expected_tensors[length], 'Actual input tensors differ'
            expected_tensors[length] = actual
        result = summarize(directory, session, cells, args.repeats)
        result['directory'] = directory.relative_to(run).as_posix()
        write(directory/'result.json', result)
        return result
    finally:
        stopped.set()
        if sampler:
            sampler.join(timeout=12)
        write(directory/'gpu_samples.json', samples)
        stop_owned(child)
        write(directory/'cleanup.json', dict(completed_utc=utc(),
              owned_service_pid=child.pid if child else None, owned_group_signalled=child is not None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-config', type=Path, required=True)
    parser.add_argument('--service-entry', default='checks/serve_navigation_latency_20261120.py')
    parser.add_argument('--cells', nargs='+', default=['0_18', '0_64', '1_18', '1_64'])
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--base-port', type=int, default=24110)
    parser.add_argument('--idle-seconds', type=float, default=1.)
    parser.add_argument('--http-mode', choices=['stream', 'nonstream'], default='stream')
    parser.add_argument('--startup-timeout', type=int, default=1200)
    args = parser.parse_args()
    assert args.warmups >= 1 and args.repeats >= 2 and args.idle_seconds >= 0
    assert 1024 <= args.base_port <= 65000
    cells = parse_cells(args.cells)
    sessions = read(args.session_config)
    assert isinstance(sessions, list) and sessions
    labels = set()
    for session in sessions:
        assert session['group'] in ('fp16', 'awq_w4a16', 'gptq_w4a16')
        assert re.fullmatch(r'[a-zA-Z0-9_-]+', session['label'])
        assert session['label'] not in labels
        labels.add(session['label'])
        session.setdefault('service_entry', args.service_entry)
        session.setdefault('model_name', 'navigation-repair-'+session['group'])
        assert resolve_project_path(session['service_entry']).is_file()
        assert isinstance(session.get('env', {}), dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k,v in session.get('env', {}).items())
        assert all(isinstance(v, str) for v in session.get('extra_args', []))
    assert args.base_port+len(sessions) < 65536
    run = args.output.resolve()
    assert run.is_relative_to(ROOT/'logs') and not run.exists()
    import fcntl
    (ROOT/'logs').mkdir(exist_ok=True)
    lock = (ROOT/'logs/navigation_latency_study_20261120.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(parents=True)
    completed, tensor_inputs = [], {}
    try:
        payloads, inputs, weights = {}, {}, {}
        for sample in (0, 1):
            source, settings, paths, hashes, encoded = prepare_input(1., sample, 'navigation')
            payloads[sample] = json.loads(encoded)
            inputs[sample] = dict(id=source['id'], image_paths=paths, image_sha256=hashes,
                                 payload_sha256=hashlib.sha256(encoded).hexdigest())
        for group in sorted({s['group'].split('_')[0] for s in sessions}):
            weights[group] = digest(ROOT/GROUPS[group]['model']/'model.safetensors')
            assert weights[group] == FROZEN[group], (group, weights[group])
        source_paths = {Path(__file__).resolve()}
        source_paths.update(resolve_project_path(s['service_entry']) for s in sessions)
        source_paths.update(ROOT/'checks'/name for name in (
            'navigation_repair_runtime.py', 'navigation_w4a16_fix_runtime.py',
            'navigation_w4a16_repair_kernels.py', 'navigation_stage2_common.py',
            'check_navigation_awq_dart_request.py', 'navigation_repair_output_gate.py'))
        # Include all newly added study modules without changing frozen files.
        source_paths.update((ROOT/'checks').glob('*latency*20261120*.py'))
        snapshot = run/'source_snapshot'
        snapshot.mkdir()
        sources = {}
        for path in sorted(source_paths):
            target = snapshot/path.name
            shutil.copyfile(path, target)
            sources[path.relative_to(ROOT).as_posix()] = digest(target)
        write(run/'protocol.json', dict(created_utc=utc(), sessions=sessions,
            http_mode=args.http_mode, cells=[f'{s}_{t}' for s,t in cells], warmups=args.warmups,
            measured_repeats=args.repeats, input_warmups=2, tail_requests=1,
            idle_seconds_outside_timer=args.idle_seconds, input_evidence=inputs,
            weight_sha256=weights, source_sha256=sources, gpu_before=gpu(),
            scope='Independent fixed-input speed/correctness study; no closed-loop navigation, no weight recalibration or token pruning.',
            timing_scope='HTTP starts after idle; warmups, capture and tail excluded. CUDA intervals may include dispatch gaps. Decode divisor is actual forward count including response-tail work. Stream token arrivals may be batched. Nonstream has no TTFT. Stream/nonstream runs must not be pooled.',
            baseline_scope='Ratios reference first same-run FP16; later FP16 sessions expose drift. Same-weight output equality is descriptive and must pass before deployment claims.'))
        for number, session in enumerate(sessions):
            write(run/'status.json', dict(status='running', updated_utc=utc(), current_session=session,
                completed_sessions=len(completed)), exclusive=False)
            result = run_session(run, number, session, args, payloads, cells, tensor_inputs)
            completed.append(result)
            write(run/'completed.json', completed, exclusive=False)
            write(run/'comparisons.json', comparisons(completed), exclusive=False)
        result = dict(status='complete', completed_utc=utc(), http_mode=args.http_mode,
            actual_input_tensors=tensor_inputs, sessions=completed, comparisons=comparisons(completed))
        write(run/'result.json', result)
        write(run/'status.json', dict(status='complete', updated_utc=utc(), completed_sessions=len(completed)), exclusive=False)
    except BaseException as error:
        write(run/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), at_utc=utc()))
        write(run/'status.json', dict(status='failed', updated_utc=utc(), completed_sessions=len(completed)), exclusive=False)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
