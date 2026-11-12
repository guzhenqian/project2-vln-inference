"""Serial, fixed-input W4A16 optimization experiments; not a navigation cohort."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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


def stop_child(child):
    if child is None:
        return
    # Only the process group created by this controller is targeted. Children
    # can survive their server parent, so do not skip cleanup on parent exit.
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


def gpu():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()


def request(port, group, spec, payloads):
    payload = dict(payloads[spec['sample']])
    count = spec['tokens']
    payload.update(model='w4a16-fix-'+group, max_tokens=count, min_new_tokens=count,
        ignore_eos=True, temperature=0, top_p=1, top_k=1, seed=7, stream=True,
        stream_options={'include_usage': True}, return_token_ids=True, skip_special_tokens=False)
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    arrivals, tokens, contents, reasons, usage = [], [], [], [], None
    start = time.perf_counter()
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=encoded,
                  headers={'Content-Type': 'application/json'}, method='POST')
    with OPENER.open(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            value = line[6:].strip()
            if value == b'[DONE]':
                break
            chunk = json.loads(value)
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                delta = choice.get('delta') or {}
                new = delta.get('gen_tokens') or []
                if new:
                    arrivals.extend([time.perf_counter()-start]*len(new))
                    tokens.extend(new)
                contents.append(delta.get('content') or '')
                if choice.get('finish_reason'):
                    reasons.append(choice['finish_reason'])
    elapsed = time.perf_counter()-start
    assert len(tokens) == count and usage['completion_tokens'] == count, (len(tokens), usage)
    assert usage['prompt_tokens'] == (701 if spec['sample'] == 0 else 2815), usage
    assert reasons == ['length'], reasons
    return dict(spec=spec, request_sha256=hashlib.sha256(encoded).hexdigest(), usage=usage,
        ttft_s=arrivals[0], decode_ms_per_token=1000*(arrivals[-1]-arrivals[0])/(count-1),
        total_s=elapsed, output_tokens=tokens, text=''.join(contents), token_arrival_s=arrivals)


def summarize(directory, session):
    requests = read(directory/'requests.json')
    rows = []
    for i, row in enumerate(requests[:-1]):
        timing = read(directory/'service'/f'timing_{i:03d}.json')
        assert timing['index'] == i and timing['plan'] == row['spec']
        steps = timing['steps']
        assert steps[0]['phase'] == 'prefill' and steps[0]['rows'] == row['usage']['prompt_tokens']
        assert all(x['phase'] == 'decode' and x['rows'] == 1 for x in steps[1:])
        if row['spec']['phase'] != 'measured':
            continue
        modes = Counter(x['execution_mode'] for x in steps[1:])
        expected = 'replay' if session['graph'] == 'on' else 'eager'
        assert modes == {expected: len(steps)-1}, (session, modes)
        pre, dec = steps[0], steps[1:]
        assert dec
        rows.append(dict(sample=row['spec']['sample'], tokens=row['spec']['tokens'],
            repeat=row['spec']['repeat'], http_total_s=row['total_s'], ttft_s=row['ttft_s'],
            stream_decode_ms_per_token=row['decode_ms_per_token'],
            prefill_language_ms=pre['timers']['language']['cuda_interval_ms'],
            prefill_vision_ms=pre['timers']['vision']['cuda_interval_ms'],
            prefill_model_ms=pre['timers']['model']['cuda_interval_ms'],
            decode_model_ms_per_token=statistics.mean(x['timers']['model']['cuda_interval_ms'] for x in dec),
            decode_through_logits_ms_per_token=statistics.mean(x['model_through_logits_ms'] for x in dec),
            actual_decode_forwards=len(dec), execution_modes=dict(modes), output_tokens=row['output_tokens']))
    result = []
    for sample in (0, 1):
        for count in (18, 64):
            chosen = [r for r in rows if (r['sample'], r['tokens']) == (sample, count)]
            assert chosen
            keys = [k for k in chosen[0] if k.endswith(('_s', '_ms', '_per_token'))]
            result.append(dict(sample=sample, tokens=count, n=len(chosen),
                medians={k: statistics.median(r[k] for r in chosen) for k in keys},
                ranges={k: [min(r[k] for r in chosen), max(r[k] for r in chosen)] for k in keys},
                output_sequences_identical=all(r['output_tokens'] == chosen[0]['output_tokens'] for r in chosen),
                output_tokens=chosen[0]['output_tokens']))
    return dict(session=session, summary=result, measurements=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', nargs='+', required=True, help='group:variant:off/on')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--base-port', type=int, default=23510)
    args = parser.parse_args()
    run = args.output.resolve()
    assert run.is_relative_to(ROOT/'logs') and not run.exists()
    run.mkdir(parents=True)
    sessions = [dict(zip(('group', 'variant', 'graph'), value.split(':'))) for value in args.sessions]
    for s in sessions:
        assert s['group'] in FROZEN and s['graph'] in ('off', 'on')
    payloads, inputs, hashes = {}, {}, {}
    for sample in (0, 1):
        source, settings, paths, image_hashes, encoded = prepare_input(1., sample, 'navigation')
        payloads[sample] = json.loads(encoded)
        inputs[sample] = dict(id=source['id'], images=paths, sha256=image_hashes,
                             payload_sha256=hashlib.sha256(encoded).hexdigest())
    for group in sorted({s['group'] for s in sessions}):
        hashes[group] = digest(ROOT/GROUPS[group]['model']/'model.safetensors')
        assert hashes[group] == FROZEN[group], (group, hashes[group])
        print('WEIGHT_SHA_VERIFIED', group, flush=True)
    source_names = ['run_navigation_w4a16_fixes.py', 'serve_navigation_w4a16_fix.py',
        'navigation_w4a16_fix_runtime.py', 'navigation_w4a16_fix_module_map.py',
        'navigation_w4a16_candidate_kernels.py']
    write(run/'protocol.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), sessions=sessions,
        warmups=args.warmups, repeats=args.repeats, actual_weight_sha256=hashes, inputs=inputs,
        sources_sha256={name: digest(ROOT/'checks'/name) for name in source_names},
        scope='Same 2/9 image inputs, 701/2815 prompt tokens, 18/64 forced greedy outputs. Native instrumentation common to all configurations. No closed-loop navigation or precision recalibration.',
        timing_scope='CUDA intervals may include host dispatch gaps; stream measurements include scheduling/sampling. Sampled whole-card memory includes desktop use. Warmup/capture excluded.'))
    completed, child = [], None
    tensor_inputs = {}
    try:
        for index, session in enumerate(sessions):
            group, variant, graph = (session[k] for k in ('group', 'variant', 'graph'))
            directory = run/f'{index:02d}_{group}_{variant}_{graph}'
            service = directory/'service'
            service.mkdir(parents=True)
            plan = [dict(phase='warmup', sample=s, tokens=t, repeat=r)
                    for s in (0, 1) for t in (18, 64) for r in range(args.warmups)]
            plan += [dict(phase='measured', sample=s, tokens=t, repeat=r)
                     for r in range(args.repeats) for s in (0, 1) for t in (18, 64)]
            plan.append(dict(phase='flush', sample=0, tokens=18, repeat=0))
            write(service/'request_plan.json', plan)
            port = args.base_port+index
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
            write(run/'status.json', dict(status='running', session=session, completed_sessions=len(completed)), exclusive=False)
            with (directory/'console.log').open('x') as console:
                child = subprocess.Popen([PYTHON, '-B', '-u', 'checks/serve_navigation_w4a16_fix.py',
                    '--group', group, '--variant', variant, '--graph', graph, '--run-dir', str(service),
                    '--port', str(port)], cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                    PYTHONUNBUFFERED='1'), stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
            write(directory/'ownership.json', dict(pid=child.pid, process_group=child.pid, port=port))
            print('SESSION_START', index, session, flush=True)
            deadline = time.monotonic()+600
            while True:
                if child.poll() is not None:
                    raise RuntimeError(f'Service exited: {directory}')
                try:
                    with OPENER.open(f'http://127.0.0.1:{port}/v1/models', timeout=2) as response:
                        assert json.load(response)['data'][0]['id'] == 'w4a16-fix-'+group
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Service startup')
                    time.sleep(2)
            audit = read(service/'execution_audit.json')
            assert audit['group'] == group and audit['variant'] == variant
            assert audit['graph_requested'] == graph
            assert audit['first_native_forward_verified']
            if group != 'fp16':
                assert audit['native_awq_receipt']['verified']
                assert audit['native_awq_receipt']['module_count'] == 144
                assert audit['native_awq_receipt']['loaded_quantized_tensor_count'] == 756
            if graph == 'on':
                assert read(service/'startup_graph_capture.json')['actual_capture_observed']
            samples, stopped = [], threading.Event()

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
            try:
                for spec in plan:
                    row = request(port, group, spec, payloads)
                    assert (service/f"input_{row['usage']['prompt_tokens']}.json").is_file(), 'Runtime instrumentation absent'
                    rows.append(row)
                    write(directory/'requests.json', rows, exclusive=False)
                    if spec['phase'] == 'measured':
                        print('MEASURED', group, variant, graph, spec['sample'], spec['tokens'],
                            round(row['total_s'], 4), round(row['decode_ms_per_token'], 3), flush=True)
            finally:
                stopped.set()
                sampler.join(timeout=12)
                write(directory/'gpu_samples.json', samples)
            stop_child(child)
            child = None
            for length in (701, 2815):
                observed = read(service/f'input_{length}.json')
                value = {k: observed[k] for k in ('input_ids', 'pixel_values')}
                if length in tensor_inputs:
                    assert value == tensor_inputs[length], 'Actual input tensors differ'
                tensor_inputs[length] = value
            result = summarize(directory, dict(session, directory=directory.relative_to(run).as_posix()))
            write(directory/'result.json', result)
            completed.append(result)
            write(run/'completed.json', completed, exclusive=False)
            print('SESSION_COMPLETE', session, flush=True)
        write(run/'result.json', dict(status='complete', actual_input_tensors=tensor_inputs, sessions=completed))
        write(run/'status.json', dict(status='complete', completed_sessions=len(completed)), exclusive=False)
    except BaseException as error:
        write(run/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        write(run/'status.json', dict(status='failed', completed_sessions=len(completed)), exclusive=False)
        raise
    finally:
        stop_child(child)


if __name__ == '__main__':
    main()
