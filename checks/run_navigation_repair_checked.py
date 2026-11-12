"""Serial repair screens on a frozen roster, with raw execution/quality receipts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import shutil
import statistics
import subprocess
import time
import traceback
from urllib.request import Request

from navigation_stage2_common import ROOT, read, write, digest
from run_navigation_w4a16_fixes import PYTHON, OPENER, stop_child, gpu
from check_navigation_awq_dart_request import prepare_input
from navigation_precision_quality import verify_step
from collect_navigation_stage2 import percentile

BASELINE = None
FIXED_ONLY = False
FROZEN_TABLE = None


def fixed_summary(service, fixed_rows):
    result = {}
    for sample, tokens in ((s,t) for s in (0,1) for t in (18,64)):
        rows = [r for r in fixed_rows if r['phase'] == 'measured' and r['sample'] == sample and r['tokens'] == tokens]
        stages = [inspect_timing(service, r['index'], r['response']['usage']) for r in rows]
        times = [r['http_total_s'] for r in rows]
        result[f'{sample}_{tokens}'] = dict(n=len(rows), http_p50_s=statistics.median(times),
            http_p95_s=percentile(times, .95), stage_summary=summarize_stages(stages))
    return result


def source_snapshot(run, groups):
    names = ['run_navigation_repair_checked.py', 'run_navigation_repair.py',
        'serve_navigation_repair.py', 'navigation_repair_runtime.py', 'navigation_repair_module_map.py',
        'run_navigation_precision_diagnostic.py', 'navigation_precision_quality.py',
        'simplevln_policy.py', 'simplevln_env.py', 'run_closed_loop_baseline.py',
        'navigation_w4a16_fix_runtime.py', 'navigation_stage2_common.py',
        'check_navigation_awq_dart_request.py', 'navigation_repair_output_gate.py']
    if any(group.endswith('_w4a16') for group in groups):
        names += ['navigation_w4a16_repair_kernels.py']
    if any(group != 'fp16' and not group.endswith('_w4a16') for group in groups):
        names += ['navigation_precision_repair_kernels.py', 'serve_navigation_precision_repair.py',
                  'navigation_precision_repair_runtime.py', 'navigation_precision_repair_module_map.py']
    folder = run/'source_snapshot'
    folder.mkdir()
    rows = {}
    for name in names:
        path, target = ROOT/'checks'/name, folder/name
        shutil.copyfile(path, target)
        rows['checks/'+name] = dict(sha256=digest(target), snapshot=target.relative_to(ROOT).as_posix())
    write(run/'source_snapshot.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), files=rows))
    return rows


def http(port, group, payload, fixed=0):
    payload = dict(payload, model='navigation-repair-'+group, stream=False, seed=7)
    if fixed:
        payload.update(max_tokens=fixed, min_new_tokens=fixed, ignore_eos=True,
                       temperature=0, top_p=1, top_k=1, return_token_ids=True)
    encoded = json.dumps(payload, ensure_ascii=False).encode()
    time.sleep(1)  # fixed-input isolated load only; never inserted in navigation
    start = time.perf_counter()
    with OPENER.open(Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=encoded,
        headers={'Content-Type': 'application/json'}), timeout=300) as response:
        answer = json.load(response)
    elapsed = time.perf_counter()-start
    assert answer['model'] == 'navigation-repair-'+group
    assert len(answer['choices']) == 1 and answer['usage']['completion_tokens'] > 0
    if fixed:
        assert answer['usage']['completion_tokens'] == fixed
        assert answer['choices'][0]['finish_reason'] == 'length'
    return dict(response=answer, http_total_s=elapsed, request_sha256=hashlib.sha256(encoded).hexdigest())


def audit_navigation(folder, service, first_index):
    manifest, summary = read(folder/'manifest.json'), read(folder/'summary.json')
    assert summary['status'] == 'complete' and summary['episode_count'] == len(manifest['episodes'])
    ids = dict(zip(map(str, manifest['episodes']), map(str, manifest['trajectory_ids'])))
    assert set(ids) == {str(e['episode_id']) for e in summary['episodes']}
    requests, ordered, hashes, errors = [], [], {}, Counter()
    warmups = read(folder/'warmups.json')
    assert len(warmups) == 2
    for row in warmups:
        verify_step(row)
    index = first_index
    for row in warmups:
        inspect_timing(service, index, row['usage'])
        index += 1
    stages = []
    for ep in summary['episodes']:
        eid = str(ep['episode_id'])
        assert str(ep['trajectory_id']) == ids[eid]
        path = folder/('episode_'+eid)/'steps.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == ep['steps'] and all(r['step'] == i for i, r in enumerate(rows))
        for row in rows:
            error = verify_step(row)
            if error:
                errors[error] += 1
            if row['model_called']:
                stages.append(inspect_timing(service, index, row['usage']))
                index += 1
                requests.append(row)
        last = rows[-1]
        success = int(last['action'] == 'STOP' and ep['terminal_stop_source'] == 'model'
                      and not last.get('error') and ep['metrics']['distance_to_goal'] < 3)
        spl = success*ep['start_distance_m']/max(ep['start_distance_m'], ep['path_length_m'])
        if success:
            assert math.isclose(spl, ep['metrics']['spl'], abs_tol=1e-5)
        ordered.append(dict(episode_id=eid, trajectory_id=ids[eid], success=success, spl=spl,
            steps=len(rows), actual_model_calls=sum(r['model_called'] for r in rows),
            errors=sum(bool(r.get('error')) for r in rows),
            collisions=sum(bool(r['metrics']['collisions']['is_collision']) for r in rows),
            terminal_stop_source=ep['terminal_stop_source']))
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    latencies = [r['model_request_s'] for r in requests]
    memory = [float(line.split(',')[1]) for line in (folder/'gpu_samples.csv').read_text().splitlines() if len(line.split(',')) >= 3]
    return dict(status='complete', case_count=len(ids), success_count=sum(e['success'] for e in ordered),
        sr=sum(e['success'] for e in ordered)/len(ids), spl=sum(e['spl'] for e in ordered)/len(ids),
        actual_model_calls=len(requests), completion_tokens=sum(r['usage']['completion_tokens'] for r in requests),
        prompt_tokens=sum(r['usage']['prompt_tokens'] for r in requests),
        request_median_s=statistics.median(latencies), request_p95_s=percentile(latencies, .95),
        request_total_s=sum(latencies), sampled_whole_card_peak_mib=max(memory),
        error_count=sum(errors.values()), quality_errors=dict(errors), infrastructure_errors=0,
        warmup_errors=[r['error'] for r in warmups if r.get('error')], episodes=ordered,
        step_files_sha256=hashes, timing_start=first_index, timing_end_exclusive=index,
        stage_summary=summarize_stages(stages), manifest_sha256=digest(folder/'manifest.json'))


def inspect_timing(service, index, usage):
    value = read(service/f'timing_{index:03d}.json')
    steps = value['steps']
    assert value['index'] == index
    assert steps[0]['phase'] == 'prefill' and steps[0]['rows'] == usage['prompt_tokens'] <= 3072
    dec = steps[1:]
    assert dec and all(s['phase'] == 'decode' and s['rows'] == 1 and s['execution_mode'] == 'replay' for s in dec)
    assert len(steps) >= usage['completion_tokens']
    return dict(index=index, prefill_language_ms=steps[0]['timers']['language']['cuda_interval_ms'],
        prefill_vision_ms=steps[0]['timers']['vision']['cuda_interval_ms'],
        prefill_model_ms=steps[0]['timers']['model']['cuda_interval_ms'],
        decode_forwards=len(dec), output_tokens=usage['completion_tokens'],
        decode_model_total_ms=sum(s['timers']['model']['cuda_interval_ms'] for s in dec))


def summarize_stages(rows):
    values = [r['prefill_language_ms'] for r in rows]
    forwards = sum(r['decode_forwards'] for r in rows)
    dec = [r['decode_model_total_ms']/r['decode_forwards'] for r in rows]
    return dict(requests=len(rows), prefill_language_p50_ms=statistics.median(values),
        prefill_language_p95_ms=percentile(values, .95), actual_decode_forwards=forwards,
        decode_ms_per_actual_forward=sum(r['decode_model_total_ms'] for r in rows)/forwards,
        decode_request_mean_p50_ms=statistics.median(dec), decode_request_mean_p95_ms=percentile(dec, .95))


def run_group(run, group, manifest_path, port):
    directory = run/group
    directory.mkdir()
    service = directory/'service'
    service.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', LIBGL_ALWAYS_SOFTWARE='1')
    env.pop('DISPLAY', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    env.pop('PROJECT2_W4A16_REPAIR_TABLE', None)
    if FROZEN_TABLE is not None:
        assert group.endswith('_w4a16')
        env['PROJECT2_W4A16_REPAIR_TABLE'] = str(FROZEN_TABLE)
    child = evaluator = None
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        with (directory/'service_console.log').open('x') as console:
            service_entry = 'checks/serve_navigation_repair.py' if group == 'fp16' or group.endswith('_w4a16') else 'checks/serve_navigation_precision_repair.py'
            child = subprocess.Popen([PYTHON, '-B', '-u', service_entry,
                '--group', group, '--run-dir', str(service), '--port', str(port)], cwd=ROOT,
                env=env, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        write(directory/'ownership.json', dict(service_pid=child.pid, port=port))
        deadline = time.monotonic()+1200
        while True:
            assert child.poll() is None, 'Service exited; inspect retained console'
            try:
                with OPENER.open(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                    assert json.load(response)['data'][0]['id'] == 'navigation-repair-'+group
                break
            except OSError:
                assert time.monotonic() < deadline, 'Service startup timeout'
                time.sleep(2)
        evidence = read(service/'execution_audit.json')
        assert evidence['first_native_forward_verified']
        assert read(service/'startup_graph_capture.json')['actual_capture_observed']
        repair = read(service/'repair_execution_audit.json')
        if group != 'fp16':
            assert evidence['native_awq_receipt']['verified']
            assert evidence['native_awq_receipt']['module_count'] == 144
            assert evidence['native_awq_receipt']['loaded_quantized_tensor_count'] == 756
        payloads = [json.loads(prepare_input(1., sample, 'navigation')[-1]) for sample in (0, 1)]
        fixed_rows = []
        plan = [(sample, 'smoke', 0) for sample in (0,1)]
        plan += [(sample, 'warmup', tokens) for _ in range(2) for sample in (0,1) for tokens in (18,64)]
        plan += [(sample, 'measured', tokens) for _ in range(10) for sample in (0,1) for tokens in (18,64)]
        for sample, phase, tokens in plan:
            row = http(port, group, payloads[sample], fixed=tokens)
            row.update(index=len(fixed_rows), sample=sample, phase=phase, tokens=tokens)
            fixed_rows.append(row)
            write(directory/'fixed_requests.json', fixed_rows, exclusive=False)
            if phase == 'smoke':
                from navigation_repair_output_gate import inspect_answer
                output_gate = inspect_answer(row['response'])
                write(directory/f'smoke{sample}_output_gate.json', output_gate)
                assert output_gate['passed'], 'Natural output quality gate failed: '+str(output_gate)
            print('FIXED_REQUEST', group, phase, sample, round(row['http_total_s'],3), flush=True)
        # Explicit tail request flushes the last fixed measurement before the
        # speed gate. It is excluded and included in navigation's timing offset.
        write(directory/'fixed_flush_request.json', http(port, group, payloads[0], fixed=18))
        fixed_metrics = fixed_summary(service, fixed_rows)
        write(directory/'fixed_result.json', fixed_metrics)
        if FIXED_ONLY:
            result = dict(group=group, status='fixed_input_only_complete', case_count=0,
                success_count=None, fixed_input=fixed_metrics,
                source_hashes=read(service/'launch.json')['sources_sha256'])
            write(directory/'result.json', result)
            return result
        if BASELINE is not None:
            baseline = read(BASELINE)
            ratios = {key:{metric:values[metric]/baseline['fixed_input'][key][metric]
                for metric in ('http_p50_s', 'http_p95_s')} for key, values in fixed_metrics.items()}
            gate = dict(passed=all(r <= 1.10 for v in ratios.values() for r in v.values()),
                        ratios=ratios, baseline_sha256=digest(BASELINE), max_ratio=1.10)
            write(directory/'fixed_speed_gate.json', gate)
            if not gate['passed']:
                result = dict(group=group, status='fixed_speed_gate_failed', case_count=0,
                    success_count=None, fixed_input=fixed_metrics, gate=gate,
                    source_hashes=read(service/'launch.json')['sources_sha256'])
                write(directory/'result.json', result)
                return result
        manifest = read(manifest_path)
        actual_recipe = read(service/'launch.json')['recipe']
        manifest.update(dtype='FP16' if group == 'fp16' else 'W%dA%d' %
                        (actual_recipe['weight_bits'], actual_recipe['activation_bits']),
                        repair_recipe=actual_recipe,
                        source_precision='Frozen checkpoint; actual runtime precision in repair_recipe')
        manifest.update(model='navigation-repair-'+group, repair_group=group,
            backend='LMDeploy0.11.1 native decode graph, full prefill3072, FP16 KV64x64',
            scope=f"Repair evaluation: frozen {len(manifest['episodes'])} cases; unchanged author navigation policy")
        write(directory/'case_manifest.json', manifest)
        folder = directory/'navigation'
        with (directory/'navigation_console.log').open('x') as console:
            evaluator = subprocess.Popen(['bash', 'scripts/run_habitat_task_python.sh', '-u',
                'checks/run_navigation_precision_diagnostic.py', '--policy', 'memory',
                '--manifest', str(directory/'case_manifest.json'), '--output', str(folder),
                '--endpoint', f'http://127.0.0.1:{port}/v1'], cwd=ROOT, env=env,
                stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        write(directory/'evaluator_ownership.json', dict(pid=evaluator.pid))
        while evaluator.poll() is None:
            assert child.poll() is None, 'Service exited during navigation'
            time.sleep(3)
        assert evaluator.returncode == 0, 'Evaluator failed; inspect retained logs'
        evaluator = None
        write(directory/'flush_request.json', http(port, group, payloads[0], fixed=18))
        # Flush records previous navigation request synchronously before response.
        result = audit_navigation(folder, service, len(fixed_rows)+1)
        result.update(group=group, fixed_input={})
        for sample, tokens in ((s,t) for s in (0,1) for t in (18,64)):
            rows = [r for r in fixed_rows if r['phase'] == 'measured' and r['sample'] == sample and r['tokens'] == tokens]
            stages = [inspect_timing(service, r['index'], r['response']['usage']) for r in rows]
            times = [r['http_total_s'] for r in rows]
            result['fixed_input'][f'{sample}_{tokens}'] = dict(n=len(rows), http_p50_s=statistics.median(times),
                http_p95_s=percentile(times,.95), stage_summary=summarize_stages(stages))
        result['source_hashes'] = read(service/'launch.json')['sources_sha256']
        for name, sha in result['source_hashes'].items():
            assert digest(ROOT/'checks'/name) == sha, 'Runtime source changed during cohort'
        result['passed_stability'] = result['error_count'] == 0 and not result['warmup_errors']
        write(directory/'result.json', result)
        return result
    finally:
        stop_child(evaluator)
        stop_child(child)


def main():
    global BASELINE, FIXED_ONLY, FROZEN_TABLE
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--groups', nargs='+', default=['fp16','awq_w4a16','gptq_w4a16'])
    parser.add_argument('--base-port', type=int, default=23710)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--fixed-only', action='store_true')
    parser.add_argument('--frozen-launch-table', type=Path)
    args = parser.parse_args()
    FIXED_ONLY = args.fixed_only
    FROZEN_TABLE = args.frozen_launch_table.resolve() if args.frozen_launch_table else None
    if FROZEN_TABLE is not None:
        assert FROZEN_TABLE.is_relative_to(ROOT/'logs') and FROZEN_TABLE.is_file()
        assert len(args.groups) == 1 and args.groups[0].endswith('_w4a16')
    BASELINE = args.baseline.resolve() if args.baseline else None
    if BASELINE is not None:
        assert read(BASELINE)['group'] == 'fp16' and read(BASELINE)['passed_stability']
    run = args.output.resolve()
    assert run.is_relative_to(ROOT/'logs') and not run.exists()
    lock = (ROOT/'tmp/navigation_repair_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    assert not compute, 'GPU compute already active: '+compute
    run.mkdir(parents=True)
    snapshots = source_snapshot(run, args.groups)
    write(run/'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        groups=args.groups, manifest_sha256=digest(args.manifest), gpu_before=gpu(),
        controller_sha256=digest(Path(__file__)), navigation_policy_sha256=digest(ROOT/'checks/simplevln_policy.py'),
        evaluator_sha256=digest(ROOT/'checks/run_navigation_precision_diagnostic.py'),
        source_snapshot_sha256=digest(run/'source_snapshot.json'),
        frozen_launch_table_source=str(FROZEN_TABLE.relative_to(ROOT)) if FROZEN_TABLE else None,
        frozen_launch_table_source_sha256=digest(FROZEN_TABLE) if FROZEN_TABLE else None,
        fixed_input='2/9 images, 18/64 forced outputs, 2 warmups+10 repeats each, 1s untimed idle',
        navigation='Original temperature0.3/top_p0.95/max512/seed7; no added idle; two excluded warmups'))
    completed = []
    try:
        for i, group in enumerate(args.groups):
            write(run/'status.json', dict(status='running', active_group=group, completed=completed), exclusive=False)
            result = run_group(run, group, args.manifest, args.base_port+i)
            completed.append(group)
            print('SCREEN_COMPLETE', group, result['success_count'], result['case_count'], flush=True)
            for path, evidence in snapshots.items():
                assert digest(ROOT/path) == evidence['sha256'], 'Source changed during this invocation'
        write(run/'status.json', dict(status='complete', completed=completed), exclusive=False)
    except BaseException as error:
        write(run/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        write(run/'status.json', dict(status='failed', completed=completed), exclusive=False)
        raise


if __name__ == '__main__':
    main()
