"""Run the exact 150 completed FP16 baseline episodes with the learned AWQ model."""
import argparse
import base64
import collections
import copy
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'data/baseline/overnight200_fp16_20261031'
PLAN = ROOT / 'data/baseline/overnight150_awq_20261104'
OUT = ROOT / 'logs/overnight150_awq_20261104'
REPORT = ROOT / 'reports/benchmark_awq150_vs_fp16_20261104'
BASELINE = ROOT / 'reports/benchmark_overnight200_20261031.json'
MODEL = 'SimpleVLN-Qwen3VL-4B-RL-AWQ-learning-20261104'
MODEL_DIR = ROOT / 'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104'
ENDPOINT = 'http://127.0.0.1:23333/v1'
TZ = dt.timezone(dt.timedelta(hours=8))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def now():
    return dt.datetime.now(TZ).isoformat()


def stats(values):
    if not values:
        return None
    values = sorted(values)
    p = (len(values) - 1) * .95
    lo = int(p)
    return dict(count=len(values), median=statistics.median(values),
                p95=values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo),
                mean=statistics.mean(values), total=sum(values))


def prepare():
    baseline = read(BASELINE)
    assert baseline['completed_episode_count'] == 150
    expected = {e['episode_id'] for e in baseline['episodes']}
    batches = [read(ORIGINAL / ('batch_%02d.json' % i)) for i in range(1, 4)]
    ids = [eid for batch in batches for eid in batch['episodes']]
    assert len(ids) == len(set(ids)) == 150 and set(ids) == expected
    calibration = read(ROOT / 'data/baseline/navigation_awq_calibration_v1/manifest.json')
    assert not set(calibration['scenes']).intersection(batches[0]['scenes'])
    kernel = read(ROOT / 'reports/navigation_awq_kernel_learning_20261104.json')
    audit = read(MODEL_DIR / 'conversion_audit.json')
    assert kernel['status'] == 'passed' and len(kernel['checks']) == 18
    assert kernel['model_sha256'] == audit['sha256']
    provenance = read(ROOT / 'logs/overnight200_fp16_20261031/batch_02/provenance.json')
    source_checks = {}
    for name, expected_hash in provenance['scripts'].items():
        path = ROOT / 'checks' / name
        assert digest(path) == expected_hash, 'Baseline code changed: ' + name
        source_checks[path.relative_to(ROOT).as_posix()] = expected_hash
    policy_path = ROOT / 'tmp/habitat_setup/navigation_source_review/eval/vln/eval.py'
    assert digest(policy_path) == batches[0]['author_policy_sha256']
    source_checks[policy_path.relative_to(ROOT).as_posix()] = digest(policy_path)
    dataset = ROOT / batches[0]['dataset_path'].format(split=batches[0]['split'])
    assert digest(dataset) == batches[0]['source_annotation_sha256']
    source_checks[dataset.relative_to(ROOT).as_posix()] = digest(dataset)
    if PLAN.exists():
        raise FileExistsError('Frozen AWQ plan already exists; use --check-only')
    PLAN.mkdir(parents=True)
    planned = []
    for i, original in enumerate(batches, 1):
        new = copy.deepcopy(original)
        new.update(version='overnight150_awq_20261104', model=MODEL, dtype='W4A16',
                   scope='Exact 150 completed FP16 baseline episodes; three fixed batches of50; AWQ comparison',
                   source_precision='F32 source rounded to FP16 before calibration; AWQ W4 weights and FP16 runtime',
                   differences_from_author_benchmark='Habitat0.1.7/LMDeploy; AWQ W4A16; fixed150 development subset, not full1839',
                   pre_registered='Original task membership frozen20261031; same completed150 used for20261104 AWQ comparison')
        allowed = {'version', 'model', 'dtype', 'scope', 'source_precision', 'differences_from_author_benchmark', 'pre_registered'}
        assert all(new[k] == original[k] for k in original if k not in allowed)
        path = PLAN / ('batch_%02d.json' % i)
        write(path, new)
        planned.append(dict(batch=i, episodes=new['episodes'], sha256=digest(path),
                            fp16_manifest_sha256=digest(ORIGINAL / path.name)))
    write(PLAN / 'manifest.json', dict(created_local=now(), planned_count=150, model=MODEL,
          model_directory=MODEL_DIR.relative_to(ROOT).as_posix(), model_sha256=audit['sha256'],
          kernel_report='reports/navigation_awq_kernel_learning_20261104.json',
          baseline_report=BASELINE.relative_to(ROOT).as_posix(), baseline_report_sha256=digest(BASELINE),
          batches=planned, source_checks=source_checks, calibration_evaluation_scene_overlap=[],
          protocol='seed7, 400 atomic actions, model STOP with geodesic distance<3m; original parser; no recovery actions',
          latency='HTTP request timings from model_called=true only; two warmups per batch excluded',
          memory='whole-card memory.used sampled every200ms; includes desktop and engine cache; not model-exclusive',
          precision_note='Old manifest source_precision text contained stale BF16 wording; actual baseline runtime was FP16.',
          timing_note='FP16 batch1 interruption/recovery preserved; episode735 prefix IO omitted from loop timing.'))
    check()
    print('AWQ150_PLAN_FROZEN', len(ids), flush=True)


def check():
    plan = read(PLAN / 'manifest.json')
    assert digest(BASELINE) == plan['baseline_report_sha256']
    for rel, expected in plan['source_checks'].items():
        assert digest(ROOT / rel) == expected, rel
    for item in plan['batches']:
        assert digest(PLAN / ('batch_%02d.json' % item['batch'])) == item['sha256']
    kernel = read(ROOT / plan['kernel_report'])
    assert kernel['status'] == 'passed' and kernel['model_sha256'] == plan['model_sha256']
    assert (MODEL_DIR / 'model.safetensors').stat().st_size == read(MODEL_DIR / 'conversion_audit.json')['bytes']
    return plan


def summarize(state):
    plan = read(PLAN / 'manifest.json')
    baseline = read(BASELINE)
    base = {e['episode_id']: e for e in baseline['episodes']}
    episodes, calls, errors, peaks, complete_batches, pending = [], [], [], [], [], []
    for item in plan['batches']:
        i = item['batch']
        run = OUT / ('batch_%02d' % i)
        if (run / 'summary.json').exists():
            summary = read(run / 'summary.json')
            actual_ids = [e['episode_id'] for e in summary['episodes']]
            assert summary['status'] == 'complete' and len(actual_ids) == len(set(actual_ids)) == 50
            assert set(actual_ids) == set(item['episodes'])
            complete_batches.append(i)
        if (run / 'gpu_samples.csv').exists():
            for line in (run / 'gpu_samples.csv').read_text().splitlines():
                try:
                    peaks.append(float(line.split(',')[1]))
                except (IndexError, ValueError):
                    pass
        for eid in item['episodes']:
            folder = run / ('episode_' + eid)
            if not (folder / 'result.json').exists():
                if folder.exists():
                    pending.append(dict(batch=i, episode_id=eid, status='started_incomplete'))
                continue
            result = read(folder / 'result.json')
            rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
            assert len(rows) == result['steps'] <= 400 and all(r['step'] == j for j, r in enumerate(rows))
            expected_success = rows[-1]['action'] == 'STOP' and result['metrics']['distance_to_goal'] < 3
            assert bool(result['metrics']['success']) == expected_success
            actual_calls = [r for r in rows if r.get('model_called') is True]
            calls.extend(r['model_request_s'] for r in actual_calls)
            collisions = [j for j, r in enumerate(rows) if r['metrics']['collisions']['is_collision']]
            moves = [math.sqrt(sum((a-b)**2 for a, b in zip(r['before']['position'], r['after']['position']))) for r in rows]
            longest = streak = 0
            for row, movement in zip(rows, moves):
                streak = streak + 1 if row['action'] == 'MOVE_FORWARD' and movement < .02 else 0
                longest = max(longest, streak)
            success = bool(result['metrics']['success']) and result['terminal_stop_source'] == 'model'
            episodes.append(dict(result, batch=i, batch_complete=i in complete_batches,
                protocol_success=int(success), protocol_spl=result['metrics']['spl'] if success else 0.0,
                actual_model_calls=len(actual_calls), request_latency_s=stats([r['model_request_s'] for r in actual_calls]),
                collision_diagnostics=dict(first_collision_step=collisions[0]+1 if collisions else None,
                    later_movement_over_2cm=sum(d>.02 for d in moves[collisions[0]+1:]) if collisions else None,
                    longest_blocked_forward_streak=longest,
                    max_identical_response_count=max(collections.Counter(r['raw'] for r in actual_calls).values(), default=0)),
                steps_sha256=digest(folder / 'steps.jsonl'),
                fp16_success=base[eid]['metrics']['success'], fp16_spl=base[eid]['metrics']['spl']))
            errors.extend(dict(episode_id=eid, step=r['step']+1, error=r['error']) for r in rows if r.get('error'))
    common = [base[e['episode_id']] for e in episodes]
    paired = dict(count=len(episodes),
        fp16_sr=statistics.mean(e['metrics']['success'] for e in common) if common else None,
        fp16_spl=statistics.mean(e['metrics']['spl'] for e in common) if common else None,
        awq_sr=statistics.mean(e['protocol_success'] for e in episodes) if episodes else None,
        awq_spl=statistics.mean(e['protocol_spl'] for e in episodes) if episodes else None,
        gained_success=[e['episode_id'] for e in episodes if not e['fp16_success'] and e['protocol_success']],
        lost_success=[e['episode_id'] for e in episodes if e['fp16_success'] and not e['protocol_success']])
    report = dict(updated_local=now(), controller_state=state, planned_count=150,
        completed_episode_count=len(episodes), completed_batches=complete_batches, incomplete_episodes=pending,
        episodes=episodes, sr=paired['awq_sr'], spl=paired['awq_spl'], paired_comparison=paired,
        actual_request_latency_s=stats(calls), sampled_whole_card_peak_mib=max(peaks) if peaks else None,
        action_steps=sum(e['steps'] for e in episodes), errors=errors,
        model_stop_success_count=sum(e['protocol_success'] for e in episodes),
        fallback_stop_success_count=sum(e['metrics']['success']==1 and e['terminal_stop_source']=='error_fallback' for e in episodes),
        step_limit_count=sum(e['terminal_stop_source']=='step_limit' for e in episodes),
        collision_episode_count=sum(e['metrics']['collisions']['count']>0 for e in episodes),
        episode_loop_total_s=sum(e['elapsed_s'] for e in episodes), plan_sha256=digest(PLAN / 'manifest.json'),
        model_sha256=plan['model_sha256'], fp16_full=dict(count=150, sr=baseline['sr'], spl=baseline['spl'],
            actual_request_latency_s=baseline['actual_request_latency_s'],
            sampled_whole_card_peak_mib=baseline['sampled_whole_card_peak_mib']),
        notes=[plan['protocol'], plan['latency'], plan['memory'], plan['timing_note'],
               'Closed-loop inputs/trajectory lengths can diverge; request-latency comparison is workload-level, not fixed-input kernel speedup.',
               'All fixed tasks retained regardless of outcomes; 2cm collision threshold is diagnostic only.',
               'Only common completed episodes form the paired quality comparison; incomplete tasks explicitly listed.',
               'Model file sizes and whole-card sampled VRAM are different metrics; engine cache budgets are fractions of free memory.'])
    write(REPORT.with_suffix('.json'), report)
    def number(value, digits=3):
        return '待完成' if value is None else ('%.*f' % (digits, value))
    latency = report['actual_request_latency_s'] or {}
    lines = ['# AWQ与FP16同清单150任务对照', '', '更新时间：'+now(),
        '', '- 状态：'+state['status']+'；'+state.get('reason', ''),
        '- AWQ已完成%d/150条，整批%s；部分开始未完成任务%d条。' % (len(episodes), complete_batches, len(pending)),
        '- GPU算子核验已通过；本轮量化权重SHA256：'+plan['model_sha256'],
        '- 本次为固定开发子集，不是官方全量基准。', '',
        '|指标|原FP16全部150条|本轮AWQ已完成任务|', '|---|---:|---:|',
        '|任务数|150|%d|' % len(episodes),
        '|SR|54.00%%|%s%%|' % number(report['sr']*100 if episodes else None, 2),
        '|SPL|%.6f|%s|' % (baseline['spl'], number(report['spl'], 6)),
        '|真实模型请求中位秒|%.3f|%s|' % (baseline['actual_request_latency_s']['median'], number(latency.get('median'))),
        '|真实模型请求P95秒|%.3f|%s|' % (baseline['actual_request_latency_s']['p95'], number(latency.get('p95'))),
        '|真实模型请求次数|8338|%d|' % len(calls),
        '|整卡采样峰值MiB|11655|%s|' % number(report['sampled_whole_card_peak_mib'], 0),
        '|请求/解析错误|0|%d|' % len(errors), '',
        '**部分完成时，质量结论只比较下列同一任务子集，不把AWQ部分结果直接与FP16全部150条比较。**',
        '- 配对已完成任务：%d；FP16 SR=%s，SPL=%s；AWQ SR=%s，SPL=%s。' %
        (len(episodes), number(paired['fp16_sr'], 4), number(paired['fp16_spl'], 6), number(paired['awq_sr'], 4), number(paired['awq_spl'], 6)),
        '- AWQ新增成功：'+str(paired['gained_success']), '- AWQ丢失成功：'+str(paired['lost_success']),
        '- AWQ达到步限：%d；发生碰撞：%d。' % (report['step_limit_count'], report['collision_episode_count']),
        '- 模型主动STOP成功：%d；兜底STOP环境指标成功：%d（不计协议成功）。' %
        (report['model_stop_success_count'], report['fallback_stop_success_count']), '',
        '口径：真实请求只取model_called=true，排除动作队列0秒及每批2次预热。显存为200ms整卡采样，含桌面与服务缓存，非模型独占。',
        '闭环轨迹可能不同，延迟差不能单独解释为相同输入下的纯GPU加速。',
        '原FP16第一批有维护中断与验证前缀恢复；原结果保留，任务735循环时间缺少前缀IO，不能用于精确全程耗时比较。',
        '本轮原始证据：logs/overnight150_awq_20261104/batch_01至batch_03，逐步JSON和PNG均保留。', '',
        '|批次|任务|FP16成功|AWQ成功|FP16 SPL|AWQ SPL|AWQ步数|', '|---|---|---:|---:|---:|---:|---:|']
    lines += ['|%d|%s|%d|%d|%.6f|%.6f|%d|' % (e['batch'], e['episode_id'], e['fp16_success'], e['protocol_success'], e['fp16_spl'], e['protocol_spl'], e['steps']) for e in episodes]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def smoke():
    samples = read(ROOT / 'data/calibration/navigation_awq_v1/samples.json')
    records = []
    for sample in samples[:2]:
        messages = copy.deepcopy(sample['messages'])
        for message in messages:
            for item in message['content']:
                if item['type'] == 'image':
                    encoded = base64.b64encode((ROOT / item['image']).read_bytes()).decode()
                    item.clear()
                    item.update(type='image_url', image_url={'url': 'data:image/jpeg;base64,'+encoded})
        payload = dict(model=MODEL, messages=messages, max_tokens=512, temperature=.3, top_p=.95, seed=7, stream=False)
        started = time.monotonic()
        request = urllib.request.Request(ENDPOINT+'/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            answer = json.load(response)
        choice = answer['choices'][0]
        records.append(dict(sample_id=sample['id'], request_s=time.monotonic()-started, response=answer))
        write(OUT / 'service_smoke.json', records)
        assert choice['finish_reason'] == 'stop' and choice['message']['content'].strip()
        assert any(word in choice['message']['content'].lower() for word in ['forward', 'left', 'right', 'stop'])
    print('AWQ_SERVICE_SMOKE_PASSED', len(records), flush=True)


def terminate_owned(process):
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=10)
            return
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def main():
    check()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'controller.lock').write_text(str(os.getpid()))
    state = dict(status='starting', started_local=now(), planned_count=150, completed_count=0,
                 completed_batches=0, controller_pid=os.getpid(), model=MODEL, reason='')
    write(OUT / 'status.json', state)
    shutil.copyfile(PLAN / 'manifest.json', OUT / 'frozen_plan.json')
    service = child = None
    try:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', 23333)) == 0:
                raise RuntimeError('Port23333 occupied; preserve existing service, do not auto-kill it')
        with (OUT / 'model_service.log').open('w') as service_log:
            service = subprocess.Popen(['bash', 'scripts/start_simplevln_rl_awq_learning.sh'], cwd=str(ROOT),
                stdout=service_log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(service_pid=service.pid, status='loading_model')
        write(OUT / 'status.json', state)
        deadline = time.monotonic() + 900
        while True:
            if service.poll() is not None:
                raise RuntimeError('Model service exited during loading')
            try:
                with urllib.request.urlopen(ENDPOINT+'/models', timeout=5) as response:
                    available = json.load(response)
                assert any(m['id']==MODEL for m in available['data'])
                break
            except (OSError, AssertionError, ValueError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Model service did not become ready within15min')
                time.sleep(5)
        smoke()
        state.update(service_smoke_passed=True)
        summarize(state)
        durations = []
        for batch in range(1, 4):
            if shutil.disk_usage(ROOT).free < 50*1024**3:
                raise RuntimeError('Less than50GiB free; partial evidence preserved')
            run = OUT / ('batch_%02d' % batch)
            run.mkdir()
            state.update(status='running', current_batch=batch, batch_started_local=now(), progress=None)
            env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', PYTHONUNBUFFERED='1')
            env.pop('DISPLAY', None)
            started = time.monotonic()
            with (run / 'run.log').open('w') as log:
                child = subprocess.Popen(['bash', 'scripts/run_habitat_task_python.sh', 'checks/run_simplevln_diagnostic.py',
                    '--policy', 'memory', '--manifest', str(PLAN / ('batch_%02d.json' % batch)), '--output', str(run)],
                    cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state['evaluator_pid'] = child.pid
                while child.poll() is None:
                    if service.poll() is not None:
                        raise RuntimeError('Model service exited; stop evaluator and preserve partial evidence')
                    if time.monotonic()-started > 8*3600:
                        raise RuntimeError('Batch exceeded8h watchdog; partial evidence preserved')
                    if shutil.disk_usage(ROOT).free < 50*1024**3:
                        raise RuntimeError('Disk safety threshold reached; partial evidence preserved')
                    try:
                        state['progress'] = read(run / 'progress.json')
                    except (OSError, ValueError):
                        pass
                    state.update(updated_local=now(), batch_elapsed_s=time.monotonic()-started)
                    write(OUT / 'status.json', state)
                    time.sleep(15)
            if child.returncode != 0:
                raise RuntimeError('Evaluator exited%d; see batch run.log' % child.returncode)
            if not (run / 'summary.json').exists():
                raise RuntimeError('Evaluator ended without summary')
            durations.append(time.monotonic()-started)
            state.update(completed_batches=batch, completed_count=50*batch, batch_wall_s=durations)
            report = summarize(state)
            if report['errors']:
                raise RuntimeError('Request/parse errors observed; results preserved for diagnosis')
        state.update(status='completed', reason='All150 fixed episodes finished')
    except BaseException as error:
        state.update(status='failed', reason=str(error), traceback=traceback.format_exc())
    finally:
        for process in (child, service):
            try:
                terminate_owned(process)
            except Exception as error:
                state.setdefault('cleanup_errors', []).append(str(error))
        state.update(finished_local=now(), owned_service_stopped=service is None or service.poll() is not None)
        write(OUT / 'status.json', state)
        try:
            summarize(state)
        except Exception:
            (OUT / 'report_error.txt').write_text(traceback.format_exc())
        print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state['status']=='completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    options = parser.parse_args()
    if options.prepare:
        prepare()
    elif options.check_only:
        check()
        print('AWQ150_PREFLIGHT_OK')
    elif options.summarize_only:
        summarize(read(OUT / 'status.json'))
    else:
        raise SystemExit(main())
