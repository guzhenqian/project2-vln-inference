"""Bounded, sequential baseline controller. Uses the unchanged frozen evaluator."""
import collections
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'data/baseline/overnight200_fp16_20261031'
OUT = ROOT / 'logs/overnight200_fp16_20261031'
REPORT = ROOT / 'reports/benchmark_overnight200_20261031'
TZ = dt.timezone(dt.timedelta(hours=8))
DEADLINE = dt.datetime(2026, 11, 1, 8, tzinfo=TZ)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def now():
    return dt.datetime.now(TZ).isoformat()


def stats(values):
    if not values:
        return None
    v = sorted(values)
    i = (len(v)-1)*.95
    lo = int(i)
    p95 = v[lo] + (v[min(lo+1, len(v)-1)]-v[lo])*(i-lo)
    return dict(count=len(v), median=statistics.median(v), p95=p95,
                mean=statistics.mean(v), total=sum(v))


def summarize(state):
    episodes, calls, peaks, batches, errors = [], [], [], [], []
    for i in range(1, 5):
        run = OUT / state.get('run_directories', {}).get(str(i), 'batch_%02d' % i)
        manifest = read(PLAN / ('batch_%02d.json' % i))
        summary_path = run / 'summary.json'
        completed_batch = summary_path.exists()
        if completed_batch:
            summary = read(summary_path)
            assert summary['status'] == 'complete' and summary['episode_count'] == 50
            # Habitat groups episodes by scene to reduce reloads. Membership is fixed;
            # its deterministic execution order need not equal manifest listing order.
            actual_ids = [e['episode_id'] for e in summary['episodes']]
            assert len(set(actual_ids)) == 50 and set(actual_ids) == set(manifest['episodes'])
            batches.append(i)
            if summary['gpu_memory']['sampled_peak_mib'] is not None:
                peaks.append(summary['gpu_memory']['sampled_peak_mib'])
            if i == 1 and run.name != 'batch_01':
                for line in (OUT/'batch_01/gpu_samples.csv').read_text().splitlines():
                    try:
                        peaks.append(float(line.split(',')[1]))
                    except (ValueError, IndexError):
                        pass
        for eid in manifest['episodes']:
            folder = run / ('episode_' + eid)
            if not (folder / 'result.json').exists():
                continue
            e = read(folder / 'result.json')
            rows = [json.loads(x) for x in (folder / 'steps.jsonl').read_text().splitlines()]
            assert len(rows) == e['steps'] <= 400
            assert all(r['step'] == j for j,r in enumerate(rows))
            assert bool(e['metrics']['success']) == (rows[-1]['action']=='STOP' and e['metrics']['distance_to_goal'] < 3)
            e = dict(e, batch=i, batch_complete=completed_batch)
            actual = [r for r in rows if r['model_called']]
            e['actual_model_calls'] = len(actual)
            e['request_latency_s'] = stats([r['model_request_s'] for r in actual])
            calls.extend(r['model_request_s'] for r in actual)
            collisions = [j for j,r in enumerate(rows) if r['metrics']['collisions']['is_collision']]
            movement = [math.sqrt(sum((a-b)**2 for a,b in zip(r['before']['position'],r['after']['position']))) for r in rows]
            streak = longest = 0
            for r, d in zip(rows, movement):
                streak = streak+1 if r['action']=='MOVE_FORWARD' and d<.02 else 0
                longest = max(streak,longest)
            e['collision_diagnostics'] = dict(first_collision_step=collisions[0]+1 if collisions else None,
                later_movement_over_2cm=sum(d>.02 for d in movement[collisions[0]+1:]) if collisions else None,
                longest_blocked_forward_streak=longest,
                max_identical_response_count=max(collections.Counter(r['raw'] for r in actual).values(),default=0))
            e['steps_sha256'] = hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest()
            e['scene'] = next(x['scene'] for x in read(PLAN/'manifest.json')['sample_details'] if x['episode_id']==eid)
            errors.extend(dict(episode=eid, step=r['step']+1, error=r['error']) for r in rows if r.get('error'))
            episodes.append(e)
    report = dict(updated_local=now(), controller_state=state, planned_count=200,
        completed_batches=batches, complete_batch_episode_count=50*len(batches),
        completed_episode_count=len(episodes), episodes=episodes,
        sr=statistics.mean(e['metrics']['success'] for e in episodes) if episodes else None,
        spl=statistics.mean(e['metrics']['spl'] for e in episodes) if episodes else None,
        model_stop_success_count=sum(e['metrics']['success']==1 and e['terminal_stop_source']=='model' for e in episodes),
        fallback_stop_success_count=sum(e['metrics']['success']==1 and e['terminal_stop_source']=='error_fallback' for e in episodes),
        errors=errors, action_steps=sum(e['steps'] for e in episodes), actual_request_latency_s=stats(calls),
        episode_loop_total_s=sum(e['elapsed_s'] for e in episodes),
        sampled_whole_card_peak_mib=max(peaks) if peaks else None,
        plan_sha256=hashlib.sha256((PLAN/'manifest.json').read_bytes()).hexdigest(),
        notes=['One instruction per distinct route; excludes old12 routes; development subset, not official full1839 score.',
               'FP16 unquantized; policy/parser/400step/STOP<3m rules unchanged; no collision early stopping.',
               'Each50 batch has2 warmup requests excluded. Request latency only model_called rows.',
               'Whole-card sampled memory includes desktop. Episode loops exclude reset/warmup/model startup.',
               'Partial batch episodes remain visible and are not silently dropped to achieve a round number.',
               '2cm diagnostic threshold is observational only, not a new termination or success rule.',
               'Habitat GROUP_BY_SCENE=true deterministically regroups each frozen batch; no task membership changes.',
               'Batch1 was interrupted during controller maintenance. Exact saved action prefix is replayed with pixel/pose checks; completed outcomes retained. Timings span sessions; partial-prefix loop IO not measured. See timing_note per episode.'])
    REPORT.parent.mkdir(exist_ok=True)
    write(REPORT.with_suffix('.json'), report)
    lines = ['# 夜间未量化FP16基线', '', '更新时间：'+now(), '',
        '- 状态：'+state['status']+'；原因：'+state.get('reason',''),
        '- 计划200条；完成整批{}条；全部已完成任务{}条（含未完成批次的已完成任务）。'.format(50*len(batches),len(episodes)),
        '- 模型SimpleVLN-Qwen3VL-4B-RL-FP16，seed7；最多400步，主动STOP且终距<3m成功。',
        '- 固定11场景、不同路线与路线长度分层抽样；旧12路线排除。不会按成败删任务。',
        '- 本次是开发基线，不是官方全量成绩。今晚未进行量化。', '']
    if episodes:
        lines += ['- SR {:.2%}；SPL {:.6f}；模型主动STOP成功{}条；兜底STOP成功{}条。'.format(report['sr'],report['spl'],report['model_stop_success_count'],report['fallback_stop_success_count']),
            '- {}原子动作；{}次真实模型请求；请求中位{:.3f}s，P95 {:.3f}s。'.format(report['action_steps'],len(calls),report['actual_request_latency_s']['median'],report['actual_request_latency_s']['p95']),
            '- 请求/解析错误{}次；整卡采样峰值{}。'.format(len(errors),str(report['sampled_whole_card_peak_mib'])+'MiB' if peaks else '待本批完成后汇总'),
            '- 任务循环累计{:.1f}分钟，不含reset/预热/模型启动。'.format(report['episode_loop_total_s']/60), '']
    lines += ['|批次|任务|场景|步数|成功|SPL|耗时秒|碰撞数|最长受阻前进序列|', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for e in episodes:
        lines.append('|{}|{}|{}|{}|{}|{:.4f}|{:.1f}|{}|{}|'.format(e['batch'],e['episode_id'],e['scene'].split('/')[1],e['steps'],int(e['metrics']['success']),e['metrics']['spl'],e['elapsed_s'],e['metrics']['collisions']['count'],e['collision_diagnostics']['longest_blocked_forward_streak']))
    lines += ['', '原始证据：logs/overnight200_fp16_20261031/batch_01至batch_04；逐任务JSON/PNG、steps.jsonl、环境配置、manifest和provenance均保留。',
              '详细调用统计、错误、碰撞恢复与证据哈希见同名JSON。每批summary原有混合0秒请求统计不作性能结论。',
              '缩减数量只根据固定批次顺序和耗时预算；异常导致的部分任务必须单列，不包装成完成整批。', '']
    REPORT.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Resume supervises an already-running evaluator; it never replays completed tasks.
    lock = OUT/'controller.lock'
    status_path = OUT/'status.json'
    recover = '--recover' in sys.argv
    resume = '--resume' in sys.argv or recover
    if resume:
        previous_pid = int(lock.read_text())
        try:
            os.kill(previous_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Previous controller is still alive')
        state = read(status_path)
        assert state['status'] == 'running', 'Resume only the existing running batch'
        first_batch = state['current_batch']
        adopted_pid = state['evaluator_pid']
        proc_cmd = Path('/proc/{}/cmdline'.format(adopted_pid))
        if recover:
            assert not proc_cmd.exists(), 'Do not recover while evaluator is alive'
            state['recovery_prefix'] = str(OUT/('batch_%02d'%first_batch))
            state.setdefault('run_directories', {})[str(first_batch)] = 'batch_%02d_resumed' % first_batch
        elif not (OUT/('batch_%02d'%first_batch)/'summary.json').exists():
            cmdline = proc_cmd.read_bytes().replace(b'\0',b' ').decode()
            assert 'run_simplevln_diagnostic.py' in cmdline and str(OUT) in cmdline
        state.update(pid=os.getpid(), resumed_local=now(), previous_controller_pid=previous_pid)
        lock.write_text(str(os.getpid()))
    else:
        with lock.open('x') as f:
            f.write(str(os.getpid()))
        state = dict(status='starting', started_local=now(), deadline_local=DEADLINE.isoformat(),
                     planned_count=200, completed_batches=0, pid=os.getpid())
        first_batch = 1
        adopted_pid = None
    write(status_path,state)
    child = None
    try:
        shutil.copyfile(PLAN/'manifest.json', OUT/'frozen_plan.json')
        with urllib.request.urlopen('http://127.0.0.1:23333/v1/models', timeout=10) as r:
            assert any(m['id']=='SimpleVLN-Qwen3VL-4B-RL-FP16' for m in json.load(r)['data'])
        durations = state.get('batch_wall_s', [])
        for batch in range(first_batch,5):
            recovering = recover and batch == first_batch
            adopting = resume and not recovering and batch == first_batch
            remaining = (DEADLINE-dt.datetime.now(TZ)).total_seconds()
            estimate = 1.3*max(durations) if durations else 0
            if not adopting and not recovering and remaining <= estimate:
                state.update(status='completed_reduced', reason='Insufficient budget for another fixed50 batch', next_batch_estimate_s=estimate)
                break
            if not adopting and shutil.disk_usage(ROOT).free < 50*1024**3:
                state.update(status='stopped_disk_budget', reason='Less than50GiB free; preserve all saved evidence')
                break
            run = OUT/state.get('run_directories', {}).get(str(batch), 'batch_%02d'%batch)
            if not adopting:
                run.mkdir(exist_ok=False)
                if not recovering:
                    state.update(status='running', current_batch=batch, batch_started_local=now(), reason='')
                else:
                    state.update(recovery_started_local=now(), reason='Exact-prefix recovery after controller maintenance interruption')
            write(status_path,state)
            started = time.monotonic() - ((dt.datetime.now(TZ)-dt.datetime.fromisoformat(state['batch_started_local'])).total_seconds() if adopting or recovering else 0)
            env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE='1', PYTHONUNBUFFERED='1')
            env.pop('DISPLAY',None)
            log = None
            child = None
            if not adopting:
                log = (run/'run.log').open('w')
                command = ['bash','scripts/run_habitat_task_python.sh','checks/resume_overnight_batch.py' if recovering else 'checks/run_simplevln_diagnostic.py',
                    '--policy','memory','--manifest',str(PLAN/('batch_%02d.json'%batch)),
                    '--output',str(run)]
                if recovering:
                    command.extend(['--prefix', state['recovery_prefix']])
                child = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
                state['evaluator_pid'] = child.pid
            try:
                while (child.poll() is None if child else Path('/proc/{}'.format(adopted_pid)).exists() and not (run/'summary.json').exists()):
                    state['updated_local'] = now()
                    state['batch_elapsed_s'] = time.monotonic()-started
                    try:
                        state['progress'] = read(run/'progress.json')
                    except (FileNotFoundError,ValueError):
                        pass
                    write(status_path,state)
                    if state['batch_elapsed_s'] > 8*3600:
                        if child:
                            child.terminate()
                        else:
                            os.kill(adopted_pid, 15)
                        raise RuntimeError('Batch exceeded8h watchdog; partial results preserved')
                    time.sleep(15)
            finally:
                if log:
                    log.close()
            if child and child.returncode != 0:
                raise RuntimeError('Evaluator exited {}; inspect {}'.format(child.returncode,run/'run.log'))
            if not (run/'summary.json').exists():
                raise RuntimeError('Evaluator exited without summary; partial evidence preserved')
            durations.append(time.monotonic()-started)
            state.update(completed_batches=batch, completed_count=50*batch, batch_wall_s=durations)
            report = summarize(state)
            if report['errors']:
                state.update(status='stopped_errors',reason='Request/parse errors observed; preserving all results for review')
                break
        else:
            state.update(status='completed',reason='All200 fixed episodes finished')
    except Exception as error:
        state.update(status='failed',reason=str(error),traceback=traceback.format_exc())
    finally:
        state['finished_local'] = now()
        write(status_path,state)
        try:
            summarize(state)
        except Exception:
            (OUT/'report_error.txt').write_text(traceback.format_exc())
        print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    if '--summarize-only' in sys.argv:
        summarize(read(OUT/'status.json'))
    else:
        main()
