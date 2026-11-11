"""Export a completed navigation run as an audited benchmark; no inference."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def percentile(values, q):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Output stem, without extension')
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    expected = [str(x) for x in manifest['episodes']]
    assert summary['status'] == 'complete'
    assert [e['episode_id'] for e in summary['episodes']] == expected
    records, episodes, evidence = [], [], {}
    for eid in expected:
        path = run / ('episode_' + eid) / 'steps.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        result = next(e for e in summary['episodes'] if e['episode_id'] == eid)
        assert len(rows) == result['steps'] <= manifest['max_episode_steps']
        assert all(r['step'] == i for i, r in enumerate(rows))
        assert all('model_called' in r for r in rows), 'Cannot distinguish calls from queued actions'
        success = rows[-1]['action'] == 'STOP' and result['metrics']['distance_to_goal'] < manifest['success_distance_m']
        assert int(success) == result['metrics']['success']
        assert abs(result['metrics']['spl'] - int(success) * result['start_distance_m'] /
                   max(result['start_distance_m'], result['path_length_m'])) < 1e-5
        episodes.append(dict(result, model_calls=sum(r['model_called'] for r in rows)))
        records.extend(rows)
        evidence[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('manifest.json', 'summary.json'):
        evidence[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    calls = [r['model_request_s'] for r in records if r['model_called']]
    assert len(records) == summary['decision_count']
    report = dict(
        run=str(run.relative_to(ROOT)), protocol=manifest, evidence_sha256=evidence,
        episode_count=len(episodes), action_steps=len(records), model_calls=len(calls),
        queued_action_steps=len(records)-len(calls), episodes=episodes,
        sr=statistics.mean(e['metrics']['success'] for e in episodes),
        spl=statistics.mean(e['metrics']['spl'] for e in episodes),
        episode_loop_total_s=sum(e['elapsed_s'] for e in episodes),
        actual_request_latency_s=dict(median=statistics.median(calls), p95=percentile(calls, .95),
                                      mean=statistics.mean(calls), total=sum(calls)),
        simulator_steps_total_s=sum(r['sim_step_s'] for r in records),
        error_count=sum(bool(r.get('error')) for r in records), gpu_memory=summary['gpu_memory'],
        notes=[
            'Local six-episode teaching/diagnostic subset; not the full official benchmark.',
            'Success requires STOP and geodesic distance strictly below 3m. Entering the radius alone is insufficient.',
            '400 counts primitive environment actions, including rotations, collisions, and STOP, not model calls.',
            'No early termination on collisions or stalled progress in this protocol.',
            'Loop time excludes env.reset, model service startup and two warmup requests; it is not full wall time.',
            'Actual request latency excludes queued actions and warmups; includes HTTP/backend work, not pure GPU time.',
            'Original summary latency mixes model calls and zero-time queued actions; retained unchanged.',
            'GPU figure is whole-card sampled peak, including desktop; not model-only allocated peak.',
            'Infrastructure/parse fallback STOP is separately recorded; it must not be reported as a model STOP success.',
        ])
    assert abs(report['sr'] - summary['sr']) < 1e-8
    assert abs(report['spl'] - summary['spl']) < 1e-8
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = [
        '# 导航教学6任务 benchmark', '',
        '由完成运行的原始日志离线生成；不启动模型，不重新评测。', '',
        '## 固定协议', '',
        '- 模型：{}；精度：{}；种子：{}。'.format(manifest['model'], manifest['dtype'], manifest['seed']),
        '- 数据：{} / {}；固定任务：{}。'.format(manifest['dataset'], manifest['split'], ', '.join(expected)),
        '- 每条最多400个原子动作步。前进25cm，转向15°；碰撞、旋转和STOP均计步。',
        '- 成功：主动STOP时，沿可通行路径到目标的距离严格小于3m。',
        '- 提前STOP但距离≥3m：立即结束，失败。进入3m后又离开：不能算成功。',
        '- 达到400步仍未有效STOP：结束，失败；第400步若有效STOP则可以成功。',
        '- 当前不因连续碰撞或原地不动提前结束，避免改变固定对照协议。',
        '- 请求/解析异常记录错误并执行兜底STOP，任务仍保留在分母；本次0错误。',
        '- 本组3场景、短路线筛选的小样本仅用于教学与诊断，不代表官方全量精度。', '',
        '## 本次结果', '',
        '|任务|动作步|模型请求|耗时（秒）|终距（米）|结果|',
        '|---|---:|---:|---:|---:|---|',
    ]
    for e in episodes:
        lines.append('|{}|{}|{}|{:.1f}|{:.3f}|{}|'.format(e['episode_id'], e['steps'], e['model_calls'],
            e['elapsed_s'], e['metrics']['distance_to_goal'], '成功' if e['metrics']['success'] else '失败'))
    lines.extend(['', '- SR：{:.1%}；SPL：{:.6f}。'.format(report['sr'], report['spl']),
        '- 总动作{}步，真实模型请求{}次，队列动作{}步。'.format(len(records), len(calls), report['queued_action_steps']),
        '- 任务循环累计{:.1f}秒；不含环境reset、预热及模型服务启动。'.format(report['episode_loop_total_s']),
        '- 真实模型请求中位{:.3f}秒，P95 {:.3f}秒；不含队列动作0秒值及2次预热。'.format(statistics.median(calls), percentile(calls,.95)),
        '- 整卡采样峰值{} MiB；包括桌面等占用，不等于模型专属显存。'.format(summary['gpu_memory']['sampled_peak_mib']),
        '- 原summary中的请求中位数0是混入队列动作造成，不能据此宣称推理延迟接近0。', '',
        '## 文件与复用', '',
        '- 执行器：checks/run_simplevln_diagnostic.py；协议：运行目录manifest.json。',
        '- 逐步证据：episode_*/steps.jsonl、PNG；聚合：summary.json。原始证据未修改。',
        '- 同名JSON保存完整协议、逐条结果、真实请求统计及证据SHA256。',
        '- 60条任务需先固定新的清单，沿用相同规则；不要按结果删除失败任务。', '',
        '```bash',
        'python checks/export_navigation_benchmark.py --run {} --output {}'.format(run.relative_to(ROOT).as_posix(), output.relative_to(ROOT).as_posix()),
        '```', ''])
    output.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('episode_count','action_steps','model_calls','queued_action_steps',
        'sr','spl','episode_loop_total_s','actual_request_latency_s')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
