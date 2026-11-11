"""Export complete same150 FP16 / DART75 / DART50 quality and request metrics."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from navigation_dart50_support import ROOT, read, write, digest, require_c75_complete


def success(row):
    return int(bool(row['metrics']['success']) and row['terminal_stop_source'] == 'model')


def make_comparison(c75_path, c50_run):
    require_c75_complete(c75_path)
    c75 = read(c75_path)
    c50 = read(c50_run / 'report.json')
    if not (c50['status'] == 'complete' and c50['completed_episode_count'] == 150
            and c50['all_runtime_audits_passed'] is True and c50['error_count'] == 0
            and c50['completed_batches'] == [1, 2, 3]):
        raise ValueError('C50 report is not complete and audited')
    baseline_path = ROOT / 'reports/benchmark_overnight200_20261031.json'
    baseline = read(baseline_path)
    reports = {'FP16': baseline, 'FP16_DART75': c75, 'FP16_DART50': c50}
    originals = [read(ROOT / f'data/baseline/overnight200_fp16_20261031/batch_{i:02d}.json') for i in (1, 2, 3)]
    order = [str(eid) for batch in originals for eid in batch['episodes']]
    maps = {name: {str(row['episode_id']): row for row in report['episodes']} for name, report in reports.items()}
    for name, mapping in maps.items():
        if len(mapping) != len(reports[name]['episodes']) or set(mapping) != set(order) or len(mapping) != 150:
            raise ValueError('Task identity/denominator mismatch: ' + name)
    metrics = {}
    for name, report in reports.items():
        rows = maps[name].values()
        passed = sum(success(row) for row in rows)
        spl = sum(float(row['metrics']['spl']) if success(row) else 0 for row in rows) / 150
        latency = report['actual_request_latency_s']
        metrics[name] = {'completed': 150, 'success_count': passed, 'sr': passed / 150, 'spl': spl,
                         'actual_model_calls': latency['count'], 'request_median_s': latency['median'],
                         'request_p95_s': latency['p95'],
                         'request_total_s': latency['total'],
                         'whole_card_peak_mib': report['sampled_whole_card_peak_mib']}
    a, b = maps['FP16_DART75'], maps['FP16_DART50']
    gains = [eid for eid in order if success(b[eid]) and not success(a[eid])]
    losses = [eid for eid in order if success(a[eid]) and not success(b[eid])]
    per_episode = [{'episode_id': eid, 'trajectory_id': b[eid]['trajectory_id'],
                    **{name + '_success': success(mapping[eid]) for name, mapping in maps.items()},
                    **{name + '_spl': float(mapping[eid]['metrics']['spl']) if success(mapping[eid]) else 0
                       for name, mapping in maps.items()}} for eid in order]
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
            'same_task_count': 150, 'metrics': metrics,
            'keep50_vs_keep75': {'gained_success': gains, 'lost_success': losses,
                                'sr_change_percentage_points': (metrics['FP16_DART50']['sr'] - metrics['FP16_DART75']['sr']) * 100,
                                'spl_change': metrics['FP16_DART50']['spl'] - metrics['FP16_DART75']['spl'],
                                'request_median_reduction_percent': (1 - metrics['FP16_DART50']['request_median_s'] / metrics['FP16_DART75']['request_median_s']) * 100,
                                'request_total_increase_percent': (metrics['FP16_DART50']['request_total_s'] / metrics['FP16_DART75']['request_total_s'] - 1) * 100},
            'episodes': per_episode,
            'source_timing_context': {
                name: {'notes': report.get('notes', []),
                       'recovery': report.get('recovery'),
                       'warmup_requests_excluded': report.get('warmup_requests_excluded'),
                       'episode_timing_notes': [
                           {'episode_id': str(row['episode_id']),
                            'prefix_reused_steps': row.get('prefix_reused_steps', 0),
                            'timing_note': row['timing_note']}
                           for row in report['episodes'] if row.get('timing_note')]}
                for name, report in reports.items()},
            'sources': {str(path.relative_to(ROOT)): digest(path) for path in
                        (baseline_path, c75_path, c50_run / 'report.json')},
            'notes': ['Same frozen150 episodes, seed7, max400 primitive actions, unchanged author policy/parser.',
                      'Model STOP and distance<3m defines success; all failures remain in denominator.',
                      'Only model_called=true request latencies; queue0s and all warmups, including additional recovery warmups, excluded.',
                      'C75/C50 use eager, complete prefill3072, batch1, fixed64x64 KV blocks and FP16 with noAWQ.',
                      'Original FP16 uses graph/1024 prefill; it is not a configuration-matched latency control.',
                      'Closed-loop paths, prompts, output lengths and request counts can differ; latency ratios are not pure kernel/algorithm speedups.',
                      'Request total is the sum of real HTTP request latencies, not end-to-end navigation wall-clock time; excludes simulation, image/log IO and downtime.',
                      'Memory is200ms whole-card sampling including desktop/cache; not model-exclusive and may miss instantaneous peaks.',
                      'Original FP16 and C75 had interrupted recovery; preserve their timing limitations and original evidence.',
                      'Fixed KV allocation is not reduced by token pruning; no claim of allocated cache savings.',
                      'One run per setting; no statistical-significance claim.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--c75-report', type=Path, required=True)
    parser.add_argument('--c50-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = make_comparison(args.c75_report.resolve(), args.c50_run.resolve())
    write(args.output, result, exclusive=True)
    rows = result['metrics']
    names = ['FP16', 'FP16_DART75', 'FP16_DART50']
    lines = ['# 同清单150条：FP16、DART保留75%、DART保留50%', '',
             '三个设置均为同一批150条；两档DART的运行期审计通过。', '',
             '| 指标 | 原FP16 | DART75% | DART50% |', '|---|---:|---:|---:|']
    for title, key, format_value in (
        ('成功数 /150', 'success_count', lambda x: str(x)),
        ('SR', 'sr', lambda x: f'{x:.2%}'), ('SPL', 'spl', lambda x: f'{x:.6f}'),
        ('真实请求数', 'actual_model_calls', str),
        ('请求中位延迟 /秒', 'request_median_s', lambda x: f'{x:.3f}'),
        ('请求P95 /秒', 'request_p95_s', lambda x: f'{x:.3f}'),
        ('真实请求累计耗时 /小时', 'request_total_s', lambda x: f'{x / 3600:.3f}'),
        ('整卡采样峰值 /MiB', 'whole_card_peak_mib', lambda x: f'{x:.0f}')):
        lines.append('| ' + title + ' | ' + ' | '.join(format_value(rows[name][key]) for name in names) + ' |')
    pair = result['keep50_vs_keep75']
    lines += ['', f"50%相对75%新增成功{len(pair['gained_success'])}例，丢失成功{len(pair['lost_success'])}例。", '',
              f"本批150例中，50%相对75%的成功率变化为{pair['sr_change_percentage_points']:+.2f}个百分点；单次请求中位延迟降低{pair['request_median_reduction_percent']:.2f}%，但真实请求累计耗时增加{pair['request_total_increase_percent']:.2f}%。", '',
              '延迟仅统计真实模型请求，排除队列零秒及全部预热，包括恢复额外产生的预热；不同轨迹与回答长度仍会影响闭环请求耗时。', '',
              '累计耗时为真实HTTP请求延迟求和，不是导航总墙钟时间，不含仿真、图片与日志IO或中断停机时间。', '',
              '75%与50%保持相同执行配置；原FP16为graph/1024分块，两档DART为eager/3072完整prefill，不能将差值称为纯算法加速。', '',
              '显存为200ms整卡采样，包含桌面、缓存和激活；固定64个KV blocks并未因裁剪减少分配。原FP16和75%恢复记录的计时限制保留。', '',
              '75%恢复前后的请求各计一次；案例1550的循环耗时缺少原前缀图片和日志IO，是下界，不用于本表请求延迟的计算。原恢复说明与逐例计时备注完整保存在同名JSON。', '',
              '逐任务结果及源文件SHA见同名JSON。']
    with args.output.with_suffix('.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('DART75_VS_DART50_COMPARISON_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
