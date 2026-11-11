"""Export the completed, identical150 A/B/C75/D75 runs without model calls."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from navigation_dart50_support import ROOT, read, write, digest, require_c75_complete
from navigation_awq_dart_profiles import served_model_name

SOURCES = {
    'A': ROOT / 'reports/benchmark_overnight200_20261031.json',
    'B': ROOT / 'reports/benchmark_awq150_vs_fp16_20261104.json',
    'C': ROOT / 'logs/navigation_dart150_keep75_resumed_20261108T135623640378Z_99f2e6a6/report.json',
}
LABELS = {'A': 'FP16基线', 'B': 'AWQ', 'C': 'FP16＋DART75%', 'D': 'AWQ＋DART75%'}

def succeeded(row):
    return int(bool(row['metrics']['success']) and row['terminal_stop_source'] == 'model')

def make_comparison(d_run):
    sources = dict(SOURCES, D=Path(d_run) / 'report.json')
    require_c75_complete(sources['C'])
    reports = {key: read(path) for key, path in sources.items()}
    d = reports['D']
    if not (d['status'] == 'complete' and d['completed_batches'] == [1, 2, 3]
            and d['all_runtime_audits_passed'] is True and d['error_count'] == 0
            and d['controller_state']['model'] == served_model_name(.75)):
        raise ValueError('D75 is not a completed audited AWQ+DART150 run')
    audits = d['runtime_audits']
    if not (len(audits) == 3 and [item['batch'] for item in audits] == [1, 2, 3]
            and all(item['audit']['passed'] is True and item['audit']['warmup_requests'] == 2
                    and item['audit']['runtime_trace_count'] == item['audit']['actual_model_calls'] + 2
                    for item in audits)
            and sum(item['audit']['actual_model_calls'] for item in audits) == d['actual_request_latency_s']['count']):
        raise ValueError('D75 runtime evidence does not reconcile with all real requests')
    manifests = [read(ROOT / f'data/baseline/overnight200_fp16_20261031/batch_{i:02d}.json') for i in (1, 2, 3)]
    original = {str(e): str(t) for m in manifests for e, t in zip(m['episodes'], m['trajectory_ids'])}
    if len(original) != 150:
        raise ValueError('Original membership is not exact150')
    maps, metrics = {}, {}
    for key, report in reports.items():
        rows = report['episodes']
        mapping = {str(row['episode_id']): row for row in rows}
        if (report['completed_episode_count'] != 150 or len(rows) != len(mapping) or len(rows) != 150
                or {eid: str(row['trajectory_id']) for eid, row in mapping.items()} != original
                or report.get('errors') or report.get('error_count', 0)):
            raise ValueError('Incomplete/error/mismatched task evidence: ' + key)
        maps[key] = mapping
        latency = report['actual_request_latency_s']
        metrics[key] = {'label': LABELS[key], 'completed': 150,
                        'success_count': sum(succeeded(row) for row in rows),
                        'sr': sum(succeeded(row) for row in rows) / 150,
                        'spl': sum(float(row['metrics']['spl']) if succeeded(row) else 0 for row in rows) / 150,
                        'actual_model_calls': latency['count'], 'request_median_s': latency['median'],
                        'request_p95_s': latency['p95'], 'request_total_s': latency['total'],
                        'whole_card_peak_mib': report['sampled_whole_card_peak_mib']}
    pairs = {}
    for key in ('A', 'B', 'C'):
        pairs['D_vs_' + key] = {
            'gained_success': [eid for eid in original if succeeded(maps['D'][eid]) and not succeeded(maps[key][eid])],
            'lost_success': [eid for eid in original if succeeded(maps[key][eid]) and not succeeded(maps['D'][eid])],
            'sr_change_percentage_points': (metrics['D']['sr'] - metrics[key]['sr']) * 100,
            'spl_change': metrics['D']['spl'] - metrics[key]['spl'],
            'request_median_change_percent': (metrics['D']['request_median_s'] / metrics[key]['request_median_s'] - 1) * 100,
            'request_total_change_percent': (metrics['D']['request_total_s'] / metrics[key]['request_total_s'] - 1) * 100,
        }
    return {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'same_task_count': 150, 'metrics': metrics, 'paired_comparisons': pairs,
            'episodes': [{'episode_id': eid, 'trajectory_id': trajectory,
                          **{key + '_success': succeeded(mapping[eid]) for key, mapping in maps.items()},
                          **{key + '_spl': float(mapping[eid]['metrics']['spl']) if succeeded(mapping[eid]) else 0
                             for key, mapping in maps.items()}} for eid, trajectory in original.items()],
            'sources': {path.relative_to(ROOT).as_posix(): digest(path) for path in sources.values()},
            'source_notes': {key: report.get('notes', []) for key, report in reports.items()},
            'c_recovery': reports['C'].get('recovery'),
            'd_runtime_audits': [{'batch': item['batch'], 'passed': item['audit']['passed'],
                                  'actual_model_calls': item['audit']['actual_model_calls'],
                                  'warmup_requests': item['audit']['warmup_requests'],
                                  'runtime_trace_count': item['audit']['runtime_trace_count']}
                                 for item in d['runtime_audits']],
            'notes': [
                'Identical150 episode/trajectory identities; original author policy/parser, seed7, max400 primitive actions, model STOP and distance<3m.',
                'A/B graph/1024 prefill differ from C/D eager/full3072. D additionally records144 native AWQ module invocations per forward; latency is not a pure algorithm or matched-kernel speedup.',
                'AWQ changes activations and may change DART retained indices. Same75% per-image budget does not mean the same selected tokens or trajectories.',
                'Request timing includes model_called=true only; excludes queued0s and all warmups. Total is summed HTTP time, not end-to-end navigation wall-clock time.',
                'Memory is whole-card200ms sampling including desktop/cache/activations. Fixed64x64 KV allocation remains unchanged by DART.',
                'A/C interrupted-recovery timing caveats are preserved in source notes; saved requests are counted once.',
                'One run per setting on the fixed150 development subset; not full1839 evaluation or statistical significance evidence.',
            ]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = make_comparison(args.d_run.resolve())
    write(args.output, result, exclusive=True)
    metrics = result['metrics']
    lines = ['# A/B/C/D同150条导航实验对比', '',
             'A：FP16；B：AWQ；C：FP16＋DART保留75%；D：AWQ＋DART保留75%。四组均完整150条，使用相同案例和任务协议。', '',
             '| 指标 | A：FP16 | B：AWQ | C：DART75% | D：AWQ＋DART75% |', '|---|---:|---:|---:|---:|']
    for title, field, render in (
        ('成功数 /150', 'success_count', str), ('SR', 'sr', lambda x: f'{x:.2%}'),
        ('SPL', 'spl', lambda x: f'{x:.6f}'), ('真实请求数', 'actual_model_calls', str),
        ('请求中位延迟 /秒', 'request_median_s', lambda x: f'{x:.3f}'),
        ('请求P95 /秒', 'request_p95_s', lambda x: f'{x:.3f}'),
        ('真实请求累计耗时 /小时', 'request_total_s', lambda x: f'{x / 3600:.3f}'),
        ('整卡采样峰值 /MiB', 'whole_card_peak_mib', lambda x: f'{x:.0f}')):
        lines.append('| ' + title + ' | ' + ' | '.join(render(metrics[key][field]) for key in 'ABCD') + ' |')
    for key in ('A', 'B', 'C'):
        pair = result['paired_comparisons']['D_vs_' + key]
        lines += ['', f"D相对{key}：新增成功{len(pair['gained_success'])}例，丢失成功{len(pair['lost_success'])}例，SR变化{pair['sr_change_percentage_points']:+.2f}个百分点，SPL变化{pair['spl_change']:+.6f}。"]
    lines += ['', 'D的真实运行审计同时检查原生AWQ算子执行与每图300→225视觉token、DeepStack、位置编码和KV一致性。', '',
              'A/B使用graph/1024分块prefill，C/D使用eager/3072完整prefill；D还有AWQ执行记录开销。因此表中延迟是各组闭环实测，不能直接解释为纯算法加速或严格匹配配置的交互效应。', '',
              '请求累计时间仅为真实HTTP请求求和，排除预热和队列零秒，不含仿真、图片/日志IO及中断停机；原A/C恢复计时限制保留。不同轨迹、请求数和回答长度会影响性能。', '',
              '显存为200ms整卡采样，包含桌面和缓存；DART没有减少固定KV池的分配。每档只运行一次，不作统计显著性结论。', '',
              '逐案例结果、配对变化、源报告SHA和运行审计摘要见同名JSON。']
    with args.output.with_suffix('.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print('ABCD_SAME150_EXPORTED', args.output)

if __name__ == '__main__':
    main()
