"""Independently recount saved stage2 evidence and publish compact comparisons.

Run only after the corresponding GPU stage has stopped. This reads evidence;
it never starts inference, changes a roster, or overwrites an earlier delivery.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'logs/navigation_stage2_sequence_20261109'
GROUPS = ('fp16', 'awq', 'gptq', 'fastv75', 'fastv50', 'sparsevlm75',
          'sparsevlm50', 'dart75', 'dart50')
NAMES = dict(fp16='FP16基线', awq='AWQ', gptq='GPTQ', fastv75='FastV 75%',
             fastv50='FastV 50%', sparsevlm75='SparseVLM-v1单层无合并 75%',
             sparsevlm50='SparseVLM-v1单层无合并 50%', dart75='DART 75%', dart50='DART 50%')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify(row, expected):
    assert row['status'] == 'complete' and row['case_count'] == len(expected)
    by_id = {str(e['episode_id']): e for e in row['episodes']}
    assert len(by_id) == len(row['episodes']) == len(expected)
    assert {e: str(v['trajectory_id']) for e, v in by_id.items()} == expected
    values, recount, memory, summaries = [], {}, [], {}
    for name, expected_hash in row['step_files_sha256'].items():
        path = ROOT / name
        assert sha(path) == expected_hash, name
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        eid = str(records[0]['episode_id'])
        assert eid in by_id and eid not in recount
        assert all(str(r['episode_id']) == eid and r['step'] == i for i, r in enumerate(records))
        summary_path = path.parent.parent / 'summary.json'
        if summary_path not in summaries:
            summary = read(summary_path)
            assert summary['status'] == 'complete'
            summaries[summary_path] = {str(e['episode_id']): e for e in summary['episodes']}
        summary = summaries[summary_path][eid]
        assert str(summary['trajectory_id']) == expected[eid]
        last = records[-1]
        success = int(last['action'] == 'STOP' and summary['terminal_stop_source'] == 'model'
                      and not last.get('error') and summary['metrics']['distance_to_goal'] < 3.)
        assert success == int(bool(summary['metrics']['success']))
        spl = success * summary['start_distance_m'] / max(summary['start_distance_m'], summary['path_length_m'])
        assert math.isclose(spl, summary['metrics']['spl'], abs_tol=1e-5)
        calls = [float(r['model_request_s']) for r in records if r.get('model_called') is True]
        assert calls and all(math.isfinite(v) and v > 0 for v in calls)
        errors = sum(bool(r.get('error')) for r in records)
        collisions = sum(bool(r['metrics']['collisions']['is_collision']) for r in records)
        stored = by_id[eid]
        assert (stored['success'], stored['steps'], stored['actual_model_calls'], stored['error_count'], stored['collisions']) == (success, len(records), len(calls), errors, collisions)
        assert math.isclose(stored['spl'], summary['metrics']['spl'] if success else 0., abs_tol=1e-12)
        assert math.isclose(stored['request_total_s'], sum(calls), abs_tol=1e-6)
        recount[eid] = (success, stored['spl'], errors, collisions)
        values.extend(calls)
    assert set(recount) == set(expected)
    for name, expected_hash in row['memory_source_files'].items():
        path = ROOT / name
        assert sha(path) == expected_hash, name
        for line in path.read_text(encoding='utf-8').splitlines():
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    memory.append(float(parts[1]))
                except ValueError:
                    pass
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    lower = int(position)
    p95 = ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (position - lower)
    calculated = dict(success_count=sum(v[0] for v in recount.values()),
        sr=sum(v[0] for v in recount.values()) / len(expected),
        spl=sum(v[1] for v in recount.values()) / len(expected), actual_model_calls=len(values),
        request_median_s=statistics.median(values), request_p95_s=p95, request_total_s=sum(values),
        sampled_whole_card_peak_mib=max(memory), error_count=sum(v[2] for v in recount.values()),
        collisions=sum(v[3] for v in recount.values()))
    for key, value in calculated.items():
        assert math.isclose(row[key], value, rel_tol=0, abs_tol=1e-6 if key == 'request_total_s' else 1e-12), key
    trace_count = 0
    for name, expected_hash in row.get('runtime_audit_sha256', {}).items():
        folder = ROOT / name
        audit_path = folder / 'runtime_audit.json'
        assert sha(audit_path) == expected_hash
        audit = read(audit_path)
        assert audit['passed'] and audit['group'] == row['group'] and audit['episodes'] == 50
        assert sha(folder / 'summary.json') == audit['summary_sha256']
        service = folder.parent / 'service'
        assert sha(service / 'launch.json') == audit['launch_sha256']
        launch = read(service / 'launch.json')
        for source, hashed in launch['sources_sha256'].items():
            assert sha(ROOT / 'checks' / source) == hashed, source
        for trace in audit['traces']:
            for kind in ('prefill', 'first_decode'):
                assert sha(service / (kind + '_' + trace['trace_id'] + '.json')) == trace[kind + '_sha256']
            trace_count += 1
        assert len(audit['traces']) == audit['actual_calls'] + audit['warmups']
    if not row['reused']:
        assert len(row['runtime_audit_sha256']) == len(expected) // 50
        assert trace_count == len(values) + 2 * (len(expected) // 50)
        assert sha(ROOT / 'setup/navigation-stage2-execution-20261109.json') == row['execution_contract_sha256']
        assert sha(RUN / 'input_integrity.json') == row['input_integrity_sha256']
        assert sha(ROOT / 'checks/run_navigation_stage2_screen.py') == row['controller_sha256']
    return dict(passed=True, episodes_recounted=len(expected), calls_recounted=len(values),
                trace_pairs_rehashed=trace_count, source_steps_and_memory_hashes_match=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, choices=(50, 100), required=True)
    args = parser.parse_args()
    stem = f'navigation_stage2_same{args.count}_20261109'
    for name in (stem + '.json', stem + '.md', f'navigation_stage2_delivery_audit_{args.count}_20261109.json'):
        assert not (ROOT / 'reports' / name).exists(), 'Preserve existing delivery: ' + name
    state = read(RUN / 'status.json')
    assert state['status'] in ('initial50_complete', 'rescreen100_complete', 'complete'), 'Wait for GPU stage completion'
    expected = {}
    for index in range(1, args.count // 50 + 1):
        roster = read(ROOT / f'data/baseline/navigation_stage2_20261109/source_batch_{index:02d}.json')
        expected.update(zip(map(str, roster['episodes']), map(str, roster['trajectory_ids'])))
    rows, sources, checks = {}, {}, {}
    for group in GROUPS:
        path = RUN / ('reused_dart_v2' if group.startswith('dart') else 'results') / f'{group}_{args.count}.json'
        if path.exists():
            row = read(path)
            assert row['group'] == group
            checks[group] = verify(row, expected)
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
            rows[group] = row
    assert 'fp16' in rows
    if args.count == 50:
        assert set(rows) == set(GROUPS)
    baseline = {e['episode_id']: e['success'] for e in rows['fp16']['episodes']}
    paired = {}
    for group, row in rows.items():
        gained = [e['episode_id'] for e in row['episodes'] if e['success'] and not baseline[e['episode_id']]]
        lost = [e['episode_id'] for e in row['episodes'] if not e['success'] and baseline[e['episode_id']]]
        assert len(gained) - len(lost) == row['success_count'] - rows['fp16']['success_count']
        paired[group] = dict(gained_count=len(gained), lost_count=len(lost), gained_ids=gained, lost_ids=lost)
    quality = {}
    for group, row in rows.items():
        steps = sum(e['steps'] for e in row['episodes'])
        quality[group] = dict(atomic_steps=steps, collision_events=row['collisions'],
            collisions_per_100_atomic_actions=100 * row['collisions'] / steps,
            terminal_source_counts=dict(Counter(e['terminal_stop_source'] for e in row['episodes'])))
    result = dict(status='complete', case_count=args.count, created_utc=datetime.now(timezone.utc).isoformat(),
                  sources_sha256=sources, metrics=rows, paired_against_fp16=paired,
                  quality_details=quality,
                  scope='Fixed development subset, single run; SparseVLM is PriorTR v1.0 single-layer no-merge; DART reuses original measurements.')
    save(ROOT / 'reports' / (stem + '.json'), result)
    save(ROOT / 'reports' / f'navigation_stage2_delivery_audit_{args.count}_20261109.json',
         dict(status='passed', report_sha256=sha(ROOT / 'reports' / (stem + '.json')),
              exporter_sha256=sha(Path(__file__)), groups=checks))
    lines = [f'# 第二阶段：同批{args.count}条开发案例', '',
             '| 配置 | 成功数 | SR | SPL | P50/秒 | P95/秒 | 总请求/小时 | 整卡峰值/MiB | 错误数 | 来源 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for group, row in rows.items():
        lines.append(f"| {NAMES[group]} | {row['success_count']}/{args.count} | {row['sr']:.1%} | {row['spl']:.4f} | {row['request_median_s']:.3f} | {row['request_p95_s']:.3f} | {row['request_total_s']/3600:.3f} | {row['sampled_whole_card_peak_mib']:.0f} | {row['error_count']} | {'复用' if row['reused'] else '新运行'} |")
    lines += ['', '固定原150条开发池中的分层子集，每档运行一次；不是独立最终测试或显著性证明。统一eager、3072完整prefill与原统计方式。',
              '', '延迟只统计实际HTTP请求，排除预热和动作队列零秒；评分、数据搬运与详细审计开销均保留。闭环路径、输入和输出长度会变化，不能将差异解释为固定输入下的纯算法加速。累计请求时间不含仿真、IO和加载，不能代替总墙钟时间。',
              '', '显存为200ms整卡采样，包含桌面和缓存；固定KV池不随裁剪缩小。DART75保留原中断恢复的计时与采样范围限制。SparseVLM明确为PriorTR v1.0单层无合并适配版，不是SparseVLM+或原论文完整多层回收版本。',
              '', f'[逐案例与证据哈希]({stem}.json) · [独立复核](navigation_stage2_delivery_audit_{args.count}_20261109.json) · [适配台账](navigation_algorithm_adaptation_log.md)',
              '', '相对本轮FP16的配对变化：', '', '| 配置 | 新增成功 | 丢失成功 |', '|---|---:|---:|']
    for group, pair in paired.items():
        if group != 'fp16':
            lines.append(f"| {NAMES[group]} | {pair['gained_count']} | {pair['lost_count']} |")
    lines += ['', '碰撞与终止情况（总次数受路径长度影响，附每100个原子动作的碰撞事件数；不是每100次前进的碰撞概率）：',
              '', '| 配置 | 原子动作数 | 碰撞事件 | 每100动作碰撞 | 终止来源及数量 |', '|---|---:|---:|---:|---|']
    for group, q in quality.items():
        terminals = '、'.join(f'{key}: {value}' for key, value in q['terminal_source_counts'].items())
        lines.append(f"| {NAMES[group]} | {q['atomic_steps']} | {q['collision_events']} | {q['collisions_per_100_atomic_actions']:.2f} | {terminals} |")
    with (ROOT / 'reports' / (stem + '.md')).open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='passed', case_count=args.count, groups=list(rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
