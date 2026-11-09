"""Summarize recorded token budgets and selector intervals; no inference."""
import argparse
from collections import Counter
import json
import statistics
from export_navigation_stage2_delivery import ROOT, RUN, NAMES, read, save, sha


def p95(values):
    ordered = sorted(values)
    position = (len(ordered) - 1) * .95
    index = int(position)
    return ordered[index] + (ordered[min(index + 1, len(ordered) - 1)] - ordered[index]) * (position - index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, choices=(50, 100), required=True)
    args = parser.parse_args()
    source = ROOT / f'reports/navigation_stage2_same{args.count}_20261109.json'
    audit = read(ROOT / f'reports/navigation_stage2_delivery_audit_{args.count}_20261109.json')
    assert audit['status'] == 'passed' and audit['report_sha256'] == sha(source)
    report = read(source)
    result = {}
    for group, row in report['metrics'].items():
        if row['reused']:
            result[group] = dict(source='retained historical runtime audit',
                visual_tokens_per_image_before=300,
                visual_tokens_per_image_after=225 if group == 'dart75' else 150,
                selector_interval_status='not_recorded_separately_in_historical_run',
                note='DART standalone selector time cannot be reconstructed. Its overhead remains in the original HTTP timings.')
            continue
        hist, prepare, select, fallback = Counter(), [], [], 0
        for folder in row['batch_directories']:
            batch = ROOT / folder
            runtime = read(batch / 'runtime_audit.json')
            assert runtime['passed'] and runtime['warmups'] == 2
            for proof in runtime['traces'][2:]:
                path = batch.parent / 'service' / ('prefill_' + proof['trace_id'] + '.json')
                assert sha(path) == proof['prefill_sha256']
                trace = read(path)
                before, after = trace['original_image_tokens'], trace['retained_image_tokens']
                assert before == 300 * trace['image_count']
                keep = .75 if group.endswith('75') else .5 if group.endswith('50') else 1.
                assert after == int(before * keep)
                hist[f'{before}->{after}'] += 1
                if group.startswith(('fastv', 'sparsevlm')):
                    timing = trace['selection']
                    prepare.append(timing['prepare_s'])
                    select.append(timing['selection_including_transfer_s'])
                    fallback += bool((timing.get('prepare') or {}).get('uniform_score_fallback'))
        assert sum(hist.values()) == row['actual_model_calls']
        entry = dict(actual_calls=row['actual_model_calls'], token_budget_histogram=dict(sorted(hist.items())),
            selector_interval_status='recorded_partial_wall_clock' if select else 'not_applicable')
        if select:
            assert len(select) == len(prepare) == row['actual_model_calls']
            combined = [p + s for p, s in zip(prepare, select)]
            entry.update(prepare_median_s=statistics.median(prepare), selection_median_s=statistics.median(select),
                combined_median_s=statistics.median(combined), combined_p95_s=p95(combined),
                combined_total_s=sum(combined), uniform_score_fallback_requests=fallback,
                scope='prepare_s + selection_including_transfer_s: includes CPU scoring, copies and synchronization waits; excludes other gather/scatter/KV/DeepStack bookkeeping and is not pure GPU or complete compression time.')
        result[group] = entry
    stem = f'navigation_stage2_diagnostics_{args.count}_20261109'
    save(ROOT / 'reports' / (stem + '.json'), dict(status='complete', case_count=args.count,
        source_report_sha256=sha(source), exporter_sha256=sha(ROOT / 'checks/export_navigation_stage2_diagnostics.py'), groups=result))
    lines = [f'# 第二阶段：{args.count}条的Token与选择器记录', '',
        '全部新运行按真实请求排除每批2次预热。每图原始300个视觉token；75%/50%的多图总预算为225N/150N。FastV和SparseVLM全局分配，不能理解为每图都恰好保留225/150个；DART为逐图固定配额。全部文字与特殊token保留。', '',
        '| 配置 | 真实请求 | 准备+选择P50/秒 | 准备+选择P95/秒 | 累计记录区间/秒 | 空重要文本分支 |',
        '|---|---:|---:|---:|---:|---:|']
    for group, row in result.items():
        if 'combined_median_s' in row:
            lines.append(f"| {NAMES[group]} | {row['actual_calls']} | {row['combined_median_s']:.4f} | {row['combined_p95_s']:.4f} | {row['combined_total_s']:.3f} | {row['uniform_score_fallback_requests']} |")
    lines += ['', '上表仅是已记录的prepare和selection区间墙钟：包含CPU计算、搬运及隐式同步等待，未覆盖全部特征整理、散回和KV/DeepStack维护，不能称为纯算法内核耗时或完整压缩开销。整体收益仍以HTTP延迟判断，这些区间已包含在HTTP计时内。',
        '', '历史DART没有单独记录同口径选择器时间，不能事后从端到端时间中拆出；继续保留原HTTP性能，不补造分项。FP16/AWQ/GPTQ没有Token裁剪，选择器分项不适用。',
        '', f'[每档实际Token数量与请求频数]({stem}.json)']
    with (ROOT / 'reports' / (stem + '.md')).open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='complete', case_count=args.count, groups=list(result)), ensure_ascii=False))


if __name__ == '__main__':
    main()
