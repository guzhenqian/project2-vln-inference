"""Export a completed, audited DART150 run into a concise report and per-task CSV.

Reads saved evidence only. Does not send requests, alter results, or overwrite
an existing export. Timing differences are deliberately not called speedups.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def export_context(run, report, plan):
    """Read optional recovery metadata without changing the frozen metrics.

    Completion is checked before this helper is called. Recovery controllers may
    add metadata while a run is active; missing fields must never imply success.
    The per-episode reuse records remain the authority for which results moved
    across sessions. Warmups are counted separately from those saved requests.
    """
    recovery_path = run / 'recovery_plan.json'
    recovery_plan = read(recovery_path) if recovery_path.exists() else {}
    recovery = report.get('recovery') or {}
    if not isinstance(recovery, dict):
        raise ValueError('Recovery metadata must be an object')
    reused, partial = [], []
    for episode in report['episodes']:
        count = episode.get('prefix_reused_steps', 0)
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= episode['steps']:
            raise ValueError('Invalid reused-prefix step count: ' + str(episode['episode_id']))
        if count:
            target = reused if count == episode['steps'] else partial
            target.append({'episode_id': str(episode['episode_id']), 'batch': episode['batch'],
                           'prefix_reused_steps': count, 'timing_note': episode.get('timing_note', '')})
    recovered = bool(recovery_plan or recovery or reused or partial)
    by_batch = (recovery.get('warmup_requests_by_batch')
                or report.get('warmup_requests_by_batch')
                or recovery_plan.get('warmup_requests_by_batch'))
    if by_batch is None:
        if recovered:
            raise ValueError('Recovered final export requires explicit warmup_requests_by_batch metadata')
        by_batch = {str(item['batch']): 2 for item in plan['batches']}
    if not isinstance(by_batch, dict):
        raise ValueError('warmup_requests_by_batch must be an object')
    by_batch = {str(key): value for key, value in by_batch.items()}
    expected_batches = {str(item['batch']) for item in plan['batches']}
    if (set(by_batch) != expected_batches or
            any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in by_batch.values())):
        raise ValueError('Incomplete or invalid warmup accounting')
    total = sum(by_batch.values())
    declared = report.get('warmup_requests_excluded', recovery.get('warmup_requests_excluded', total))
    if declared != total:
        raise ValueError('Warmup total differs from per-batch accounting')
    if recovered:
        if by_batch != {'1': 4, '2': 2, '3': 2}:
            raise ValueError('This recovery export expects batch1 old2+new2, batch2/3 two warmups each')
        if not reused or not partial or any(item['batch'] != 1 for item in reused + partial):
            raise ValueError('Missing or unexpected first-batch prefix reuse evidence')
        if any(not item['timing_note'] for item in reused + partial):
            raise ValueError('Recovered episodes must retain their timing notes')
    return {'recovered': recovered, 'warmup_requests_excluded': total,
            'warmup_requests_by_batch': by_batch,
            'completed_prefix_episode_count': len(reused), 'completed_prefix_episodes': reused,
            'partial_prefix_episodes': partial,
            'recovery_plan_sha256': hashlib.sha256(recovery_path.read_bytes()).hexdigest() if recovery_path.exists() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = read(run / 'report.json')
    plan = read(run / 'frozen_plan.json')
    if not (report.get('status') == 'complete' and report.get('completed_episode_count') == 150
            and report.get('completed_batches') == [1, 2, 3]
            and report.get('all_runtime_audits_passed') is True and report.get('error_count') == 0):
        raise ValueError('Final export requires all150 tasks and all3 runtime audits complete, with no run errors')
    baseline = read(ROOT / plan['baseline_report'])
    context = export_context(run, report, plan)
    expected = [str(eid) for batch in plan['batches'] for eid in batch['episodes']]
    results = {str(item['episode_id']): item for item in report['episodes']}
    original = {str(item['episode_id']): item for item in baseline['episodes']}
    if len(results) != 150 or set(results) != set(expected) or set(original) != set(expected):
        raise ValueError('DART and FP16 task membership differs')
    latency, old_latency = report['actual_request_latency_s'], baseline['actual_request_latency_s']
    paired = report['paired_comparison']
    stem = ROOT / 'reports' / run.name
    paths = [stem.with_suffix(suffix) for suffix in ('.md', '.csv', '.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('An export already exists; retain it instead of overwriting')
    recovery_notes = []
    if context['recovered']:
        recovery_notes = [
            f"- 批1中断前已完成的{context['completed_prefix_episode_count']}例复用原结果与原请求记录；仅重放保存动作进行像素/位姿校验，没有重新调用模型，也没有把这些案例重复计入150分母。",
            '- 批1预热共4次（中断前2次、恢复后2次），批2和批3各2次，合计8次，全部排除于真实请求延迟与导航轨迹统计。',
        ]
        for item in context['partial_prefix_episodes']:
            recovery_notes.append(
                f"- 案例{item['episode_id']}恢复了前{item['prefix_reused_steps']}步及历史RGB、待执行动作队列；其elapsed_s为原前缀decision+sim时间加续跑时间，缺少原前缀图片/日志IO，因此是循环耗时下界，不能当作不中断的完整墙钟时间。")
        recovery_notes.extend([
            '- 已完成旧案例保留原elapsed_s；重放校验时间不算入导航耗时。旧请求与新请求分别匹配各自服务trace差集，不能要求复用请求再次产生trace。',
            '- 整卡采样峰值合并中断前与恢复后的采样证据；保留各时段原始CSV，不把中断等待与重放时间解释为推理时间。',
        ])
    warmup_note = (
        f"- 延迟只统计model_called=true的真实HTTP请求，排除动作队列的0秒及共{context['warmup_requests_excluded']}次预热；不是纯GPU核时间。")
    lines = [
        '# FP16 + DART keep75：同清单150个导航任务', '',
        '本轮150/150已完成，三批运行期剪枝审计全部通过；无请求或解析错误。', '',
        '|指标|原FP16 150例|本轮FP16+DART keep75 150例|',
        '|---|---:|---:|',
        f"|SR|{paired['fp16_sr']:.2%}|{paired['dart_sr']:.2%}|",
        f"|SPL|{paired['fp16_spl']:.6f}|{paired['dart_spl']:.6f}|",
        f"|真实模型请求数|{old_latency['count']}|{latency['count']}|",
        f"|真实请求中位延迟（秒）|{old_latency['median']:.3f}|{latency['median']:.3f}|",
        f"|真实请求P95（秒）|{old_latency['p95']:.3f}|{latency['p95']:.3f}|",
        f"|整卡采样峰值（MiB）|{baseline['sampled_whole_card_peak_mib']:.0f}|{report['sampled_whole_card_peak_mib']:.0f}|", '',
        f"- 模型主动STOP成功：{report['model_stop_success_count']}/150。",
        f"- 相对原FP16，新增成功{len(paired['gained_success'])}例，丢失成功{len(paired['lost_success'])}例。",
        f"- 共执行{report['completed_episode_steps']}个原子动作，累计碰撞{report['collisions_completed_episodes']}次。", '',
        '## 口径与限制', '',
        '- 使用原冻结150任务、原权重、原作者策略、原动作解析及历史图片策略；每例最多400个原子动作，seed7。',
        '- FP16权重与激活，未使用AWQ或KV量化；每张图片保留75%的视觉token，文字与特殊token全部保留。',
        '- SR成功要求模型主动STOP且距目标严格小于3米；失败任务全部计入150分母，预检单例不额外加入。',
        warmup_note,
        '- 原FP16为CUDA graph/1024分块prefill，本轮为eager/3072完整prefill；闭环轨迹和请求量也会改变，不能把时延差称为纯DART加速倍数。',
        '- 显存为200ms整卡memory.used采样，包含桌面和缓存，可能漏掉瞬时峰值；不是模型独占显存。',
        '- DART减少高层有效token/KV长度，但本轮仍固定分配64个KV blocks，不据此声称分配缓存变小。',
        '- 此150例是固定开发子集，不是官方全量基准；单轮结果未提供统计显著性。',
        *recovery_notes, '',
        '## 原始证据', '',
        f'- 运行目录：{run.relative_to(ROOT).as_posix()}',
        f'- 汇总SHA256：{hashlib.sha256((run / "report.json").read_bytes()).hexdigest()}',
        '- 各批manifest/provenance、逐步JSON和图片、整卡采样CSV、runtime_before及runtime_audit均保留。',
        '- 运行期trace按单请求服务的快照差集与顺序输入签名关联，不声称拥有HTTP request ID绑定。', '',
        '## 逐任务对照', '',
        '|任务ID|FP16成功|DART成功|FP16 SPL|DART SPL|DART步数|',
        '|---|---:|---:|---:|---:|---:|',
    ]
    rows = []
    for eid in expected:
        new, old = results[eid], original[eid]
        old_success = int(old['metrics']['success'] == 1 and old['terminal_stop_source'] == 'model')
        rows.append(dict(episode_id=eid, trajectory_id=new['trajectory_id'], batch=new['batch'],
                         fp16_success=old_success, dart_success=new['protocol_success'],
                         fp16_spl=old['metrics']['spl'] if old_success else 0.0,
                         dart_spl=new['protocol_spl'], dart_steps=new['steps'],
                         dart_model_calls=new['actual_model_calls'],
                         dart_final_distance_m=new['metrics']['distance_to_goal'],
                         dart_collisions=new['metrics']['collisions']['count'],
                         dart_stop_source=new['terminal_stop_source'],
                         dart_elapsed_s=new['elapsed_s'],
                         dart_prefix_reused_steps=new.get('prefix_reused_steps', 0),
                         dart_timing_note=new.get('timing_note', 'Original uninterrupted episode loop time')))
        lines.append(f"|{eid}|{old_success}|{new['protocol_success']}|{rows[-1]['fp16_spl']:.6f}|{new['protocol_spl']:.6f}|{new['steps']}|")
    with paths[0].open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    with paths[1].open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[2].open('x', encoding='utf-8') as stream:
        json.dump(dict(report, export_context=context), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print('DART150_FINAL_EXPORT_OK')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
