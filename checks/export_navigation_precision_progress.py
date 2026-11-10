"""Publish audited complete rows and completion counts; never extrapolate partial SR."""
import json
from datetime import datetime, timezone, timedelta
from navigation_precision_common import ROOT, RUN, GROUPS, read, digest
from export_navigation_precision_results import verify_result


def main():
    selection = read(ROOT/'datasets/navigation_20261110/selection.json')
    annotations = read(ROOT/'datasets/navigation_20261110/val_unseen150.json')
    trajectories = {str(e['episode_id']): str(e['trajectory_id']) for e in annotations['episodes']}
    state = read(RUN/'status.json')
    rows = []
    keys = ('case_count','success_count','sr','spl','actual_model_calls','request_median_s',
            'request_p95_s','sampled_whole_card_peak_mib','error_count','collisions','request_total_s',
            'quality_error_counts')
    for algorithm in ('awq','gptq'):
        for precision in ('w4a16','w4a4','w4a8','w8a16'):
            group = algorithm+'_'+precision
            old = precision == 'w4a16'
            result = (ROOT/'logs/navigation_stage2_sequence_20261109/results'/(algorithm+'_100.json')
                      if old else RUN/'results'/(group+'_100.json'))
            row = dict(group=group, reused=old, completed_cases=0, status='pending', metrics=None)
            if result.exists():
                metrics = verify_result(result, selection['same100'], trajectories)
                if not old:
                    for name, sha in metrics['runtime_audit_sha256'].items():
                        audit_path = ROOT/name/'runtime_audit.json'
                        assert digest(audit_path) == sha
                        audit = read(audit_path)
                        assert audit['passed'] and audit['infrastructure_errors'] == 0
                row.update(status='complete_reused' if old else 'complete', completed_cases=100,
                           metrics={key: metrics[key] for key in keys},
                           result_source=result.relative_to(ROOT).as_posix(), result_sha256=digest(result))
            elif not old:
                folders = [p for p in (RUN/'formal').glob(group+'_*') if p.is_dir()]
                assert len(folders) <= 1, 'Explicit recovery accounting required'
                if folders:
                    count = 0
                    for batch in (1,2):
                        folder = folders[0]/f'batch_{batch:02d}'
                        if (folder/'summary.json').exists():
                            count += read(folder/'summary.json')['episode_count']
                        elif (folder/'progress.json').exists():
                            count += read(folder/'progress.json')['completed_episodes']
                    row.update(completed_cases=count, status='running' if state['status']!='failed' else 'interrupted')
                row['technical_execution_validated'] = (RUN/'validation'/(group+'_passed.json')).exists()
            rows.append(row)
    data = dict(updated_utc=datetime.now(timezone.utc).isoformat(), status=state['status'],
        active_group=state.get('active_group'), rows=rows,
        completed_new_cases=sum(r['completed_cases'] for r in rows if not r['reused']), new_cases_planned=600,
        all_completed_metrics_independently_recomputed=True,
        exporter_sources_sha256={name:digest(ROOT/'checks'/name) for name in (
            'export_navigation_precision_progress.py','export_navigation_precision_results.py')},
        precision_config_sha256=digest(ROOT/'setup/navigation-precision-20261110.json'),
        scope='Same100 per group; experimental kernels; only completed audited groups receive performance numbers')
    data['updated_shanghai'] = datetime.fromisoformat(data['updated_utc']).astimezone(timezone(timedelta(hours=8))).isoformat()
    output = ROOT/'reports/navigation_precision_progress_20261110'
    output.with_suffix('.json').write_bytes((json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    lines = ['# 八组精度比较进度', '',
        f"快照时间（北京时间）：{data['updated_shanghai']}；新运行已完成{data['completed_new_cases']}/600条逐配置案例。此页是人工导出的进度快照，不是实时监控。", '',
        '两种W4A16沿用原同100条；六种新配置通过技术执行验收后逐组运行。只为完成100条并通过审计的组填写统计，未完成组不按部分案例推算成功率。', '',
        '| 配置 | 完成案例 | 状态 | 成功/100 | SPL | P95/s | 整卡峰值/MiB | 格式/截断错误 |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    labels = dict(pending='待运行',running='运行中',interrupted='中断待恢复',complete='完成',complete_reused='完成（复用）')
    for row in rows:
        prefix = f"| {row['group'].upper()} | {row['completed_cases']}/100 | {labels[row['status']]} |"
        m = row['metrics']
        lines.append(prefix+(f" {m['success_count']} | {m['spl']:.4f} | {m['request_p95_s']:.3f} | {m['sampled_whole_card_peak_mib']:.0f} | {m['error_count']} |" if m else ' — | — | — | — | — |'))
    for row in rows:
        m = row['metrics']
        if m and m['error_count']:
            labels = {'truncated_model_output':'输出截断', 'invalid_action_random_fallback_rejected':'动作格式错误'}
            breakdown = '、'.join(f"{labels[key]}{value}条" for key,value in m['quality_error_counts'].items())
            lines += ['', f"{row['group'].upper()}：成功{m['success_count']}/100，{breakdown}。这是本次激活方案和实现的结果，不能推广为所有同位宽方法的结论。"]
    lines += ['',
        '六组新配置采用项目实验算子，旧W4A16采用原生算子；这里比较实际配置，不能归因为纯位宽效应。W4A4激活打包四位后扩展到INT8 MMA，不代表原生INT4算力。A4/A8仅作用于语言投影输入，其余部分保留FP16。', '',
        '格式错误与输出截断保留在100条分母中，错误触发的STOP不计成功；两次预热与动作队列0秒不计请求延迟。基础设施错误不能通过审计。开发子集单次运行不能证明统计优越性。', '',
        '[接入验证](navigation_precision_integration_20261110.md) · [配置边界](navigation_precision_plan_20261110.md) · [速度排查](navigation_quant_timing_20261110.md) · [数据清单](../datasets/navigation_20261110/README.md)', '']
    output.with_suffix('.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps(dict(status='exported',completed_new_cases=data['completed_new_cases'],new_cases_planned=600)))


if __name__ == '__main__':
    main()
