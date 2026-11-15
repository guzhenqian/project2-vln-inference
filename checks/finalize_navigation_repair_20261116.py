"""Publish a compact verified conclusion from the completed nine-group audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports'
SOURCE = REPORTS/'navigation_repair_20261114_continuation_inventory_after_gptq_w8a16.json'
ACCEPTANCE = REPORTS/'navigation_repair_20261114_continuation_20.json'
OUTPUT = REPORTS/'navigation_repair_final_20261116.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert not OUTPUT.exists() and not OUTPUT.with_suffix('.md').exists()
    inventory, acceptance = read(SOURCE), read(ACCEPTANCE)
    assert inventory['all_nine_100_complete'] and inventory['all_eight_quantized_100_comparisons_available']
    assert inventory['latest20_source']['sha256'] == sha(ACCEPTANCE)
    assert acceptance['audit_source_sha256'] == sha(ROOT/'checks/audit_navigation_repair.py')
    original = read(ROOT/'data/baseline/navigation_repair_20261112/original100.json')
    identities = dict(zip(map(str,original['episodes']),map(str,original['trajectory_ids'])))
    groups = ['fp16']+[a+'_'+p for a in ('awq','gptq') for p in ('w4a16','w4a4','w4a8','w8a16')]
    assert [r['group'] for r in inventory['table']] == groups and len(identities)==100
    rows, references = [], {}
    for row in inventory['table']:
        group, reference = row['group'], row['report100']
        path = (ROOT/reference['path']).resolve()
        assert path.is_relative_to(REPORTS) and sha(path)==reference['sha256']
        report = read(path)
        assert report['status']=='complete' and report['case_count']==100 and report['passed_stability']
        assert report['error_count']==0 and report['infrastructure_errors']==0
        assert dict((str(e['episode_id']),str(e['trajectory_id'])) for e in report['episodes'])==identities
        assert len(report['episodes'])==100 and [b['metrics']['case_count'] for b in report['batches']]==[20,80]
        assert row['comparison']['initial_input_identity_cases']==100 and not row['comparison']['initial_input_mismatches']
        gate = next(r for r in acceptance['table'] if r['group']==group)
        assert gate['passed'] and gate['status']=='passed'
        assert all(value['passed'] for value in gate['fixed_speed_cells'].values())
        max_ratio = max(max(value['http_p50_ratio_to_fp16'],value['http_p95_ratio_to_fp16'])
                        for value in gate['fixed_speed_cells'].values())
        assert max_ratio<=1.10
        metrics = row['metrics100']
        assert all(report[k]==v for k,v in metrics.items())
        references[group] = reference
        rows.append(dict(group=group,status='complete_100',metrics=metrics,
            comparison=row['comparison'],same20_max_fixed_p50_p95_ratio=max_ratio,
            quality_interpretation='Observed success/SPL loss versus FP16; output stability restored in these100 cases.'
                if group.endswith('w4a4') else 'Descriptive fixed100 result; no statistical equivalence or superiority claim.'))
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),status='repair_evaluation_complete',
        all_nine_100_complete=True,case_evaluations=900,
        actual_model_calls=sum(row['metrics']['actual_model_calls'] for row in rows),
        all_requested_precision_execution_verified=True,all_groups_zero_output_or_infrastructure_errors=True,
        all_quantized_quality_fully_restored=False,remaining_quality_limitations=['awq_w4a4','gptq_w4a4'],
        table=rows,report_references=references,
        evidence=dict(final_raw_audit=dict(path=SOURCE.relative_to(ROOT).as_posix(),sha256=sha(SOURCE)),
            accepted20=dict(path=ACCEPTANCE.relative_to(ROOT).as_posix(),sha256=sha(ACCEPTANCE)),
            original100_sha256=sha(ROOT/'data/baseline/navigation_repair_20261112/original100.json'),
            finalizer_sha256=sha(Path(__file__))),
        limitations=['New accepted20 + unchanged remaining80; no historical cancelled cases reused.',
            'Same100 development cases;20 engineering gates do not establish statistical equivalence.',
            'HTTP quantiles pool real requests; subsequent trajectories/output lengths differ across models.',
            'Memory is whole-card200ms sampled peak including desktop and prewarm, not model allocator memory.',
            'W4A4 uses packedINT4 widened toINT8 MMA; W4A8 usesINT8 MMA; W8A16 unpacks byte weights intoFP16 MMA.',
            'Artifact arithmetic/packing/execution correctness and output stability do not prove full navigation quality recovery.'])
    OUTPUT.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    names = dict(fp16='FP16', **{g:g.split('_')[0].upper()+' '+g.split('_')[1].upper() for g in groups[1:]})
    lines = ['# 项目2.8量化修复：最终同100条结果','',
        'FP16与AWQ/GPTQ各四种精度全部完成本轮同100条复测，共900次案例评估，且全部通过原始轨迹、真实请求计时、执行配置及逐例首次输入核对。九组均无请求、格式或截断错误。**实验已完成，但两组W4A4仍有导航质量损失，不能宣称八组质量全部恢复。**','',
        '| 配置 | 成功/100 | SPL | 相对FP16成功数 | HTTP P50/s | HTTP P95/s | 整卡峰值/MiB | 错误数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m,c=row['metrics'],row['comparison']
        lines.append(f"| {names[row['group']]} | {m['success_count']} | {m['spl']:.4f} | {c['success_count_delta']:+d} | {m['request_median_s']:.3f} | {m['request_p95_s']:.3f} | {m['sampled_whole_card_peak_mib']:.0f} | {m['error_count']} |")
    lines += ['',
        'W4A16两组在固定相同输入的速度门槛和同100条质量观察上表现较好；AWQ/GPTQ分别66/68次成功，不能据两次成功之差推断算法优劣。W4A8分别65/63次成功。W8A16分别65/66次成功，但本轮闭环HTTP中位数高于FP16。', '',
        'W4A4分别53/58次成功，比FP16少13/8次，SPL分别低约0.0964/0.0501。新配方在本轮消除了旧配置观察到的截断问题，但完整100条仍显示质量损失；数值内核正确与输出稳定不等于导航质量无损。', '',
        '闭环HTTP分位直接汇总全部真实请求，排除排队动作的零耗时、预热和尾请求，不平均两批分位。不同模型后续轨迹及输出长度不同，因此该表不能单独证明量化算法的纯速度收益。整卡峰值为200ms采样，包含桌面与预热，不是模型独占显存。','',
        '## 固定输入与真实执行验证','',
        '同20条验收使用两图/九图×18/64输出，四格各2次预热、10次正式计时；下表为四格P50/P95相对同轮FP16的最大比值，预注册上限1.10。两次FP16固定输入复测均未触发超过10%的漂移复核。','',
        '| 配置 | 四格最大P50/P95比值 | 技术与20条验收 |', '|---|---:|---|']
    lines += [f"| {names[r['group']]} | {r['same20_max_fixed_p50_p95_ratio']:.3f} | 通过 |" for r in rows]
    lines += ['',
        'W4A16主数值104项及独立进程固定表回放52项通过；其余六种精度共288项真实投影数值检查通过。验证包含实际打包矩阵SHA、独立算术参考、实际MMA指令、动态长度和冻结缓存证据。A4为每token每K128组动态scale，打包INT4扩展为INT8 MMA；A8为整token动态scale和INT8 MMA；W8为字节权重解包后FP16 MMA。均不冒充原生INT4 Tensor Core实现。','',
        '## 修复内容与统计边界','',
        '- 仅在明确的固定预热输入采集像素SHA，避免闭环每个新输入长度触发额外CPU搬运。',
        '- W4A16启动时准备代表长度并冻结launch表，保持真实M、完整3072 prefill和原始权重；追加80条复用对应20条实际选择的表。',
        '- 六种精度使用独立真实执行内核、启动准备与冻结，保留原生decode CUDA Graph；A4改变激活量化粒度，未在正式100条轨迹上选参。',
        '- 评测保持原导航策略、temperature0.3、top_p0.95、max512、seed7、400步上限，batch/TP/DP均为1，FP16 KV64×64；无DART/Token组合。',
        '- 每组先通过独立选择的同20条，再追加原100清单中不重叠的80条。权重、实现、评测及launch条件一致；历史失败和取消结果均保留且不混入。',
        '- 成功只计模型主动STOP且无错误、距离目标小于3m。SPL按保存轨迹重算；初始输入逐例与FP16一致，后续轨迹允许不同。',
        '- 九组使用同一批开发案例，20条工程验收与100条描述性结果均不是统计质量等价或泛化证明。','',
        '## 可追溯证据','',
        '- [最终九组原始证据汇总](navigation_repair_20261114_continuation_inventory_after_gptq_w8a16.json)',
        '- [最终结果与来源SHA](navigation_repair_final_20261116.json)',
        '- [九组同20条独立验收](navigation_repair_20261114_continuation_20.json)',
        '- [六种精度独立内核核验](navigation_repair_six_kernel_audit_20261114.json)',
        '- [预注册协议](navigation_repair_protocol_20261112.md)', '',
        '本轮修复、完整复测与交付已完成；A4质量损失及W8闭环延迟限制明确保留，不以成功完成测试替代质量无损结论。','']
    OUTPUT.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(output=OUTPUT.relative_to(ROOT).as_posix(),case_evaluations=900,actual_model_calls=record['actual_model_calls']),ensure_ascii=False))


if __name__=='__main__':
    main()
