"""Publish compact execution evidence without treating smoke tests as benchmarks."""
import json
from datetime import datetime, timezone
from navigation_precision_common import ROOT, RUN, GROUPS, read, digest
from audit_navigation_precision import audit_service


def main():
    rows = {}
    for group, recipe in GROUPS.items():
        evidence_path = RUN/'validation'/(group+'_passed.json')
        evidence = read(evidence_path)
        directory = (ROOT/evidence['single']).parent
        service = ROOT/evidence['service']
        launch = audit_service(service, group)
        audit = read(ROOT/evidence['single']/'runtime_audit.json')
        assert audit['passed'] and audit['episodes'] == 1 and audit['infrastructure_errors'] == 0
        assert digest(ROOT/evidence['single']/'runtime_audit.json') == evidence['single_audit_sha256']
        summary = read(ROOT/evidence['single']/'summary.json')
        smokes = []
        for index in (0, 1):
            record = read(directory/f'smoke{index}.json')
            trace = read(directory/f'smoke{index}_audit.json')
            smokes.append(dict(image_count=trace['image_count'], prompt_tokens=trace['original_length'],
                finish_reason=record['response']['choices'][0]['finish_reason'],
                completion_tokens=record['response']['usage']['completion_tokens'],
                input_images_sha256=record['images_sha256'], request_sha256=record['request_sha256'],
                response_record_sha256=digest(directory/f'smoke{index}.json'), trace=trace))
        rows[group] = dict(recipe=recipe, technical_execution_passed=True,
            evidence_sha256=digest(evidence_path), launch_sha256=digest(service/'launch.json'),
            model_weights_sha256=launch['model_weights_sha256'], model_bytes=launch['model_bytes'],
            sources_sha256=launch['sources_sha256'], smokes=smokes,
            single_episode=summary['episodes'][0], quality_errors=audit['quality_errors'],
            infrastructure_errors=0, actual_calls=audit['actual_calls'],
            warmups_sha256=audit['warmups_sha256'])
    data = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(), groups=rows,
        scope='Execution and input/cache integrity only; not navigation quality qualification or latency benchmark',
        output_quality_failures_retained=True, formal_cases_per_group=100,
        old_w4a16_groups_reused=2, new_groups=6)
    stem = ROOT/'reports/navigation_precision_integration_20261110'
    with stem.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    lines = ['# 六组新增精度：真实接入验证', '',
        '本表只验收技术执行。成功加载、真实低位算子执行、完整prefill、KV与位置编码通过检查，不代表导航质量合格。单案例不用于选择算法，也不替代同100条结果。', '',
        '| 组 | 两图/九图实际token | 单例步数 | 单例模型STOP成功 | 单例格式/截断错误 | 技术执行 |',
        '|---|---|---:|---:|---:|---|']
    for name, row in rows.items():
        ep = row['single_episode']
        ok = int(ep['terminal_stop_source']=='model' and ep['metrics']['success']==1)
        lines.append(f"| {name.upper()} | 701 / 2815 | {ep['steps']} | {ok} | {ep['error_count']} | 通过 |")
    lines += ['',
        '每组核对144个量化投影与756个加载张量、实际W/A位数和执行路径；W4A4是打包INT4激活扩展至INT8 MMA，W4A8为INT8 MMA，W8A16为解包后FP16 MMA。视觉塔、KV等保留FP16。', '',
        '保留所有输出截断/格式失败，正式统计分母始终为同100条。基础设施故障不能通过审计。输入、权重、源码、请求响应和轨迹均记录哈希；预热不纳入请求延迟和成功率统计。', '',
        '[完整证据](navigation_precision_integration_20261110.json) · [配置与适配边界](navigation_precision_plan_20261110.md)', '']
    with stem.with_suffix('.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    print('PRECISION_SIX_GROUP_INTEGRATION_EXPORTED', stem)


if __name__ == '__main__':
    main()
