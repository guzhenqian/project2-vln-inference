"""Publish compact local acceptance evidence only after all seven groups pass."""
from datetime import datetime, timezone
from navigation_stage2_common import ROOT, RUN, GROUPS, read, write, digest


def export():
    target=ROOT/'reports/navigation_stage2_integration_20261109.json'
    if target.exists():
        previous=read(target)
        assert previous['status']=='all_seven_passed'
        return previous
    groups={}
    for group,recipe in GROUPS.items():
        passed=read(RUN/'validation'/(group+'_passed.json'))
        assert passed['passed']
        service=ROOT/passed['service']
        directory=service.parent
        single=ROOT/passed['single']
        summary=read(single/'summary.json')
        audit=read(single/'runtime_audit.json')
        assert audit['passed'] and digest(single/'runtime_audit.json')==passed['single_audit_sha256']
        episode=summary['episodes'][0]
        groups[group]=dict(passed=True,recipe=recipe,
            smoke_traces=[read(directory/f'smoke{i}_audit.json') for i in (0,1)],
            single_episode=dict(episode_id=episode['episode_id'],steps=episode['steps'],
                model_calls=audit['actual_calls'],errors=episode['error_count'],
                stop_source=episode['terminal_stop_source'],success=episode['metrics']['success'],
                distance_to_goal=episode['metrics']['distance_to_goal'],spl=episode['metrics']['spl']),
            trace_pairs=len(audit['traces']),warmups_excluded=audit['warmups'],
            launch_sha256=digest(service/'launch.json'),single_audit_sha256=passed['single_audit_sha256'],
            runtime_sources_sha256=passed['source_hashes'])
    kernel=read(ROOT/'reports/navigation_gptq_kernel_stage2_20261109.json')
    quant=read(ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_GPTQ_CT_stage2_20261109/project2_quantization.json')
    result=dict(status='all_seven_passed',created_utc=datetime.now(timezone.utc).isoformat(),groups=groups,
        gptq=dict(calibration_inputs=quant['samples'],quantized_layers=quant['quantized_layer_count'],
            calibration_seconds=quant['elapsed_s'],fallback_events=quant['fallback_events'],
            native_cuda_checks=len(kernel['checks']),max_relative_l2=kernel['max_relative_l2'],model_sha256=kernel['model_sha256']),
        input_integrity_sha256=digest(RUN/'input_integrity.json'),
        reused_dart_verification_sha256=digest(RUN/'reused_dart_v2/verification.json'),
        limitations=['Acceptance means correct runtime/format/cache handling, not navigation-quality superiority.',
            'SparseVLM is pinned PriorTR v1.0 single-layer no-merge adaptation, not SparseVLM+ or full multistage paper reproduction.',
            'DART75/50 are reused with original statistics and recovery limitations; no new DART inference.',
            'Fifty/one-hundred development-case experiments are separate from these single-case acceptance results.'])
    write(target,result)
    lines=['# 第二阶段：七组接入验收', '', '全部七组通过真实两图/九图请求及完整单案例验收；下表单例结果不是算法性能排名。', '',
        '| 组 | 接入验收 | 单例步数 | 真实调用 | 运行/解析错误 | 单例导航成功 |', '|---|---|---:|---:|---:|---|']
    for group,entry in groups.items():
        row=entry['single_episode']
        lines.append(f"| {group} | 通过 | {row['steps']} | {row['model_calls']} | {row['errors']} | {'是' if row['success'] else '否'} |")
    lines += ['', '统一使用eager、3072完整prefill、固定64×64 KV池和原导航策略。GPTQ经过真实32输入/252层量化、保值格式转换及18项CUDA核验；部署格式为AWQ GEMM，量化算法仍为GPTQ。',
        '', 'FastV在零基第2层前裁剪；SparseVLM采用PriorTR v1.0单层无合并配置，在第3层前裁剪。均使用多图真实mask和全局视觉预算，记录实际逐图分配。',
        '', 'DART75/50复用原结果的相同50/100子集；原恢复计时与整卡显存采样限制保留。原FP16/AWQ旧graph/1024成绩不作为本轮统一配置结果。',
        '', '完整问题与解决记录：[算法适配台账](navigation_algorithm_adaptation_log.md)。']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('STAGE2_ALL_SEVEN_ACCEPTANCE_EXPORTED',flush=True)
    return result


if __name__=='__main__':
    export()
