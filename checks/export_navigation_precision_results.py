"""Independently recompute the eight same100 rows before publishing a final report."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from navigation_precision_common import ROOT, RUN, GROUPS, read, digest
from navigation_precision_quality import verify_step


def percentile(values, p):
    values=sorted(values)
    pos=(len(values)-1)*p
    lo=int(pos)
    return values[lo]+(pos-lo)*(values[min(lo+1,len(values)-1)]-values[lo])


def verify_result(path, episode_ids, expected_trajectories):
    result=read(path)
    assert result['status']=='complete' and result['case_count']==100
    assert [r['episode_id'] for r in result['episodes']]==episode_ids
    latencies=[]
    success=collisions=errors=0
    spls=[]
    for row in result['episodes']:
        eid=row['episode_id']
        assert str(row['trajectory_id'])==expected_trajectories[eid]
        matches=[(name,sha) for name,sha in result['step_files_sha256'].items()
                 if name.endswith('/episode_'+eid+'/steps.jsonl')]
        assert len(matches)==1
        name,sha=matches[0]
        steps_path=ROOT/name
        assert digest(steps_path)==sha
        steps=[json.loads(line) for line in steps_path.read_text().splitlines()]
        assert len(steps)==row['steps'] and all(s['step']==i for i,s in enumerate(steps))
        for step in steps:
            verify_step(step)
        summary=read(steps_path.parent.parent/'summary.json')
        ep=next(e for e in summary['episodes'] if str(e['episode_id'])==eid)
        end=steps[-1]
        ok=int(end['action']=='STOP' and ep['terminal_stop_source']=='model'
               and not end.get('error') and ep['metrics']['distance_to_goal']<3.)
        spl=ok*ep['start_distance_m']/max(ep['start_distance_m'],ep['path_length_m'])
        assert ok==row['success'] and math.isclose(spl,row['spl'],abs_tol=1.e-5)
        real=[s['model_request_s'] for s in steps if s['model_called']]
        assert len(real)==row['actual_model_calls'] and all(math.isfinite(v) and v>0 for v in real)
        latencies.extend(real)
        success+=ok
        spls.append(spl)
        collisions+=sum(bool(s['metrics']['collisions']['is_collision']) for s in steps)
        errors+=sum(bool(s.get('error')) for s in steps)
    memory=[]
    for name,sha in result['memory_source_files'].items():
        assert digest(ROOT/name)==sha
        for line in (ROOT/name).read_text().splitlines():
            parts=line.split(',')
            try:
                memory.append(float(parts[1]))
            except (ValueError,IndexError):
                pass
    for key,value in dict(success_count=success,sr=success/100,spl=statistics.mean(spls),
        actual_model_calls=len(latencies),request_median_s=statistics.median(latencies),
        request_p95_s=percentile(latencies,.95),request_total_s=sum(latencies),
        collisions=collisions,error_count=errors,sampled_whole_card_peak_mib=max(memory)).items():
        assert math.isclose(result[key],value,rel_tol=1.e-9,abs_tol=1.e-5),(path,key,result[key],value)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stem',default='navigation_precision_same100_20261111')
    args=parser.parse_args()
    assert read(RUN/'status.json')['status']=='complete', 'All runtime gates and six same100 groups must finish first'
    selection=read(ROOT/'datasets/navigation_20261110/selection.json')
    ids=selection['same100']
    annotation=read(ROOT/'datasets/navigation_20261110/val_unseen150.json')
    trajectories={str(ep['episode_id']):str(ep['trajectory_id']) for ep in annotation['episodes']}
    rows={}
    sources={}
    for algorithm in ('awq','gptq'):
        old=ROOT/'logs/navigation_stage2_sequence_20261109/results'/(algorithm+'_100.json')
        rows[algorithm+'_w4a16']=verify_result(old,ids,trajectories)
        sources[algorithm+'_w4a16']=dict(result_path=old.relative_to(ROOT).as_posix(),sha256=digest(old),reused=True)
        for precision in ('w4a4','w4a8','w8a16'):
            group=algorithm+'_'+precision
            path=RUN/'results'/(group+'_100.json')
            rows[group]=verify_result(path,ids,trajectories)
            sources[group]=dict(result_path=path.relative_to(ROOT).as_posix(),sha256=digest(path),reused=False)
            for name,sha in rows[group]['runtime_audit_sha256'].items():
                assert digest(ROOT/name/'runtime_audit.json')==sha
                assert read(ROOT/name/'runtime_audit.json')['passed']
            for name,sha in rows[group]['source_hashes'].items():
                assert digest(ROOT/'checks'/name)==sha
    report=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        validation=dict(passed=True,groups=8,cases_per_group=100,new_case_records=600,reused_case_records=200,
                        independently_recomputed_from_steps=True,source_files_and_memory_hashes_verified=True),
        selection_sha256=digest(ROOT/'datasets/navigation_20261110/selection.json'),
        configuration=read(ROOT/'setup/navigation-precision-20261110.json'),sources=sources,groups=rows,
        runtime_validation={group:read(RUN/'validation'/(group+'_passed.json')) for group in GROUPS},
        checkpoint_conversion_audits={group:read(ROOT/recipe['model']/'conversion_audit.json') for group,recipe in GROUPS.items()},
        numerical_kernel_gate=read(ROOT/'reports/navigation_precision_kernel_gate_linked_20261110.json'))
    output=ROOT/'reports'/args.stem
    with output.with_suffix('.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    lines=['# AWQ/GPTQ八组精度：同100条导航结果','',
        '固定原150条开发池中的同100条，六组新运行全部通过真实两图/九图与完整单案例接入验收；两组W4A16复用原第二阶段相同100条。逐案例成功、SPL、请求计时、碰撞与显存已从原始记录独立重算。','',
        '| 配置 | 成功/100 | SPL | 请求P50/s | 请求P95/s | 整卡峰值/MiB | 碰撞次数 | 格式/截断错误 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f"| {name.upper()} | {r['success_count']} | {r['spl']:.4f} | {r['request_median_s']:.3f} | {r['request_p95_s']:.3f} | {r['sampled_whole_card_peak_mib']:.0f} | {r['collisions']} | {r['error_count']} |")
    lines += ['',
        '保持RTX4070Ti12GB、LMDeploy0.11.1/PyTorch2.5.1+cu121、eager、3072完整prefill、固定64×64 KV池、同提示词与历史、随机种子7和400原子动作上限。只有真实HTTP模型请求参与延迟统计，排除预热和动作队列0秒；显存为200ms整卡采样，包括桌面等公共占用。', '',
        '**实现边界：** W4A4和W4A8采用同一动态逐token对称maxabs整数激活方案，保留各自原W4权重；W8用同32个训练输入重新做AWQ/GPTQ、非对称group128。语言投影以外和投影之间的张量仍为FP16。W4A4的激活实际打包四位，进入寄存器后扩展为INT8整数MMA，不是原生INT4 Tensor Core路线。结果只评价本次激活方案与执行实现，不能推广为所有W4A4/W4A8方法的效果。', '',
        '六组新配置使用项目内独立实验算子，原W4A16使用原生AWQ算子。速度差异同时受到位宽与算子实现影响，不能解释为纯位宽因果效应，也不代表官方库原生支持这些配置。每个输入对应不同导航轨迹和输出长度，闭环P95不能代替固定输入的prefill/decode微测。', '',
        '所有组保留完整100条分母。输出截断或已有动作解析器无法解析时，保留错误、终止该任务并记失败；错误触发的STOP即使落在目标附近也不计成功。新入口仅放行这两种已知质量失败，HTTP/超时/CUDA等基础设施错误不能通过审计。两次预热的同类错误另存且不计入任务统计；未根据开发集结果调节激活参数。', '',
        '这是开发子集上的单次结果；100条中一条成败改变SR一个百分点。小幅成功数差异不构成算法稳定优越的证据。先看导航能力、格式及稳定性，再评估速度和显存；未运行Token组合。', '',
        '[精度定义与适配边界](navigation_precision_plan_20261110.md) · [原W4A16速度排查](navigation_quant_timing_20261110.md) · [数据与名单](../datasets/navigation_20261110/README.md) · [完整结果与复核]('+args.stem+'.json)', '']
    with output.with_suffix('.md').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(lines))
    print('PRECISION_EIGHT_GROUP_REPORT_VERIFIED',output,flush=True)


if __name__=='__main__':
    main()
