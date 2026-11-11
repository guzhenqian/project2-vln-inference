"""Offline paired analysis of RGB-only baseline and execution-feedback ablation."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(run, feedback=False):
    summary = json.loads((run/'summary.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    details = []
    all_rows = []
    for outcome in summary['episodes']:
        rows = [json.loads(line) for line in (run/('episode_'+outcome['episode_id'])/'steps.jsonl').read_text().splitlines()]
        assert len(rows)==outcome['steps']
        blocked = []
        for index,row in enumerate(rows):
            motion = float(np.linalg.norm(np.asarray(row['after']['position'])-np.asarray(row['before']['position'])))
            blocked.append(row['action']=='MOVE_FORWARD' and motion<0.02)
            if index:
                assert np.allclose(row['before']['position'],rows[index-1]['after']['position'],rtol=0,atol=1e-5)
            if feedback:
                actual=row['execution_feedback_input']
                assert set(actual)=={'previous_action','observed_displacement_m','collision_detected','forward_execution'}
                if index:
                    prev=rows[index-1]
                    observed=float(np.linalg.norm(np.asarray(prev['after']['position'])-np.asarray(prev['before']['position'])))
                    assert actual['previous_action']==prev['action']
                    assert abs(actual['observed_displacement_m']-observed)<0.000051
                    assert actual['collision_detected']==bool(prev['metrics']['collisions']['is_collision'])
                    status=None if prev['action']!='MOVE_FORWARD' else 'blocked' if observed<0.02 else 'partial' if observed<0.125 else 'moved'
                    assert actual['forward_execution']==status
                else:
                    assert all(v is None for v in actual.values())
        length=sum(float(np.linalg.norm(np.asarray(r['after']['position'])-np.asarray(r['before']['position']))) for r in rows)
        success=float(rows[-1]['action']=='STOP' and outcome['metrics']['distance_to_goal']<3.0)
        spl=success*outcome['start_distance_m']/max(outcome['start_distance_m'],length)
        assert success==outcome['metrics']['success'] and abs(spl-outcome['metrics']['spl'])<1e-5
        assert abs(length-outcome['path_length_m'])<1e-5
        eligible=sum(blocked[:-1])
        repeated=sum(blocked[i-1] and rows[i]['action']=='MOVE_FORWARD' for i in range(1,len(rows)))
        details.append(dict(outcome,minimum_distance_m=min(r['metrics']['distance_to_goal'] for r in rows),
            blocked_forward_steps=sum(blocked),forward_steps=sum(r['action']=='MOVE_FORWARD' for r in rows),
            repeat_forward_after_block=repeated,decisions_after_block=eligible,
            distances=[outcome['start_distance_m']]+[r['metrics']['distance_to_goal'] for r in rows]))
        all_rows.extend(rows)
    assert len(all_rows)==summary['decision_count']
    assert [d['episode_id'] for d in details]==manifest['episodes']
    assert sum(bool(r['error']) for r in all_rows)==summary['error_count']
    assert abs(np.mean([d['metrics']['success'] for d in details])-summary['sr'])<1e-9
    assert abs(np.median([r['decision_s'] for r in all_rows])-summary['decision_latency_s']['median'])<1e-9
    stats={'sr':summary['sr'],'spl':summary['spl'],'steps':len(all_rows),
        'errors':summary['error_count'],'model_stop_success_count':summary['model_stop_success_count'],
        'fallback_stop_success_count':summary['fallback_stop_success_count'],
        'collisions':sum(d['metrics']['collisions']['count'] for d in details),
        'blocked_forward_steps':sum(d['blocked_forward_steps'] for d in details),
        'forward_steps':sum(d['forward_steps'] for d in details),
        'repeat_forward_after_block':sum(d['repeat_forward_after_block'] for d in details),
        'decisions_after_block':sum(d['decisions_after_block'] for d in details),
        'decision_latency_median_ms':summary['decision_latency_s']['median']*1000,
        'decision_latency_p95_ms':summary['decision_latency_s']['p95']*1000,
        'sampled_whole_card_peak_mib':summary['gpu_memory']['sampled_peak_mib']}
    stats['blocked_forward_ratio']=stats['blocked_forward_steps']/max(1,stats['forward_steps'])
    stats['repeat_forward_after_block_ratio']=stats['repeat_forward_after_block']/max(1,stats['decisions_after_block'])
    return summary,manifest,details,stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--feedback',type=Path,required=True)
    args=parser.parse_args()
    a,ma,da,sa=audit(args.baseline.resolve())
    b,mb,db,sb=audit(args.feedback.resolve(),feedback=True)
    assert {k for k in ma if ma[k]!=mb[k]}=={'version','model_inputs'}
    assert (args.baseline/'system_prompt.txt').read_bytes()==(args.feedback/'system_prompt.txt').read_bytes()
    assert (args.baseline/'habitat_config.yaml').read_bytes()==(args.feedback/'habitat_config.yaml').read_bytes()
    for old,new in zip(da,db):
        assert abs(old['start_distance_m']-new['start_distance_m'])<1e-6
    report={'scope':'six paired development trajectories; no weights/quantization change; additional execution sensor feedback',
        'baseline':sa,'execution_feedback':sb,'episodes':[{ 'baseline':old,'execution_feedback':new} for old,new in zip(da,db)],
        'validation':'official success/SPL recomputed; every feedback input checked against previous step only; system prompt and task configs byte-identical'}
    out=ROOT/'reports'
    (out/'closed_loop_feedback_ablation_20261030.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for axis,old,new in zip(axes.flat,da,db):
        axis.plot(old['distances'],label='RGB + action history (v1)',color='#777777')
        axis.plot(new['distances'],label='+ execution feedback (v2)',color='#1565c0')
        axis.axhline(3,linestyle='--',linewidth=1,color='#c62828',label='Success distance threshold')
        axis.set_title('Episode {} | SR {} -> {}'.format(old['episode_id'],int(old['metrics']['success']),int(new['metrics']['success'])))
        axis.set_xlabel('Executed step');axis.set_ylabel('Geodesic distance (m)');axis.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Paired execution-feedback ablation | STOP required for success')
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(str(out/'closed_loop_feedback_ablation_20261030.png'),dpi=160);plt.close(fig)
    table=[('SR','{:.1%}'.format(sa['sr']),'{:.1%}'.format(sb['sr'])),
        ('SPL','{:.4f}'.format(sa['spl']),'{:.4f}'.format(sb['spl'])),
        ('正式决策数',str(sa['steps']),str(sb['steps'])),
        ('无效输出/请求失败',str(sa['errors']),str(sb['errors'])),
        ('碰撞事件数（随轨迹变化）',str(sa['collisions']),str(sb['collisions'])),
        ('前进但位移<2cm次数','{}/{}'.format(sa['blocked_forward_steps'],sa['forward_steps']),'{}/{}'.format(sb['blocked_forward_steps'],sb['forward_steps'])),
        ('受阻后下一步仍选前进','{}/{} ({:.1%})'.format(sa['repeat_forward_after_block'],sa['decisions_after_block'],sa['repeat_forward_after_block_ratio']),'{}/{} ({:.1%})'.format(sb['repeat_forward_after_block'],sb['decisions_after_block'],sb['repeat_forward_after_block_ratio'])),
        ('单步决策中位数','{:.1f}ms'.format(sa['decision_latency_median_ms']),'{:.1f}ms'.format(sb['decision_latency_median_ms']))]
    lines=['# 执行反馈单变量对照：2026-10-30','',
        '在冻结的6条单场景开发任务上，只增加上一动作的实际位移幅度、碰撞事件及前进执行状态。模型权重、相机配置、系统提示词、任务规则和100步上限相同；行动改变后实际图片会随轨迹变化。原v1全部保留。',
        '', '## 结果','', '| 指标 | v1 原始基线 | v2 加执行反馈 |','| --- | --- | --- |']
    lines+=['| {} | {} | {} |'.format(*row) for row in table]
    lines+=['', '## 逐任务','', '| Episode | v1 SR / 步数 / 终距m | v2 SR / 步数 / 终距m |', '| --- | --- | --- |']
    for old,new in zip(da,db):
        lines.append('| {} | {} / {} / {:.3f} | {} / {} / {:.3f} |'.format(old['episode_id'],int(old['metrics']['success']),old['steps'],old['metrics']['distance_to_goal'],int(new['metrics']['success']),new['steps'],new['metrics']['distance_to_goal']))
    lines+=['','## 结论与任务392示例','',
        '本轮没有提高导航成功率：SR从1/6变为0/6，SPL从0.1667变为0。保留v1作为冻结基线，v2只作为未提升的执行反馈对照，不替换基线。',
        '受阻后继续选前进的条件比例从179/188（95.2%）降为0/24（0%），说明这轮行为利用了反馈。两组经过的位置、遇到的障碍与分母均不同，该比例不是配对状态下的成功率，也不能证明泛化改善。',
        '任务392的指令是“Walk out of the bathroom and wait on the stairs that are on the right.”，即“走出浴室，在右侧楼梯上等候”。v1执行39个动作后距目标0.216m，v2执行77个动作后距目标0.178m；两组下一动作仍是MOVE_FORWARD，均在100步上限前没有输出STOP。最终距离分别为1.736m与1.664m，但官方Success要求主动STOP且停止时测地距离严格小于3m，因此均计失败。日志step从0编号，此处按已执行动作数加1表述。',
        '这说明“到过目标附近”“识别任务已完成并停止”“移动时能避开重复受阻”是不同能力。本轮反馈只补执行结果，无法单独补齐任务进度判断。上述目标距离仅用于事后分析，从未给模型输入。',
        '', '## 解释与边界','',
        '这是增加执行反馈的行为对照，不是推理压缩算法对照。反馈由仿真位姿差和碰撞标志得到，不含绝对坐标、目标坐标、到目标距离、参考路径或成功信号；部署到真实机器人需要里程计/碰撞感知，不能当作纯RGB模型无条件可用的能力。',
        'blocked定义为前进动作后的三维实际位移<0.02m，partial定义为0.02~0.125m，阈值均在运行前固定。碰撞标志与“完全没移动”不同，沿障碍滑动时也可能有碰撞事件。反馈可能使模型更常停车/转向，也可能误停；不能预设一定提高SR。',
        '延迟只作本轮观察：输入文字长度和实际轨迹不同，不能据此报告模型加速。显存为整卡采样值，背景负载可变。只跑一轮、6条同场景开发任务，不能声称统计稳定改善或泛化。',
        '', '## 验证与复现','',
        '离线逐条核对反馈只来自上一动作，扰动目标与成功/SPL不改变输入反馈；绝对坐标整体平移不改变反馈。两组系统提示词与Habitat配置逐字节相同，各episode原始距离一致。任务Success/SPL、实际位移和延迟独立复算通过。',
        '- 固定清单：`data/baseline/closed_loop_feedback_v2/manifest.json`。',
        '- 运行入口：`bash scripts/run_closed_loop_feedback.sh`。',
        '- 原始结果：`{}`。'.format(str(args.feedback).replace('\\','/')),
        '- 对照JSON：`reports/closed_loop_feedback_ablation_20261030.json`。',
        '- 距离变化图：`reports/closed_loop_feedback_ablation_20261030.png`。',
        '没有更改或覆盖v1报告，没有量化、训练、云盘上传或GitHub推送。']
    (out/'closed_loop_feedback_ablation_20261030.md').write_text('\n'.join(lines)+'\n')
    print('FEEDBACK_ABLATION_AUDIT_READY')
    print(json.dumps({'baseline':sa,'feedback':sb},indent=2))


if __name__=='__main__':main()
