"""Combine audited low-SR diagnostics, same-task checkpoint comparison and fresh scenes."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_navigation_sr import ROOT,audit


def interval(successes,count):
    # Wilson interval illustrates small-sample uncertainty; no independence claim across same-scene routes.
    z=1.959963984540054;p=successes/count;den=1+z*z/count
    center=(p+z*z/(2*count))/den
    radius=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [float(center-radius),float(center+radius)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-suite',type=Path,required=True)
    args=parser.parse_args()
    prior=json.loads((ROOT/'reports/navigation_sr_diagnosis_20261030.json').read_text())
    horizon=json.loads((ROOT/'reports/navigation_horizon_20261030.json').read_text())
    sft=horizon['h400']
    rl=audit(args.rl_suite.resolve()/'compare')
    fresh=audit(args.rl_suite.resolve()/'confirmation')
    for name in ['system_prompt.txt']:
        assert (ROOT/sft['path']/name).read_bytes()==(args.rl_suite/'compare'/name).read_bytes()
    assert (ROOT/sft['path']/'habitat_config.yaml').read_bytes()==(args.rl_suite/'compare'/'habitat_config.yaml').read_bytes()
    old=json.loads((ROOT/sft['path']/'manifest.json').read_text())
    new=json.loads((args.rl_suite/'compare/manifest.json').read_text())
    for name in ['episodes','trajectory_ids','max_episode_steps','rgb_hw','camera_height_m','temperature','top_p','seed','max_tokens','step_size_m','turn_deg','success_distance_m','model_inputs','action_parser','backend']:
        assert old[name]==new[name],name
    improved=[];regressed=[]
    for a,b in zip(sft['episodes'],rl['episodes']):
        assert a['episode_id']==b['episode_id'] and abs(a['start_distance_m']-b['start_distance_m'])<1e-5
        if b['metrics']['success']>a['metrics']['success']:improved.append(a['episode_id'])
        if b['metrics']['success']<a['metrics']['success']:regressed.append(a['episode_id'])
    assert not set(old['scenes']).intersection(json.loads((args.rl_suite/'confirmation/manifest.json').read_text())['scenes'])
    controls=json.loads((ROOT/'tmp/habitat_setup/metric_controls_confirmation_20261030/summary.json').read_text())
    assert len(controls['results'])==12
    for result in controls['results']:
        assert result['metrics']['success']==int(result['controller']=='reference_follower')
    for r in [sft,rl,fresh]:r['illustrative_wilson_interval']=interval(sum(x['metrics']['success'] for x in r['episodes']),r['count'])
    adopted=rl['errors']==0 and fresh['errors']==0 and rl['sr']>sft['sr'] and fresh['sr']>0
    conclusion='采用公开RL导航权重作为后续固定精度基线候选，保留SFT与通用2B对照；下一轮压缩前应扩充并冻结评测/校准划分' if adopted else '本轮未证明RL权重具备足够且一致的优势，保留SFT已验证基线与所有失败证据；不宣称低SR已彻底解决'
    report={'scope':'small-sample navigation quality investigation, not full benchmark reproduction or quantization result',
        'sft_h400':sft,'rl_same_tasks_h400':rl,'rl_fresh_scenes_h400':fresh,
        'paired_improved_episode_ids':improved,'paired_regressed_episode_ids':regressed,
        'conclusion':conclusion,'rl_candidate_adopted':adopted,
        'validation':'same task config/system prompt/input/decode/precision; source-config and image/token equality checks; all saved trajectory/images/metrics audited'}
    out=ROOT/'reports'
    (out/'navigation_sr_final_20261030.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(2,3,figsize=(12,6.7))
    for axis,a,b in zip(axes.flat,sft['episodes'],rl['episodes']):
        for e,label,color in [(a,'SFT','#777777'),(b,'RL','#1565c0')]:
            axis.plot(e['distances'],label=label,color=color)
            if e['stop_called']:axis.scatter(e['steps'],e['distances'][-1],color=color,marker='X',s=35,zorder=5)
        axis.axhline(3,color='#c62828',linestyle='--',linewidth=1)
        axis.set_title('Episode {} | SR {} -> {}'.format(a['episode_id'],int(a['metrics']['success']),int(b['metrics']['success'])))
        axis.set_xlabel('Primitive action');axis.set_ylabel('Goal distance (m)');axis.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2);fig.suptitle('Same 4B architecture and navigation interface | 400-step cap | STOP required')
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(str(out/'navigation_sr_sft_rl_20261030.png'),dpi=160);plt.close(fig)
    rows=[('通用2B，100步，开发',prior['runs']['rgb_dev']),('通用2B，100步，独立A',prior['runs']['rgb_holdout']),
        ('导航SFT4B，100步，开发',prior['runs']['trained_dev']),('导航SFT4B，100步，独立A',prior['runs']['trained_holdout']),
        ('导航SFT4B，400步，独立A',sft),('导航RL4B，400步，同组A',rl),('导航RL4B，400步，新组B',fresh)]
    lines=['# 低SR问题：完整处理结果与原因分析','',
        '结论：'+conclusion+'。这解决的是基线选型与可评测性问题；不将有限任务上的改善称为所有导航失败已解决。',
        '', '## 全部关键结果','', '| 配置与任务组 | 成功/任务数 | SR | SPL | 接口错误 |', '| --- | --- | --- | --- | --- |']
    for label,r in rows:
        lines.append('| {} | {}/{} | {:.1%} | {:.4f} | {} |'.format(label,sum(int(e['metrics']['success']) for e in r['episodes']),r['count'],r['sr'],r['spl'],r['errors']))
    lines+=['','开发组为最早的单场景6任务；A组为另外3场景6任务；B组为再另外3场景6任务。A在本轮中作为诊断和权重比较集，已不应继续称为完全未触碰测试集；B的任务清单和规则在第一次RL推理前固定，不用于调提示词。公开模型是否曾接触相关预训练数据不能由本实验排除，“新”指本项目未用于策略调试的场景，来源为官方val_unseen。',
        '', '## 为什么原来SR低','',
        '**首先是基线选型不完整。** 原checkpoint是通用Qwen3-VL-2B-Instruct。把它连到模拟器、要求输出四类动作，完成的是一个零样本导航接口；并不等于已经接入经过VLN训练的导航模型。原接口只提供当前图和最近动作名，缺少历史视觉证据和实际执行反馈。模型可能看懂单帧物体，却不会稳定完成整条多阶段路线。此前将“模型部署跑通”和“可靠VLN基线”靠得太近，这是需要纠正的工程判断。',
        '**第二是路线执行与停止判断分别失败。** 例如任务392，原2B曾距目标0.216m，反馈版曾到0.178m，但都继续前进并最终超时。官方Success要求主动STOP且停止时测地距离<3m，路过不算成功。API的finish_reason="stop"只表示文本生成结束，不是导航STOP。任务701也有经过目标附近后离开的记录。',
        '**第三是动作命令不等于实际运动。** 原模型在受阻后179/188次仍选前进。补上一动作实际位移和碰撞后降为0/24，但SR从1/6变为0/6。这验证了反馈能影响行为，也证明单纯避免重复受阻不足以完成路线。两组经历状态不同，不能把这个条件比例当作同状态配对实验。',
        '**第四是自写记忆可能携带错误。** 一次视觉历史与文字进度候选在开发/独立A上都是0/6，且发生2/1次输出截断。它既存在接口格式限制，也有有效回答中的错认门、捏造路线进度和提前停止。我们没有采用该候选，没有把它的全部错误简单归为模型语义能力。',
        '**第五，100步限制解释不了全部失败，但评测上限确实重要。** 固定SFT权重与A组任务，将超时路线从原前缀继续到400步，SR仍2/6、SPL仍0.3191。全部522个原始前缀动作逐像素/位姿/距离重放核对，已停止任务保持结果；没有用目标真值修正策略。另一方面，RL在B组46号任务到第315步才成功，SPL约0.299，说明长路线会受上限影响，不能由SFT无收益推断所有模型都不需要较长上限。最终导航候选采用400步，原100步成绩单独保留。',
        '', '## 实际做了哪些处理','',
        '1. 修复独立仿真环境依赖和二进制兼容，冻结已有构建；立即STOP负对照与参考控制正对照在三组各6/6按预期失败/成功，检查路长与SR/SPL。参考控制仅验证工具链，不计入模型成绩。',
        '2. 接入作者公开的SimpleVLN Qwen3-VL4B导航SFT权重和对应接口：当前RGB+有界历史观察、模型预测动作块，代替原通用2B四动作提示词。成功率与输入/计算成本均单独记录，不能将整体提升归因为训练一个因素。',
        '3. 用同一4B架构比较公开SFT与RL权重。713个张量名称/形状相同；服务侧图像与分词输入张量逐元素一致，系统提示词、任务配置、解码参数、动作解析和BF16运行精度一致。RL源文件为F32，在加载时转BF16；原始权重未改，没有在本机训练或做AWQ。两组缓存块自动分配略有差异，只用于质量对照，不报告其为速度加速实验。',
        '4. 在同任务A组检查净改善，并用未用于前面调试的新场景B组确认。所有任务都保留，失败没有删除，不用参考路径、目标坐标或距离决定动作。',
        '', '## 127号任务的具体证据','',
        '指令要求转身180°、穿过门进入阁楼、左转走向远端，并在刚经过楼梯开口后停止。SFT在目标附近往返，400步时距目标1.072m但未输出STOP，SR=0。RL第58个动作的原始回答为单词`stop`，当时终距2.176m，Success=1、SPL=0.6142，既没有接口兜底也没有真值阈值代替模型决策。停止位置距标注目标仍有2.176m且路线存在绕行，SPL并不满分。',
        '3m容差意味着该成功只证明满足本轮官方度量条件，不证明完整路线语义、每处转弯或最终朝向逐项正确。画面审阅与轨迹分析应与SR分开，不能把一次合格停止当作导航能力全面通过。',
        '', '## 同任务权重比较','',
        'A组由SFT切换到RL，新增成功任务：{}；原成功变失败任务：{}。这是一轮固定种子采样结果，小样本、同场景相关性与随机解码都会限制结论，不能从六条任务估计稳定的全数据集分数。'.format(', '.join(improved) or '无',', '.join(regressed) or '无'),
        '', '![同任务SFT与RL轨迹](navigation_sr_sft_rl_20261030.png)',
        '', '| Episode | SFT SR / SPL / 步数 | RL SR / SPL / 步数 | RL终距m | RL失败诊断 |', '| --- | --- | --- | --- | --- |']
    for a,b in zip(sft['episodes'],rl['episodes']):
        lines.append('| {} | {} / {:.3f} / {} | {} / {:.3f} / {} | {:.3f} | {} |'.format(a['episode_id'],int(a['metrics']['success']),a['metrics']['spl'],a['steps'],int(b['metrics']['success']),b['metrics']['spl'],b['steps'],b['metrics']['distance_to_goal'],b['failure_category']))
    lines+=['','## 新场景B组确认','', '| Episode | SR | SPL | 步数 | 终距m | 最近距离m | 诊断 |','| --- | --- | --- | --- | --- | --- | --- |']
    for e in fresh['episodes']:
        lines.append('| {} | {} | {:.3f} | {} | {:.3f} | {:.3f} | {} |'.format(e['episode_id'],int(e['metrics']['success']),e['metrics']['spl'],e['steps'],e['metrics']['distance_to_goal'],e['minimum_distance_m'],e['failure_category']))
    lines+=['','## 为什么这仍不是全量复现或加速结果','',
        '- 数据规模有限：每组只有6条任务、3个独立场景，路线之间有相关性；不得将该SR写成整个R2R-CE成绩。后续需在独立校准/评测划分上扩大任务量，压缩比较时固定同一任务集。',
        '- 运行环境保留已验证的Habitat0.1.7和LMDeploy；作者公开评测使用Habitat0.3.3及vLLM，部分早期协议相机/步数也不同。这里是本机导航基线验证，不宣称复现作者论文分数。',
        '- 4B和多帧观察比2B单帧更耗算力。下面模型请求耗时只统计真实请求，不把动作队列的近零执行耗时当作推理加速。原始runner summary的逐动作延迟含队列执行，应以本报告从model_called筛出的请求统计为准。整卡显存200ms采样含桌面；不是模型独占峰值。',
        '', '| 运行 | 原子动作 | 真模型请求 | 请求中位/P95 ms | 整卡峰值MiB |','| --- | --- | --- | --- | --- |']
    for label,r in [('2B单帧A',prior['runs']['rgb_holdout']),('SFT4B A(100步)',prior['runs']['trained_holdout']),('RL4B A(400步)',rl),('RL4B B(400步)',fresh)]:
        lines.append('| {} | {} | {} | {:.1f} / {:.1f} | {} |'.format(label,r['steps'],r['model_requests'],r['request_median_ms'],r['request_p95_ms'],r['whole_card_peak_mib']))
    lines+=['','## 后续工程结论','',conclusion+'。',
        '后续量化与Token压缩只改变待测优化项，固定导航模型版本、任务清单、相机、历史采样、动作接口及停止规则，同时记录SR/SPL、真实模型决策延迟与显存。不能一边换模型/提示词一边把差异当作量化效果。',
        '', '## 可追溯证据','',
        '- 原始2B、SFT及失败候选：[前段完整分析](navigation_sr_diagnosis_20261030.md)。',
        '- 100/400步：[上限敏感性](navigation_horizon_20261030.md)。',
        '- 本轮原始轨迹：`{}`。'.format(args.rl_suite.as_posix()),
        '- 完整指标JSON：`reports/navigation_sr_final_20261030.json`。',
        '- 来源：[SimpleVLN官方仓库](https://github.com/simplevln/SimpleVLN)、[公开RL权重](https://huggingface.co/honglyhly/SimpleVLN_Qwen3VL_4B_RL)。公开训练流程说明用于解释模型差别，所有本机成绩均来自本地轨迹实测。']
    (out/'navigation_sr_final_20261030.md').write_text('\n'.join(lines)+'\n')
    print('FINAL_NAVIGATION_SR_AUDIT_READY')
    print(json.dumps({'sft_sr':sft['sr'],'rl_compare_sr':rl['sr'],'rl_confirmation_sr':fresh['sr'],
        'rl_compare_spl':rl['spl'],'rl_confirmation_spl':fresh['spl'],'improved':improved,'regressed':regressed,'adopted':adopted},indent=2))


if __name__=='__main__':main()
