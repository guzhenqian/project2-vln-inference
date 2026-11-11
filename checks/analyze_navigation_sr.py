"""Audit all diagnostic trajectories and explain the low-SR investigation without new inference."""
import argparse
import gzip
import hashlib
import json
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def audit(path):
    summary=json.loads((path/'summary.json').read_text())
    manifest=json.loads((path/'manifest.json').read_text())
    results=[];records=[];requests=[]
    for result in summary['episodes']:
        rows=[json.loads(line) for line in (path/('episode_'+result['episode_id'])/'steps.jsonl').read_text().splitlines()]
        assert len(rows)==result['steps']
        length=0
        for index,row in enumerate(rows):
            assert row['step']==index
            if index:assert np.allclose(row['before']['position'],rows[index-1]['after']['position'],rtol=0,atol=1e-5)
            length+=float(np.linalg.norm(np.array(row['after']['position'])-np.array(row['before']['position'])))
            assert (path/row['frame']).exists()
            if 'visual_history_steps' in row:assert all(0<=i<=index for i in row['visual_history_steps'])
            if row.get('model_called'):
                history=list(range(max(0,index-199),index))
                if len(history)>8:history=[history[round(i*(len(history)-1)/7)] for i in range(8)]
                indices=(history or [index])+[index]
                expected=[]
                for frame_index in indices:
                    frame_path=path/('episode_'+result['episode_id'])/('frame_{:04d}.png'.format(frame_index))
                    buffer=BytesIO()
                    with Image.open(str(frame_path)) as image:image.convert('RGB').save(buffer,format='JPEG')
                    expected.append(hashlib.sha256(buffer.getvalue()).hexdigest())
                assert expected==row['request_image_sha256'], 'Image history mismatch or future-frame leakage'
                assert len(expected)==row['image_count']
            if row.get('model_called',True):requests.append(row)
        success=float(rows[-1]['action']=='STOP' and result['metrics']['distance_to_goal']<3)
        spl=success*result['start_distance_m']/max(result['start_distance_m'],length)
        assert success==result['metrics']['success'] and abs(spl-result['metrics']['spl'])<1e-5
        assert abs(length-result['path_length_m'])<1e-5
        assert sum(bool(row['error']) for row in rows)==result['error_count']
        reached=min([result['start_distance_m']]+[row['metrics']['distance_to_goal'] for row in rows])<3
        category='success' if success else 'interface_error' if result['error_count'] else 'passed_near_goal_without_valid_stop' if reached else 'route_or_destination_error'
        # A queued STOP must be traceable to the latest actual model response.
        if success:
            last_call=next(row for row in reversed(rows) if row.get('model_called',True))
            assert not rows[-1]['error'] and 'stop' in last_call['raw'].lower()
        results.append(dict(result,oracle_reached=bool(reached),failure_category=category,
            minimum_distance_m=min(row['metrics']['distance_to_goal'] for row in rows),
            distances=[result['start_distance_m']]+[row['metrics']['distance_to_goal'] for row in rows]))
        records.extend(rows)
    assert [result['episode_id'] for result in results]==manifest['episodes']
    assert len(records)==summary['decision_count'] and sum(bool(r['error']) for r in records)==summary['error_count']
    sr=float(np.mean([r['metrics']['success'] for r in results]));spl=float(np.mean([r['metrics']['spl'] for r in results]))
    assert abs(sr-summary['sr'])<1e-9 and abs(spl-summary['spl'])<1e-9
    return {'path':path.relative_to(ROOT).as_posix(),'sr':sr,'spl':spl,'oracle_reached_rate':float(np.mean([r['oracle_reached'] for r in results])),
        'count':len(results),'steps':len(records),'model_requests':len(requests),'errors':sum(bool(r['error']) for r in records),
        'truncations':sum(r.get('finish_reason')=='length' for r in records),
        'request_median_ms':float(np.median([r['model_request_s'] for r in requests]))*1000,
        'request_p95_ms':float(np.percentile([r['model_request_s'] for r in requests],95))*1000,
        'decision_call_median_ms':float(np.median([r['decision_s'] for r in requests]))*1000,
        'whole_card_peak_mib':summary['gpu_memory']['sampled_peak_mib'],'episodes':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained-suite',required=True,type=Path)
    args=parser.parse_args()
    runs={
        'rgb_dev':ROOT/'logs/closed_loop_v1_20261030T075256Z_45677',
        'feedback_dev':ROOT/'logs/closed_loop_feedback_v2_20261030T081809Z_45982',
        'memory_dev':ROOT/'logs/navigation_diagnostic_20261030T083920Z_46347/memory_dev',
        'rgb_holdout':ROOT/'logs/navigation_diagnostic_20261030T083920Z_46347/rgb_holdout',
        'memory_holdout':ROOT/'logs/navigation_diagnostic_20261030T083920Z_46347/memory_holdout',
        'trained_dev':args.trained_suite.resolve()/'dev','trained_holdout':args.trained_suite.resolve()/'holdout'}
    audited={key:audit(path) for key,path in runs.items()}
    for suffix in ['dev','holdout']:
        old=audited['rgb_'+suffix]['episodes'];new=audited['trained_'+suffix]['episodes']
        assert [r['trajectory_id'] for r in old]==[r['trajectory_id'] for r in new]
        assert all(abs(a['start_distance_m']-b['start_distance_m'])<1e-5 for a,b in zip(old,new))
    freeze=json.loads((ROOT/'reports/closed_loop_baseline_v1_20261030_freeze.json').read_text())
    for name,record in freeze['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==record['sha256'],name
    out=ROOT/'reports'
    result={'scope':'quality diagnosis and replacement navigation baseline; no quantization; not a paper benchmark reproduction',
        'validation':'all positions/paths/SR/SPL/queued model STOP/error denominators audited; fixed task IDs and initial distances; original frozen files unchanged',
        'runs':audited}
    (out/'navigation_sr_diagnosis_20261030.json').write_text(json.dumps(result,indent=2))
    fig,axis=plt.subplots(figsize=(9,4.5))
    for key,label,color in [('rgb_dev','Generic 2B RGB-only','#777777'),('feedback_dev','Generic 2B + execution feedback','#c47c0a'),('trained_dev','Navigation-trained 4B + author interface','#1565c0')]:
        episode=next(r for r in audited[key]['episodes'] if r['episode_id']=='392')
        axis.plot(episode['distances'],label=label,color=color)
        if episode['stop_called']:axis.scatter(episode['steps'],episode['distances'][-1],s=65,color=color,marker='X',zorder=5)
    axis.axhline(3,color='#c62828',linestyle='--',label='STOP must occur below 3m')
    axis.set(xlabel='Executed primitive actions',ylabel='Geodesic goal distance (m)',title='Episode 392: reaching the stairs and knowing when to stop')
    axis.grid(alpha=.2);axis.legend(fontsize=8);fig.tight_layout()
    fig.savefig(str(out/'navigation_sr_episode392_20261030.png'),dpi=170);plt.close(fig)
    names=[('原始通用2B','rgb_dev','rgb_holdout'),('2B+执行反馈','feedback_dev',None),('2B+视觉历史/文字记忆','memory_dev','memory_holdout'),('导航训练4B+作者接口','trained_dev','trained_holdout')]
    lines=['# 低SR诊断、处理与实测分析：2026-10-30','',
        '本报告区分接口故障、导航策略缺陷和基线选型。原始2B基线始终保留，导航训练4B作为独立候选；没有训练、量化或Token压缩。',
        '', '## 实测结果','', '| 方案 | 原场景6任务 SR / SPL | 3个独立场景6任务 SR / SPL | 接口错误（原场景/独立场景） |','| --- | --- | --- | --- |']
    for name,dev,held in names:
        a=audited[dev];b=audited.get(held)
        lines.append('| {} | {:.1%} / {:.4f} | {} | {} / {} |'.format(name,a['sr'],a['spl'],
            '{:.1%} / {:.4f}'.format(b['sr'],b['spl']) if b else '未运行',a['errors'],b['errors'] if b else '—'))
    lines+=['','## 先排除指标和环境故障','',
        '原场景与独立场景均做立即STOP负对照和参考路径控制正对照，各组6/6按预期失败/成功。独立场景参考控制需37、59、78、43、43、59步，均小于固定100步上限。这证明100步内存在可行路线，不证明模型100步足够；模型绕行导致超时仍按失败计，延长上限需另立配置，不补改本轮成绩。控制器读取参考路径，只验证可达性，不计入模型成绩。所有真实轨迹用位姿独立复算路长/SPL，并检查STOP与3m成功条件。',
        '原始v1的487个冻结文件哈希保持不变。早期依赖、execstack、headless、imageio错误已处理；本轮原始2B与反馈组均0请求/解析错误，因此它们低SR不能用“模型没真正跑起来”解释。',
        '', '## 从轨迹确认了哪些问题','',
        '1. **缺少执行结果会导致重复受阻。** 2B只看动作名称时，不知道前进是否成功。补真实位移幅度/碰撞后，受阻后继续前进179/188→0/24，但SR1/6→0/6。说明它响应了反馈，也说明减少撞墙不足以完成导航。条件分母来自不同轨迹，不能当作同状态对比或泛化结论。',
        '2. **到过目标附近与主动结束任务不同。** 原2B任务392在第39个动作后距目标0.216m，却继续前进；反馈版第77个动作后0.178m也不停。两者100步结束时仍在3m内，但没有STOP，Success=0。不能把日志API的finish_reason="stop"当作导航STOP；前者只表示文字生成结束。',
        '3. **提示词和自写记忆可能放大错误。** 单次历史/记忆候选没有改善；其中还出现3次160-token输出截断，已明确记为接口失败并拒绝采用，不能把这些错误全算为导航推理失败。有效回答也出现错认门、错误陈述已经走过的路线与提前停止，说明记忆内容不能自动当成事实。',
        '4. **原始checkpoint与任务不匹配。** 本地原模型是通用Qwen3-VL-2B-Instruct；原工程接口不等于已经训练好的VLN策略。当前已接入作者公开的SimpleVLN Qwen3-VL 4B SFT导航权重及相匹配的历史观察/动作接口。成功率提升若成立，应归因于整体导航策略替换，不能仅归因于参数规模、训练或历史中的一个因素。',
        '', '## 任务392具体例子','',
        '指令：Walk out of the bathroom and wait on the stairs that are on the right.（走出浴室，在右侧楼梯上等候。）',
        '原始2B和执行反馈2B均曾进入目标附近，却没有主动STOP。导航训练4B的全部原始模型回答与逐步动作已保存，成功停止须能追溯到实际模型输出；动作队列中的STOP也不来自目标距离阈值。最终成绩见上表与下面逐任务清单。',
        '![392距离变化](navigation_sr_episode392_20261030.png)',
        '', '## 新策略究竟改变了什么','',
        '- 权重：作者公开BF16导航训练4B SFT；revision `2b2ae6e30f3e344fb9c6d2ed11ee9c117ac24566`。两片权重LFS SHA校验通过，原始2B与新权重并存。',
        '- 输入：作者策略使用当前RGB及最近200观察内均匀采样最多8帧历史；RGB640×480、HFOV90、相机1.25m。没有目标、真实距离、参考路径、GPS或文字进度记忆。',
        '- 输出：模型预测动作块，采用前两条子指令、每条最多3原子动作；每步仍0.25m/15°，官方Success/SPL不变。无效随机兜底被显式报错终止，避免将随机成功冒充模型成功。',
        '- 部署：既有LMDeploy BF16，context4096，在12GB显卡上运行。仅对独立服务视图做Transformers5→4的RoPE与特殊token配置格式转换，不改权重、词表或原下载配置。',
        '', '## 成本与评测边界','',
        '新模型更大、视觉历史与分辨率更高，质量替换不是推理加速结果。队列执行不调用模型，不能把这些近零耗时当作模型延迟样本。下表只统计真实请求；显存是整卡200ms采样，含桌面，不是模型独占精确峰值。',
        '', '| 运行 | 原子动作数 | 模型请求数 | 请求中位/P95 ms | 整卡采样峰值MiB |','| --- | --- | --- | --- | --- |']
    for key in ['rgb_dev','rgb_holdout','trained_dev','trained_holdout']:
        a=audited[key];lines.append('| {} | {} | {} | {:.1f} / {:.1f} | {} |'.format(key,a['steps'],a['model_requests'],a['request_median_ms'],a['request_p95_ms'],a['whole_card_peak_mib']))
    lines+=['',
        '本轮为每组6条固定任务、100步上限，独立检查仅3场景。作者公开评测为Habitat0.3.3/vLLM与400步；这里保留已验证的Habitat0.1.7和LMDeploy。因此不得将本机小样本分数宣称为论文复现或全量R2R成绩。小样本也不能给出稳健的总体成功率估计。独立任务清单先于候选推理固定，没有根据候选成绩选样或删除失败。',
        '', '## 导航训练策略逐任务','', '| 分组 | Episode | SR | SPL | 步数 | 最终距离m | 最近距离m | 诊断 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for key in ['trained_dev','trained_holdout']:
        for a in audited[key]['episodes']:
            lines.append('| {} | {} | {} | {:.4f} | {} | {:.3f} | {:.3f} | {} |'.format(key,a['episode_id'],int(a['metrics']['success']),a['metrics']['spl'],a['steps'],a['metrics']['distance_to_goal'],a['minimum_distance_m'],a['failure_category']))
    lines+=['','## 来源与复现','',
        '作者方法、模型和训练流程：[SimpleVLN官方仓库](https://github.com/simplevln/SimpleVLN)。作者评测接口与设置：[官方评测说明](https://github.com/simplevln/SimpleVLN/blob/a450c57cff323587f54b0654c8e20e530a7202ac/eval/README.md)。公开权重：[4B SFT](https://huggingface.co/honglyhly/SimpleVLN_Qwen3VL_4B_SFT)。文中的本机结果均来自本地轨迹独立复算，不引用论文成绩代替实测。',
        '- 原始运行：`{}`。'.format(args.trained_suite.as_posix()),
        '- 入口：`bash scripts/run_simplevln_diagnostic.sh`。',
        '- 数据/配置：`data/baseline/simplevln_4b_sft_dev/manifest.json`、`simplevln_4b_sft_holdout/manifest.json`。',
        '- 原2B基线报告和反馈对照报告完整保留。',
        '- 本报告由`checks/analyze_navigation_sr.py`离线复算生成。']
    (out/'navigation_sr_diagnosis_20261030.md').write_text('\n'.join(lines)+'\n')
    print('NAVIGATION_SR_DIAGNOSIS_AUDIT_READY')
    print(json.dumps({key:{k:v for k,v in a.items() if k!='episodes'} for key,a in audited.items()},indent=2))


if __name__=='__main__':main()
