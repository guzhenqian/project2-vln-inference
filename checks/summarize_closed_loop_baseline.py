"""Independently audit the saved closed-loop trace and produce a baseline report."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
args = parser.parse_args()
run = args.run.resolve()
summary = json.loads((run/'summary.json').read_text())
manifest = json.loads((run/'manifest.json').read_text())
dataset = json.loads(gzip.decompress((ROOT/'data/datasets/R2R_VLNCE_v1-3_smoke/val_seen/val_seen.json.gz').read_bytes()).decode())
episodes = {str(e['episode_id']):e for e in dataset['episodes']}
all_rows = []
audits = []
fig,axes = plt.subplots(2,3,figsize=(13,8))
for axis,result in zip(axes.flat,summary['episodes']):
    directory = run/('episode_'+result['episode_id'])
    rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    assert len(rows)==result['steps']
    assert all(row['episode_id']==result['episode_id'] for row in rows)
    positions = [rows[0]['before']['position']]+[r['after']['position'] for r in rows]
    assert all(np.allclose(a['after']['position'],b['before']['position'],atol=1e-6) for a,b in zip(rows,rows[1:]))
    assert np.allclose(positions[0],episodes[result['episode_id']]['start_position'],atol=1e-5)
    length = sum(float(np.linalg.norm(np.array(b)-a)) for a,b in zip(positions,positions[1:]))
    assert abs(length-result['path_length_m'])<1e-5
    valid_stop = rows[-1]['action']=='STOP' and not rows[-1]['error']
    success = float(rows[-1]['action']=='STOP' and result['metrics']['distance_to_goal']<3.0)
    spl = success*result['start_distance_m']/max(result['start_distance_m'],length)
    assert success==result['metrics']['success']
    assert abs(spl-result['metrics']['spl'])<1e-5
    assert all((directory/Path(r['frame']).name).is_file() for r in rows)
    distances = [r['metrics']['distance_to_goal'] for r in rows]
    audit = dict(result,minimum_distance_m=min(distances),valid_terminal_model_stop=valid_stop,
                 markdown_normalizations=sum(bool(r.get('markdown_fence_removed')) for r in rows))
    audits.append(audit)
    all_rows.extend(rows)
    reference = np.array(episodes[result['episode_id']]['reference_path'])
    path = np.array(positions)
    axis.plot(reference[:,0],reference[:,2],'--',color='#999999',label='Reference waypoints')
    axis.plot(path[:,0],path[:,2],color='#1565C0',label='Model trajectory')
    axis.scatter(path[0,0],path[0,2],s=35,c='#2e7d32',label='Start',zorder=4)
    axis.scatter(path[-1,0],path[-1,2],s=40,c='#1565C0',marker='s',label='Final',zorder=4)
    goal = episodes[result['episode_id']]['goals'][0]['position']
    axis.scatter(goal[0],goal[2],s=70,c='#c62828',marker='x',label='Goal',zorder=5)
    axis.set_title('Episode {} | SR={} | {} steps'.format(result['episode_id'],int(success),len(rows)))
    axis.set_xlabel('X (m)');axis.set_ylabel('Z (m)')
    axis.set_aspect('equal',adjustable='datalim');axis.grid(alpha=.2)
handles,labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False)
fig.suptitle('Qwen3-VL-2B BF16 | Six fixed CE development trajectories | 100-step horizon')
fig.tight_layout(rect=(0,.05,1,.95))
out = ROOT/'reports'
fig.savefig(str(out/'closed_loop_baseline_v1_20261030_paths.png'),dpi=160)
plt.close(fig)
assert [r['episode_id'] for r in audits]==manifest['episodes']
assert len(all_rows)==summary['decision_count']
assert sum(bool(r['error']) for r in all_rows)==summary['error_count']==0
assert sum(a['metrics']['success'] for a in audits)==summary['model_stop_success_count']
assert summary['fallback_stop_success_count']==0
for name,field in [('model_request_latency_s','model_request_s'),('decision_latency_s','decision_s')]:
    assert abs(np.median([r[field] for r in all_rows])-summary[name]['median'])<1e-9
summary['episode_audit']=audits
summary['sim_step_latency_s']={'median':float(np.median([r['sim_step_s'] for r in all_rows])),
                              'p95':float(np.percentile([r['sim_step_s'] for r in all_rows],95))}
summary['markdown_normalization_count']=sum(a['markdown_normalizations'] for a in audits)
summary['total_completion_tokens']=sum((r.get('usage') or {}).get('completion_tokens',0) for r in all_rows)
(out/'closed_loop_baseline_v1_20261030.json').write_text(json.dumps(summary,indent=2))
lines = ['# Qwen3-VL-2B BF16：首版固定任务闭环基线','',
    '2026-10-30，closed_loop_v1。完成一个 MP3D 场景内6条不同轨迹的开发基线；不是全数据集或未见场景泛化结论。',
    '', '## 结果','', '| 指标 | 实测 |','| --- | --- |',
    '| SR | {:.2%}（{}/6） |'.format(summary['sr'],summary['model_stop_success_count']),
    '| SPL | {:.4f} |'.format(summary['spl']),
    '| 单步决策耗时 中位 / P95 | {:.1f} / {:.1f} ms |'.format(summary['decision_latency_s']['median']*1000,summary['decision_latency_s']['p95']*1000),
    '| 模型 HTTP 请求耗时 中位 / P95 | {:.1f} / {:.1f} ms |'.format(summary['model_request_latency_s']['median']*1000,summary['model_request_latency_s']['p95']*1000),
    '| 整卡显存采样最大值 | {:.0f} MiB（约 {:.2f} GiB） |'.format(summary['gpu_memory']['sampled_peak_mib'],summary['gpu_memory']['sampled_peak_mib']/1024),
    '| 正式决策请求 / 无效输出 | {} / {} |'.format(summary['decision_count'],summary['error_count']),
    '', '排除2次预热。决策耗时包括图像编码、构造请求、HTTP/服务执行及结果解析，不含模拟器执行和调试PNG落盘；HTTP耗时不是纯GPU计算。P95仅为本次样本分位数。',
    '显存为200ms间隔的整卡已用显存采样，含桌面等负载；不等于模型独占的精确allocator峰值。仿真使用CPU Mesa llvmpipe，不能将本结果称为GPU渲染性能。',
    '', '## 固定配置','',
    'Qwen3-VL-2B-Instruct BF16；既有LMDeploy0.11.1 PyTorch服务，单请求，temperature0/max_tokens32。模型源revision 89644892e4d85e24eaac8bacfd4f463576704203，权重校验见freeze清单。',
    '固定e9zR4mvMWw7场景，CE val_seen的6条不同trajectory，各取最小episode_id指令。前进0.25m、转向15°；RGB224×224、HFOV90°、传感器高度1.25m；滑动允许；最多100步。100步短于官方配置500步，已在任何模型调用前固定。',
    '模型输入仅原始指令、当前RGB、最近20个动作；不提供目标坐标、参考路径、真实距离或指标。动作是STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT。这是新增离散导航接口，未改原业务航点接口；不能与旧离线单帧延迟直接算加速比。',
    'Success要求STOP且最终navmesh测地距离严格小于3m；SPL使用官方Habitat-Lab0.1.7实现。存在3m终点容差，因此成功任务可能在精确目标前停止，且SPL封顶1；这不证明完整复现原始指令语义。',
    '', '## 每条任务','', '| Episode | 步数 | 最终距离 m | 最小距离 m | STOP/终止 | SR | SPL |','| --- | --- | --- | --- | --- | --- | --- |']
for a in audits:
    lines.append('| {} | {} | {:.3f} | {:.3f} | {} | {} | {:.3f} |'.format(a['episode_id'],a['steps'],a['metrics']['distance_to_goal'],a['minimum_distance_m'],a['terminal_stop_source'],int(a['metrics']['success']),a['metrics']['spl']))
lines += ['', '188任务在目标1.967m处由模型主动STOP，符合已固定成功条件。392任务最终距离1.736m但没有STOP，达到100步上限，所以成功率和SPL均为0。281、392、743有大量模拟器碰撞/受阻事件；701存在反复转向和离开目标附近的行为。',
    '', '## 评测链路验证','',
    '同6条任务的立即STOP负对照：6/6失败；参考路径跟随正对照：6/6成功。对照和模型每条任务的真实位置序列均独立复算路径长度及SPL，与官方测量一致。控制器读取参考路径，仅用于测试评测链路，不算模型成绩。',
    '', '## 首轮接口故障与最终冻结','',
    'v0首轮使用严格JSON解析，3次合法动作JSON带代码围栏被当无效并兜底STOP。其唯一成功是兜底停止，不作为模型有效成功。v0原始文件完整保留。v1只增加剥除一个完整Markdown代码围栏的格式适配，配置和提示词不变；结果另存，未覆盖v0。',
    'v1共{}次围栏规范化，0次无效动作/请求失败，成功任务由模型主动STOP触发。'.format(summary['markdown_normalization_count']),
    '', '## 复现与证据','',
    '- 服务入口：`bash scripts/start_lmdeploy_2b.sh`（已运行时不要重复启动）。',
    '- 基线入口：`bash scripts/run_closed_loop_baseline.sh`。输出新目录，避免覆盖。',
    '- 固定清单：`data/baseline/closed_loop_v1/manifest.json`。',
    '- 本次原始输出：`{}`，含全部{}张决策前PNG、动作回答/实际位姿/指标JSONL、最终图片、显存采样、配置与源哈希。'.format(str(run.relative_to(ROOT)).replace('\\','/'),len(all_rows)),
    '- 指标对照：`tmp/habitat_setup/metric_controls_20261030_six/summary.json`。',
    '- 汇总：`reports/closed_loop_baseline_v1_20261030.json`。',
    '- 轨迹图：`reports/closed_loop_baseline_v1_20261030_paths.png`，灰线是reference_path连线，不是障碍地图。',
    '', '当前基础链路和第一份小规模基线已完成。尚未量化或Token压缩，没有训练或云盘上传；下一步应保留这版作为开发回归，并在正式优化对照前补充场景分离的校准/评测集，避免把这个已经观察过的单场景开发集宣称为独立最终测试。']
(out/'closed_loop_baseline_v1_20261030.md').write_text('\n'.join(lines)+'\n')
print('BASELINE_AUDIT_PASSED:',summary['episode_count'],'episodes,',len(all_rows),'decisions')
print('SR:',summary['sr'],'SPL:',summary['spl'],'decision median_s:',summary['decision_latency_s']['median'])
