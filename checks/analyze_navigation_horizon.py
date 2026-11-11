"""Audit the exact-prefix horizon sensitivity and preserve distinct 100/400-step scores."""
import argparse
import json
from pathlib import Path

import yaml

from analyze_navigation_sr import ROOT,audit

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--prefix',type=Path,required=True)
args=parser.parse_args()
long_run=audit(args.run.resolve());short_run=audit(args.prefix.resolve())
assert (args.run/'system_prompt.txt').read_bytes()==(args.prefix/'system_prompt.txt').read_bytes()
old_config=yaml.safe_load((args.prefix/'habitat_config.yaml').read_text())
new_config=yaml.safe_load((args.run/'habitat_config.yaml').read_text())
assert old_config['ENVIRONMENT']['MAX_EPISODE_STEPS']==100
assert new_config['ENVIRONMENT']['MAX_EPISODE_STEPS']==400
new_config['ENVIRONMENT']['MAX_EPISODE_STEPS']=100
assert old_config==new_config
verified=0
for old,new in zip(short_run['episodes'],long_run['episodes']):
    assert old['episode_id']==new['episode_id']
    name='episode_'+old['episode_id']
    prefix=[json.loads(line) for line in (args.prefix/name/'steps.jsonl').read_text().splitlines()]
    extended=[json.loads(line) for line in (args.run/name/'steps.jsonl').read_text().splitlines()]
    for prior,row in zip(prefix,extended):
        restored=dict(row);assert restored.pop('prefix_reused') is True
        assert restored==prior,'Prefix action/state/model output changed'
        verified+=1
    if old['stop_called']:assert len(prefix)==len(extended),'Cannot reopen a terminated episode'
result={'scope':'exact-prefix horizon sensitivity, not an acceleration benchmark',
        'validation':'prefix pixels/poses replayed and checked during simulation; logged prefix JSON fields unchanged; same system/task apart from cap; independent SR/SPL and image history audit passed',
        'verified_prefix_steps':verified,'h100':short_run,'h400':long_run}
out=ROOT/'reports'
(out/'navigation_horizon_20261030.json').write_text(json.dumps(result,indent=2))
lines=['# 100与400步上限敏感性：2026-10-30','',
    '固定同一SFT4B导航权重、作者策略与六条独立场景任务，只放宽任务上限；没有更改原100步成绩。全部六条都保留在分母，已主动STOP的任务不重开。',
    '', '| 协议 | SR | SPL | 到过3m范围比例 | 总动作数 | 接口错误 |', '| --- | --- | --- | --- | --- | --- |']
for label,r in [('100步',short_run),('400步',long_run)]:
    lines.append('| {} | {:.1%} | {:.4f} | {:.1%} | {} | {} |'.format(label,r['sr'],r['spl'],r['oracle_reached_rate'],r['steps'],r['errors']))
lines+=['','| Episode | 100步 SR/步数/终距m | 400步 SR/步数/终距m |','| --- | --- | --- |']
for old,new in zip(short_run['episodes'],long_run['episodes']):
    lines.append('| {} | {} / {} / {:.3f} | {} / {} / {:.3f} |'.format(old['episode_id'],int(old['metrics']['success']),old['steps'],old['metrics']['distance_to_goal'],int(new['metrics']['success']),new['steps'],new['metrics']['distance_to_goal']))
lines+=['','## 解释','',
    '放宽上限后SR变化为{:+.1f}个百分点。在该固定队列中，延长步数{}解决低SR，不能把失败都归因于100步过短。仍需区分路线/避障错误与到达后未主动STOP。'.format((long_run['sr']-short_run['sr'])*100,'没有' if long_run['sr']==short_run['sr'] else '只能部分'),
    '源运行的全部{}个动作逐帧、位姿与距离重放验证；模型历史图像和未执行动作队列恢复后继续。策略没有接收任务上限或目标距离，因此不是用真值改变动作。新输出中的前缀JSON与原记录逐字段一致。'.format(verified),
    '性能数字不是一次连续重跑的严格速度基准：前缀保留原计时，续跑单独计时。本试验只用于解释上限对同一轨迹策略的质量影响。',
    '', '## 证据','', '- 100步原始运行：`{}`。'.format(args.prefix.as_posix()),'- 400步续跑：`{}`。'.format(args.run.as_posix()),
    '- 400步固定清单：`data/baseline/simplevln_4b_sft_holdout_h400/manifest.json`。',
    '- 数据、系统提示词、所有逐步原始模型回答及图片均保留。']
(out/'navigation_horizon_20261030.md').write_text('\n'.join(lines)+'\n')
print('HORIZON_SENSITIVITY_AUDIT_READY',json.dumps({'h100_sr':short_run['sr'],'h400_sr':long_run['sr'],'verified_prefix_steps':verified}))
