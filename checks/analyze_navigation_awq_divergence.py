"""Locate first action divergence on identical observations, without new inference."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'reports/navigation_compression_20261030.json').read_text())
output=[]
for group in ['A','B']:
    base=ROOT/report['fp16'][group]['path']; quant=ROOT/report['awq'][group]['path']
    for before,after in zip(report['fp16'][group]['episodes'],report['awq'][group]['episodes']):
        if not (before['metrics']['success'] and not after['metrics']['success']):continue
        eid=before['episode_id']
        old=[json.loads(x) for x in (base/('episode_'+eid)/'steps.jsonl').read_text().splitlines()]
        new=[json.loads(x) for x in (quant/('episode_'+eid)/'steps.jsonl').read_text().splitlines()]
        first=None
        for a,b in zip(old,new):
            assert a['before']==b['before']
            if a['action']!=b['action']:
                ca=next(r for r in reversed(old[:a['step']+1]) if r['model_called'])
                cb=next(r for r in reversed(new[:b['step']+1]) if r['model_called'])
                first=dict(step=a['step'],fp16_action=a['action'],awq_action=b['action'],
                    fp16_call_step=ca['step'],awq_call_step=cb['step'],
                    call_images_identical=ca['request_image_sha256']==cb['request_image_sha256'],
                    fp16_raw=ca['raw'],awq_raw=cb['raw'])
                break
        assert first is not None
        output.append(dict(group=group,episode_id=eid,first_action_divergence=first,
            fp16_steps=before['steps'],awq_steps=after['steps'],awq_final_distance_m=after['metrics']['distance_to_goal'],
            awq_minimum_distance_m=after['minimum_distance_m'],awq_stop_called=after['stop_called'],
            awq_failure_category=after['failure_category'],awq_collisions=after['metrics'].get('collisions')))
(ROOT/'reports/navigation_awq_divergence.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
