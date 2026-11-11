"""Identical input replay timing; not a navigation success metric."""
import argparse
import base64
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

import imageio
from PIL import Image
import simplevln_policy as policy

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'logs/simplevln_rl_diagnostic_20261030T101819Z_49596/confirmation'

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    import gzip
    dataset=json.loads(gzip.decompress((ROOT/manifest['dataset_path'].format(split=manifest['split'])).read_bytes()))
    instructions={str(e['episode_id']):e['instruction']['instruction_text'] for e in dataset['episodes']}
    payloads=[]
    for eid in manifest['episodes']:
        rows=[json.loads(line) for line in (SOURCE/('episode_'+eid)/'steps.jsonl').read_text().splitlines()]
        calls=[r for r in rows if r['model_called']]
        for index in [round(i*(len(calls)-1)/3) for i in range(4)]:
            row=calls[index];step=row['step']
            agent=policy.AuthorAgent('http://unused',args.model,str(ROOT/'tmp/replay_agent'),25,15,200,1.0,False)
            agent.rgb_list=[Image.open(str(SOURCE/r['frame'])).convert('RGB') for r in rows[max(0,step-199):step]]
            agent.predict_inference=lambda:'stop'
            agent.act({'rgb':imageio.imread(str(SOURCE/row['frame'])),'instruction':{'text':instructions[eid]}},{},eid)
            hashes=[hashlib.sha256(base64.b64decode(x['image_url']['url'].split(',',1)[1])).hexdigest() for x in agent.conversations[1]['content'] if x['type']=='image_url']
            assert hashes==row['request_image_sha256']
            payload=dict(model=args.model,messages=agent.conversations,temperature=.3,top_p=.95,seed=7,max_tokens=512,stream=False)
            payloads.append((eid,step,payload,hashes))
    args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists()
    results=[]
    for i,(eid,step,payload,hashes) in enumerate(payloads[:2]+payloads+payloads):
        started=time.perf_counter()
        req=urllib.request.Request('http://127.0.0.1:23333/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=120) as r:answer=json.load(r)
        elapsed=time.perf_counter()-started
        if i<2:continue
        results.append(dict(episode_id=eid,step=step,repeat=(i-2)//24,request_s=elapsed,
            image_sha256=hashes,payload_sha256=hashlib.sha256(json.dumps(dict(payload,model='MODEL'),sort_keys=True).encode()).hexdigest(),
            raw=answer['choices'][0]['message']['content'],usage=answer.get('usage'),finish_reason=answer['choices'][0]['finish_reason']))
        print('REPLAY',i-1,eid,step,round(elapsed,3),flush=True)
    args.output.write_text(json.dumps(dict(model=args.model,scope='24 fixed original-B image requests x2; seed7; original policy; whole HTTP request, not pure GPU time',
        median_s=statistics.median(r['request_s'] for r in results),requests=results),indent=2))

if __name__=='__main__':main()
