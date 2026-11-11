"""Pre-register one memory policy and a scene-disjoint six-task check before inference."""
import gzip
import hashlib
import io
import json
import shutil
import zipfile
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = json.loads((root/'data/baseline/closed_loop_v1/manifest.json').read_text())
with zipfile.ZipFile(str(root/'data/datasets/R2R_VLNCE_v1-3.zip')) as archive:
    data = json.loads(gzip.decompress(archive.read('R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz')).decode())
# Deterministic scope restriction for this 100-step diagnostic, never selected by model results.
eligible = [e for e in data['episodes'] if 3.0<e['info']['geodesic_distance']<=12.0]
scenes = sorted(set(e['scene_id'] for e in eligible))[:3]
selected = []
for scene in scenes:
    candidates = sorted([e for e in eligible if e['scene_id']==scene],key=lambda e:int(e['episode_id']))
    seen = set()
    for episode in candidates:
        if episode['trajectory_id'] in seen:continue
        selected.append(episode);seen.add(episode['trajectory_id'])
        if len(seen)==2:break
    assert len(seen)==2
assert len(selected)==6 and base['scene'] not in [s.split('/')[1] for s in scenes]
subset = dict(data, episodes=selected)
directory=root/'data/datasets/R2R_VLNCE_v1-3_diagnostic/val_unseen'
directory.mkdir(parents=True,exist_ok=True)
annotation=directory/'val_unseen.json.gz'
buffer=io.BytesIO()
with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as stream:
    stream.write(json.dumps(subset).encode())
compressed=buffer.getvalue()
if annotation.exists():assert annotation.read_bytes()==compressed
else:annotation.write_bytes(compressed)

files=[]
with zipfile.ZipFile(str(root/'data/scene_datasets/mp3d/downloads/mp3d_habitat.zip')) as archive:
    for scene in scenes:
        sid=scene.split('/')[1]
        for filename in [sid+'.glb',sid+'.navmesh']:
            member='mp3d/'+sid+'/'+filename
            target=root/'data/scene_datasets'/member
            info=archive.getinfo(member)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                partial=target.with_suffix(target.suffix+'.part')
                with archive.open(member) as source,partial.open('wb') as output:shutil.copyfileobj(source,output)
                partial.rename(target)
            content=target.read_bytes()
            assert len(content)==info.file_size and zlib.crc32(content)==info.CRC
            files.append({'path':target.relative_to(root).as_posix(),'sha256':hashlib.sha256(content).hexdigest()})

config={'version':'memory_policy_v3','scope':'development diagnostic; bundled temporal visual memory, progress note and explicit destination reasoning; not a single-variable ablation',
        'model_inputs':'instruction; RGB at start, current-12, current-4 and current; previous model progress note; previous motion/bump feedback; last 20 action names; no task truth',
        'max_tokens':160,'visual_history_offsets':[12,4,0], 'policy_changes':['visual history','model-authored progress memory','destination-aware instructions','JSON action/memory/reason'],
        'pre_registered':'2026-10-30; fixed before first v3 inference; one candidate, no per-episode tuning'}
dev=dict(base);dev.update(config)
holdout=dict(dev)
holdout.update(scope='six scene-disjoint val_unseen diagnostic tasks, three scenes, two distinct trajectories each; no training, no benchmark-wide claim',
    split='val_unseen',scene=None,scenes=scenes,episodes=[str(e['episode_id']) for e in selected],
    trajectory_ids=[str(e['trajectory_id']) for e in selected],
    dataset_path='data/datasets/R2R_VLNCE_v1-3_diagnostic/{split}/{split}.json.gz',
    source_annotation_sha256=hashlib.sha256(compressed).hexdigest(),
    selection='geodesic distance >3m and <=12m; first three scene paths lexicographically; two distinct trajectory IDs per scene by ascending episode_id; fixed before candidate inference',
    scene_files=files)
original=dict(holdout)
for key in config:
    if key in base:original[key]=base[key]
    else:original.pop(key,None)
original['version']='rgb_v1_scene_disjoint_check'
original['scope']=holdout['scope']
for name,manifest in [('memory_policy_v3_dev',dev),('memory_policy_v3_holdout',holdout),('rgb_v1_holdout',original)]:
    path=root/'data/baseline'/name/'manifest.json';path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(manifest,indent=2)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
print('DIAGNOSTIC_PLAN_FROZEN',[(e['episode_id'],e['scene_id']) for e in selected])
