"""Freeze the RL-only weight comparison and a further three untouched confirmation scenes."""
import gzip
import hashlib
import io
import json
import shutil
import zipfile
import zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
base=json.loads((ROOT/'data/baseline/simplevln_4b_sft_holdout_h400/manifest.json').read_text())
comparison=dict(base,version='simplevln_4b_rl_comparison',model='SimpleVLN-Qwen3VL-4B-RL',
    model_revision='4078dab77f3c93a0711b57549ac54a59c1c1e1ea',
    scope='same six400-step tasks as SFT; same policy/input/decode/runtime precision; replace navigation checkpoint with published RL weights',
    pre_registered='2026-10-30 before first RL inference; compare all six fixed tasks, no prompt tuning')
comparison['differences_from_author_benchmark']='Existing validated Habitat0.1.7 and LMDeploy instead of author0.3.3/vLLM;400-step cap matches author; small fixed subset, not full1839 benchmark'
comparison['source_precision']='F32 checkpoint; runtime cast to BF16 to match SFT deployment; no4bit quantization'
with zipfile.ZipFile(str(ROOT/'data/datasets/R2R_VLNCE_v1-3.zip')) as archive:
    data=json.loads(gzip.decompress(archive.read('R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz')).decode())
eligible=[e for e in data['episodes'] if 3<e['info']['geodesic_distance']<=12]
scenes=sorted(set(e['scene_id'] for e in eligible))[3:6]
assert not set(scenes).intersection(base['scenes'])
episodes=[]
for scene in scenes:
    chosen=set()
    for episode in sorted([e for e in eligible if e['scene_id']==scene],key=lambda e:int(e['episode_id'])):
        if episode['trajectory_id'] in chosen:continue
        episodes.append(episode);chosen.add(episode['trajectory_id'])
        if len(chosen)==2:break
    assert len(chosen)==2
subset=dict(data,episodes=episodes)
buffer=io.BytesIO()
with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0) as stream:stream.write(json.dumps(subset).encode())
compressed=buffer.getvalue()
path=ROOT/'data/datasets/R2R_VLNCE_v1-3_confirmation/val_unseen/val_unseen.json.gz'
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():assert path.read_bytes()==compressed
else:path.write_bytes(compressed)
files=[]
with zipfile.ZipFile(str(ROOT/'data/scene_datasets/mp3d/downloads/mp3d_habitat.zip')) as archive:
    for scene in scenes:
        sid=scene.split('/')[1]
        for name in [sid+'.glb',sid+'.navmesh']:
            member='mp3d/'+sid+'/'+name;info=archive.getinfo(member)
            target=ROOT/'data/scene_datasets'/member;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                part=target.with_suffix(target.suffix+'.part')
                with archive.open(member) as src,part.open('wb') as dest:shutil.copyfileobj(src,dest)
                part.rename(target)
            content=target.read_bytes()
            assert len(content)==info.file_size and zlib.crc32(content)==info.CRC
            files.append({'path':target.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(content).hexdigest()})
confirmation=dict(comparison,version='simplevln_4b_rl_confirmation',scenes=scenes,
    episodes=[str(e['episode_id']) for e in episodes],trajectory_ids=[str(e['trajectory_id']) for e in episodes],
    dataset_path='data/datasets/R2R_VLNCE_v1-3_confirmation/{split}/{split}.json.gz',
    source_annotation_sha256=hashlib.sha256(compressed).hexdigest(),scene_files=files,
    selection='same distance scope3<d<=12; next three lexicographic scene paths not used in prior diagnostic; first two distinct trajectories by episode_id; before any RL inference',
    scope='six fresh tasks in three additional scene-disjoint val_unseen scenes; not used to develop any of the previous candidate policies')
for name,manifest in [('simplevln_4b_rl_compare',comparison),('simplevln_4b_rl_confirmation',confirmation)]:
    file=ROOT/'data/baseline'/name/'manifest.json';file.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(manifest,indent=2)
    if file.exists():assert file.read_text()==text
    else:file.write_text(text)
print('RL_COMPARISON_AND_CONFIRMATION_FROZEN',[(e['episode_id'],e['scene_id']) for e in episodes])
