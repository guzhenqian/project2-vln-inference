"""Freeze scene-disjoint calibration and unchanged A/B evaluation before AWQ."""
import gzip
import hashlib
import io
import json
import shutil
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def frozen_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, path
    else:
        path.write_text(text, encoding='utf-8')

def main():
    groups = {key: json.loads((ROOT / ('data/baseline/simplevln_4b_rl_' + folder + '/manifest.json')).read_text())
              for key, folder in [('A_diagnostic', 'compare'), ('B_confirmation', 'confirmation')]}
    eval_scenes = set(s for m in groups.values() for s in m['scenes'])
    with zipfile.ZipFile(str(ROOT / 'data/datasets/R2R_VLNCE_v1-3.zip')) as archive:
        train = json.loads(gzip.decompress(archive.read('R2R_VLNCE_v1-3/train/train.json.gz')).decode())
    eligible = [e for e in train['episodes'] if 3 < e['info']['geodesic_distance'] <= 12
                and e['scene_id'] not in eval_scenes]
    scenes = sorted(set(e['scene_id'] for e in eligible))[:4]
    episodes = []
    for scene in scenes:
        selected = set()
        for ep in sorted([e for e in eligible if e['scene_id'] == scene], key=lambda e:int(e['episode_id'])):
            if ep['trajectory_id'] in selected:
                continue
            selected.add(ep['trajectory_id']); episodes.append(ep)
            if len(selected) == 2:
                break
        assert len(selected) == 2
    subset = dict(train, episodes=episodes)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as stream:
        stream.write(json.dumps(subset).encode())
    target = ROOT / 'data/datasets/R2R_VLNCE_v1-3_quant_calibration/train/train.json.gz'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert target.read_bytes() == buffer.getvalue()
    else:
        target.write_bytes(buffer.getvalue())
    files = []
    with zipfile.ZipFile(str(ROOT / 'data/scene_datasets/mp3d/downloads/mp3d_habitat.zip')) as archive:
        for scene in scenes:
            sid = scene.split('/')[1]
            for suffix in ['glb', 'navmesh']:
                member = 'mp3d/{0}/{0}.{1}'.format(sid, suffix)
                info = archive.getinfo(member)
                dest = ROOT / 'data/scene_datasets' / member
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    part = dest.with_suffix(dest.suffix + '.part')
                    with archive.open(member) as source, part.open('wb') as output:
                        shutil.copyfileobj(source, output)
                    part.rename(dest)
                content = dest.read_bytes()
                assert len(content) == info.file_size and zlib.crc32(content) == info.CRC
                files.append(dict(path=dest.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(content).hexdigest()))
    calibration = dict(version='navigation_awq_calibration_v1', split='train', scenes=scenes,
        episodes=[str(e['episode_id']) for e in episodes],
        trajectory_ids=[str(e['trajectory_id']) for e in episodes],
        dataset_path='data/datasets/R2R_VLNCE_v1-3_quant_calibration/{split}/{split}.json.gz',
        source_annotation_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(), scene_files=files,
        selection='Before quantization: first4 lexicographic train scenes, first2 distinct trajectory IDs by ascending episode ID; 3<geodesic<=12m; exclude all A/B scenes',
        capture='Reference-path follower generates observations only; 4 evenly spaced decision observations per trajectory, including start and last pre-STOP. Author input formatting and up to8 historical images from last200; no actions/goals/distances in calibration inputs.',
        sample_count=32, max_episode_steps=400,
        note='Representative teacher-path activation calibration, not gradient training and not evaluation. Follower has ground truth only for generating training-split views.')
    frozen_json(ROOT / 'data/baseline/navigation_awq_calibration_v1/manifest.json', calibration)
    for key, manifest in groups.items():
        new = dict(manifest, model='SimpleVLN-Qwen3VL-4B-RL-AWQ', dtype='W4A16',
                   version='navigation_awq_v1_' + key,
                   scope='Paired AWQ comparison on frozen '+key+'; existing diagnostic/confirmation subset, not new untouched test set',
                   backend='LMDeploy0.11.1 PyTorch; language decoder W4A16 group128 asymmetric; vision/embed/head BF16; context4096; cache0.35')
        frozen_json(ROOT / ('data/baseline/navigation_awq_v1_' + key + '/manifest.json'), new)
    profile = dict(version='navigation_compression_v1', baseline='setup/navigation-baseline-profile.json',
        calibration='data/baseline/navigation_awq_calibration_v1/manifest.json',
        evaluation={key:'data/baseline/navigation_awq_v1_'+key+'/manifest.json' for key in groups},
        calibration_evaluation_scene_overlap=sorted(set(scenes)&eval_scenes),
        first_experiment=dict(algorithm='AWQ', bits=4, group_size=128, symmetric=False,
            targets='language decoder Linear only; vision/deepstack projectors/embedding/lm_head/norms stay BF16',
            calibration_samples=32, seed=7, token_compression=False, retraining=False),
        interpretation='All6 tasks per group retained; model requests only for latency (exclude queued actions). Whole-card memory sampled every200ms. Compare fixed-input replay separately from closed-loop latency because trajectories differ. No claim of full benchmark quality.',
        stopping_rule='One prespecified32-sample AWQ candidate; report all successes/failures, no tuning on B scores.')
    assert not profile['calibration_evaluation_scene_overlap']
    frozen_json(ROOT / 'setup/navigation-compression-profile.json', profile)
    print('COMPRESSION_PROTOCOL_FROZEN', scenes, calibration['episodes'])

if __name__ == '__main__':
    main()
