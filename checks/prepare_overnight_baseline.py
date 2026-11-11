"""Freeze four outcome-independent, stratified batches of 50 unseen routes."""
import collections
import gzip
import hashlib
import io
import json
import random
import shutil
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'data/baseline/overnight200_fp16_20261031'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, 'Refusing to overwrite frozen file: ' + str(path)
    else:
        path.write_bytes(content)


def allocate(sizes, total, minimum=0):
    allocation = {k: min(minimum, n) for k, n in sizes.items()}
    remaining = total - sum(allocation.values())
    assert remaining >= 0 and total <= sum(sizes.values())
    weight = sum(sizes.values())
    # Allocate each remaining seat by largest deficit from proportional target.
    for _ in range(remaining):
        k = max((k for k in sizes if allocation[k] < sizes[k]),
                key=lambda k: (total * sizes[k] / weight - allocation[k], k))
        allocation[k] += 1
    return allocation


def main():
    rng = random.Random(20261031)
    with zipfile.ZipFile(ROOT / 'data/datasets/R2R_VLNCE_v1-3.zip') as z:
        packed = z.read('R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz')
    data = json.loads(gzip.decompress(packed))
    old = set()
    for name in ('simplevln_4b_rl_compare', 'simplevln_4b_rl_confirmation'):
        old.update(json.loads((ROOT / 'data/baseline' / name / 'manifest.json').read_text())['trajectory_ids'])
    routes = collections.defaultdict(list)
    for e in data['episodes']:
        if str(e['trajectory_id']) not in old:
            routes[(e['scene_id'], str(e['trajectory_id']))].append(e)
    scenes = sorted({k[0] for k in routes})
    bins = {}
    for scene in scenes:
        keys = sorted((k for k in routes if k[0] == scene),
                      key=lambda k: (routes[k][0]['info']['geodesic_distance'], k[1]))
        for i, key in enumerate(keys):
            bins[key] = min(2, i * 3 // len(keys))
    remaining = set(routes)
    selected, batches, details = [], [], []
    for batch in range(4):
        quota = allocate({s: sum(k[0] == s for k in remaining) for s in scenes}, 50, 1)
        chosen = []
        for scene in scenes:
            candidates = {b: sorted(k for k in remaining if k[0] == scene and bins[k] == b) for b in range(3)}
            counts = allocate({b: len(v) for b, v in candidates.items()}, quota[scene],
                              1 if quota[scene] >= 3 and all(candidates.values()) else 0)
            for b in range(3):
                for key in rng.sample(candidates[b], counts[b]):
                    variants = sorted(routes[key], key=lambda e: (len(e['instruction']['instruction_text'].split()), str(e['episode_id'])))
                    # Cycle among shortest/middle/longest instructions within each route.
                    rank = (len(selected) + len(chosen)) % 3
                    e = variants[round(rank * (len(variants) - 1) / 2)]
                    chosen.append(e)
                    remaining.remove(key)
                    details.append(dict(batch=batch + 1, episode_id=str(e['episode_id']),
                        trajectory_id=str(e['trajectory_id']), scene=e['scene_id'],
                        geodesic_distance_m=e['info']['geodesic_distance'], length_bin=b,
                        instruction_words=len(e['instruction']['instruction_text'].split()), instruction_rank=rank))
        rng.shuffle(chosen)
        assert len(chosen) == 50 and len({e['scene_id'] for e in chosen}) == 11
        batches.append(chosen)
        selected.extend(chosen)
    assert len({str(e['trajectory_id']) for e in selected}) == 200
    assert not old.intersection(str(e['trajectory_id']) for e in selected)
    subset = dict(data, episodes=selected)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as f:
        f.write(json.dumps(subset).encode())
    dataset_path = 'data/datasets/R2R_VLNCE_v1-3_overnight200/{split}/{split}.json.gz'
    freeze(ROOT / dataset_path.format(split='val_unseen'), buffer.getvalue())
    scene_files = []
    with zipfile.ZipFile(ROOT / 'data/scene_datasets/mp3d/downloads/mp3d_habitat.zip') as archive:
        for scene in scenes:
            sid = scene.split('/')[1]
            for suffix in ('.glb', '.navmesh'):
                member = 'mp3d/{0}/{0}{1}'.format(sid, suffix)
                target = ROOT / 'data/scene_datasets' / member
                info = archive.getinfo(member)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    partial = target.with_suffix(target.suffix + '.part')
                    with archive.open(member) as source, partial.open('wb') as dest:
                        shutil.copyfileobj(source, dest)
                    partial.replace(target)
                assert target.stat().st_size == info.file_size
                crc = 0
                with target.open('rb') as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                assert crc == info.CRC
                scene_files.append(dict(path=target.relative_to(ROOT).as_posix(), sha256=sha(target)))
    base = json.loads((ROOT / 'data/baseline/navigation_fp16_v1_A_diagnostic/manifest.json').read_text())
    base.update(version='overnight200_fp16_20261031',
        scope='200-route unquantized FP16 development baseline; four frozen batches of50; not full official benchmark',
        selection='seed20261031; exclude old12 routes; sequential proportional scene quotas min1/scene/batch; per-scene fixed length tertiles; no distance restriction; cycle instruction length ranks; shuffle within batch',
        sampling_seed=20261031, source_full_annotation_sha256=hashlib.sha256(packed).hexdigest(),
        source_annotation_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
        dataset_path=dataset_path, scenes=scenes, scene_files=scene_files,
        pre_registered='2026-10-31 before overnight candidate inference; user authorized200 or fixed50-sized prefix',
        horizon_note='400 primitive actions includingSTOP; no collision early termination or policy changes',
        differences_from_author_benchmark='Habitat0.1.7/LMDeploy, FP16; route-stratified subset with one instruction/route; not full1839')
    for i, es in enumerate(batches):
        manifest = dict(base, batch=i+1, episodes=[str(e['episode_id']) for e in es],
                        trajectory_ids=[str(e['trajectory_id']) for e in es])
        freeze(PLAN / ('batch_%02d.json' % (i+1)), json.dumps(manifest, indent=2).encode())
    plan = dict(base, episodes=[str(e['episode_id']) for e in selected],
                trajectory_ids=[str(e['trajectory_id']) for e in selected], sample_details=details,
                excluded_trajectories=sorted(old), deadline_local='2026-11-01T08:00:00+08:00',
                stopping_rule='Finish each started50 batch; launch next only if remaining deadline budget >=1.3*max(observed batch walltime). All outcomes kept. On infrastructure error stop and report partial evidence; no failure deletion.',
                future_use='Development baseline for quantization teaching; original60/180 plan superseded, confirmation set not yet sampled')
    freeze(PLAN / 'manifest.json', json.dumps(plan, indent=2).encode())
    print('OVERNIGHT_PLAN_READY', len(selected), 'routes; scenes', len(scenes))
    for i, es in enumerate(batches):
        print('batch', i+1, 'scenes', len({e['scene_id'] for e in es}), 'distance_range',
              min(e['info']['geodesic_distance'] for e in es), max(e['info']['geodesic_distance'] for e in es))


if __name__ == '__main__':
    main()
