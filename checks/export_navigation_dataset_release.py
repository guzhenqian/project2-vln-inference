"""Publish small task annotations/manifests, not restricted MP3D scene archives."""
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'datasets/navigation_20261110'
TERMS = 'https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        if '/data/scene_datasets/' in value:
            return 'data/scene_datasets/' + value.split('/data/scene_datasets/', 1)[1]
        if '/mnt/' in value or 'C:\\Users\\' in value or '/home/' in value:
            raise ValueError('Unexpected machine path in release')
    return value


def write(name, value):
    path = OUT/name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(portable(value), stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    frozen = ROOT/'data/baseline/navigation_stage2_20261109/manifest.json'
    manifest = read(frozen)
    pool, hundred = manifest['pool150'], manifest['cumulative100']
    assert len(pool) == len(set(pool)) == 150 and len(hundred) == len(set(hundred)) == 100
    assert hundred == pool[:100]
    calibration = read(ROOT/'data/baseline/navigation_awq_calibration_v1/manifest.json')
    samples = read(ROOT/'data/calibration/navigation_awq_v1/samples.json')
    assert len(calibration['episodes']) == 8 and len(samples) == 32
    write('selection.json', dict(dataset='R2R_VLNCE_v1-3', split='val_unseen',
          development_pool150=pool, same100=hundred, initial50=manifest['initial50'],
          reserved50=pool[100:], source_manifest_sha256=sha(frozen),
          selection=manifest['selection'], license_url=TERMS,
          scope='Development subsets, not full official evaluation or independent final holdout'))
    archive = ROOT/'data/datasets/R2R_VLNCE_v1-3.zip'
    assert sha(archive) == read(ROOT/'data/datasets/R2R_VLNCE_v1-3.source.json')['sha256']
    with zipfile.ZipFile(archive) as source:
        for split, ids, name in [('val_unseen', pool, 'val_unseen150.json'),
                                  ('train', calibration['episodes'], 'calibration_train8.json')]:
            payload = source.read(f'R2R_VLNCE_v1-3/{split}/{split}.json.gz')
            data = json.loads(gzip.decompress(payload))
            lookup = {str(ep['episode_id']): ep for ep in data['episodes']}
            selected = [lookup[str(i)] for i in ids]
            assert len({str(ep['trajectory_id']) for ep in selected}) == len(ids)
            write(name, dict(dataset='R2R_VLNCE_v1-3', split=split, license_url=TERMS,
                  source_member_sha256=hashlib.sha256(payload).hexdigest(), episodes=selected))
    eval_scenes = {ep['scene_id'] for ep in read(OUT/'val_unseen150.json')['episodes']}
    train_scenes = {ep['scene_id'] for ep in read(OUT/'calibration_train8.json')['episodes']}
    assert not eval_scenes & train_scenes
    write('calibration_inputs32.json', dict(license_url=TERMS, samples=samples,
          source_samples_sha256=sha(ROOT/'data/calibration/navigation_awq_v1/samples.json'),
          image_payloads='Not mirrored; regenerate from authorized MP3D scenes using capture script'))
    write('calibration_protocol.json', calibration)
    write('prepared_tensor_manifest.json', read(ROOT/'data/calibration/navigation_awq_learning_20261104/processed/manifest.json'))
    write('source.json', read(ROOT/'data/datasets/R2R_VLNCE_v1-3.source.json'))
    for batch in (1, 2, 3):
        write(f'batches/batch_{batch:02d}.json', read(ROOT/f'data/baseline/navigation_stage2_20261109/source_batch_{batch:02d}.json'))
    # Inventory older development inputs without relabeling them as part of same100.
    historical = []
    for path in sorted((ROOT/'data/baseline').glob('*/manifest.json')):
        data = read(path)
        if not isinstance(data, dict):
            continue
        episodes = data.get('episodes')
        historical.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path),
            split=data.get('split'), dataset=data.get('dataset'),
            episode_ids=episodes if isinstance(episodes, list) and all(isinstance(i, (str,int)) for i in episodes) else None,
            image_case_count=len(data.get('cases', [])),
            purpose=data.get('purpose') or data.get('scope'),
            note='Selection provenance only; presence does not imply every planned case was executed'))
    write('historical_selection_inventory.json', historical)
    for file in ('sample_source.json', 'desk_clip_150_source.json', 'navigation_candidates/source_manifest.json'):
        write('historical_tum/'+Path(file).name, read(ROOT/'data/tum_rgbd'/file))
    write('historical_r2r/source.json', read(ROOT/'data/r2r/annotations/source.json'))
    files = {p.relative_to(OUT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(OUT.rglob('*.json'))}
    write('release_audit.json', dict(status='passed', navigation_pool=150, comparison_cases=100,
          calibration_trajectories=8, calibration_inputs=32, evaluation_scenes=len(eval_scenes),
          calibration_scenes=len(train_scenes), scene_overlap=[], files=files,
          original_200_plan='Historical plan had200; current completed comparison pool is first150. Do not call every planned case completed.',
          excluded_payloads=['Matterport3D scene archives/meshes/textures', 'Rendered MP3D images and calibration tensors',
                             'Model weights', 'Private local logs and connection files']))
    print('DATASET_RELEASE_PREPARED', len(files), 'files', sum(v['bytes'] for v in files.values()), 'bytes')


if __name__ == '__main__':
    main()
