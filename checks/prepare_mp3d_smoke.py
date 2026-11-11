"""Prepare one real CE scene for wiring checks; not a frozen evaluation split."""
import gzip
import hashlib
import json
import shutil
import zipfile
import zlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
scene_id = 'e9zR4mvMWw7'
dataset_zip = root / 'data/datasets/R2R_VLNCE_v1-3.zip'
scene_zip = root / 'data/scene_datasets/mp3d/downloads/mp3d_habitat.zip'
dataset_dir = root / 'data/datasets/R2R_VLNCE_v1-3_smoke'
scene_dir = root / 'data/scene_datasets/mp3d' / scene_id
with zipfile.ZipFile(str(dataset_zip)) as archive:
    member = 'R2R_VLNCE_v1-3/val_seen/val_seen.json.gz'
    dataset = json.loads(gzip.decompress(archive.read(member)).decode('utf-8'))
episodes = [e for e in dataset['episodes'] if e['scene_id'] == 'mp3d/{0}/{0}.glb'.format(scene_id)]
assert episodes
episodes.sort(key=lambda e: int(e['episode_id']))
chosen = min(episodes, key=lambda e: (e['info']['geodesic_distance'], int(e['episode_id'])))
subset = dict(dataset)
subset['episodes'] = episodes
split_dir = dataset_dir / 'val_seen'
split_dir.mkdir(parents=True, exist_ok=True)
# Keep the original episode dictionaries, vocabulary and directory convention.
compressed = gzip.compress(json.dumps(subset, ensure_ascii=False).encode('utf-8'), mtime=0)
(split_dir / 'val_seen.json.gz').write_bytes(compressed)
(dataset_dir / 'episode_188.json').write_text(json.dumps(chosen, indent=2))
assert int(chosen['episode_id']) == 188, 'Source changed; inspect before rendering'
scene_dir.mkdir(parents=True, exist_ok=True)
file_records = []
with zipfile.ZipFile(str(scene_zip)) as archive:
    for filename in (scene_id + '.glb', scene_id + '.navmesh',
                     scene_id + '.house', scene_id + '_semantic.ply'):
        member = 'mp3d/' + scene_id + '/' + filename
        info = archive.getinfo(member)
        target = scene_dir / filename
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.part')
            with archive.open(member) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
            partial.rename(target)
        digest = hashlib.sha256()
        crc = 0
        with target.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
                crc = zlib.crc32(block, crc)
        assert target.stat().st_size == info.file_size and crc == info.CRC, filename
        file_records.append({'path': str(target.relative_to(root)).replace('\\', '/'),
                             'bytes': info.file_size, 'sha256': digest.hexdigest()})
report = {
    'purpose': 'wiring smoke check only; not quantization calibration or final evaluation',
    'source_dataset': 'R2R_VLNCE_v1-3', 'split': 'val_seen',
    'source_zip_sha256': hashlib.sha256(dataset_zip.read_bytes()).hexdigest(),
    'scene_id': scene_id, 'episode_ids': [e['episode_id'] for e in episodes],
    'preview_episode_id': chosen['episode_id'],
    'preview_selection_rule': 'existing candidate scene; shortest geodesic distance then episode_id',
    'preserved_format': 'original episode records and instruction_vocab in val_seen/val_seen.json.gz',
    'reference_actions_available': False,
    'scene_files': file_records,
}
(dataset_dir / 'subset_manifest.json').write_text(json.dumps(report, indent=2))
print('MP3D_SMOKE_INPUTS_READY')
print('Scene:', scene_id, 'Episodes:', len(episodes), 'Preview:', chosen['episode_id'])
print('Scene files: {:.1f} MiB'.format(sum(f['bytes'] for f in file_records) / 2**20))
print('No rendering, model inference, or navigation evaluation was performed.')
