"""Restore exact annotation subsets and metadata from an authorized local archive; no rendering."""
import argparse
import gzip
import io
import json
from pathlib import Path
import zipfile

from restore_navigation_sources import ROOT, digest, inside, install


def encode_subset(original, ids):
    lookup = {str(ep['episode_id']): ep for ep in original['episodes']}
    selected = dict(original, episodes=[lookup[str(eid)] for eid in ids])
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as compressed:
        compressed.write(json.dumps(selected).encode())
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True, help='Official R2R_VLNCE_v1-3.zip obtained separately')
    args = parser.parse_args()
    expected = json.loads((ROOT/'datasets/navigation_20261110/source.json').read_text(encoding='utf-8'))['sha256']
    if digest(args.archive.read_bytes()) != expected:
        raise ValueError('Official annotation archive SHA256 mismatch')
    records = json.loads((ROOT/'setup/reproduction/frozen_inputs.json').read_text(encoding='utf-8'))['files']
    pending = [(row, inside(ROOT, row['repository_source']).read_bytes()) for row in records]
    with zipfile.ZipFile(args.archive) as archive:
        for folder in ('navigation_awq_calibration_v1', 'overnight200_fp16_20261031'):
            manifest = json.loads((ROOT/'setup/reproduction/frozen_inputs'/folder/'manifest.json').read_text(encoding='utf-8'))
            split = manifest['split']
            original = json.loads(gzip.decompress(archive.read('R2R_VLNCE_v1-3/{0}/{0}.json.gz'.format(split))))
            content = encode_subset(original, manifest['episodes'])
            record = dict(path=manifest['dataset_path'].format(split=split), sha256=manifest['source_annotation_sha256'])
            if digest(content) != record['sha256']:
                raise ValueError('Reconstructed gzip differs from frozen bytes (check Python/zlib versions): ' + record['path'])
            pending.append((record, content))
    # Validate every destination before writing any file; preserve existing artifacts.
    for record, content in pending:
        if digest(content) != record['sha256']:
            raise ValueError('Packaged metadata hash mismatch: ' + record['path'])
        path = inside(ROOT, record['path'])
        if path.exists() and digest(path.read_bytes()) != record['sha256']:
            raise ValueError('Existing destination differs; refusing overwrite: ' + record['path'])
    rows = [dict(path=record['path'], status=install(ROOT, record, content)) for record, content in pending]
    print(json.dumps(dict(status='passed', files=rows, images_generated=False, model_loaded=False), indent=2))


if __name__ == '__main__':
    main()
