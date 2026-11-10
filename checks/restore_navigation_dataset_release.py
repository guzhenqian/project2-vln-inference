"""Rebuild exact Habitat subsets from an independently downloaded official archive."""
import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/'datasets/navigation_20261110'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output; never overwrite previous datasets')
    expected = json.loads((RELEASE/'source.json').read_text(encoding='utf-8'))['sha256']
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == expected, 'Official archive hash mismatch'
    with zipfile.ZipFile(args.archive) as archive:
        restored = []
        for split, name in [('val_unseen','val_unseen150.json'),('train','calibration_train8.json')]:
            published = json.loads((RELEASE/name).read_text(encoding='utf-8'))
            member = archive.read(f'R2R_VLNCE_v1-3/{split}/{split}.json.gz')
            assert hashlib.sha256(member).hexdigest() == published['source_member_sha256']
            original = json.loads(gzip.decompress(member))
            lookup = {str(ep['episode_id']):ep for ep in original['episodes']}
            selected = [lookup[str(ep['episode_id'])] for ep in published['episodes']]
            assert selected == published['episodes'], 'Published annotations differ from official source'
            original['episodes'] = selected
            target = output/split/(split+'.json.gz')
            target.parent.mkdir(parents=True, exist_ok=False)
            target.write_bytes(gzip.compress(json.dumps(original, ensure_ascii=False).encode(), mtime=0))
            restored.append(dict(split=split, episodes=len(selected),
                                 sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        (output/'restoration_audit.json').write_text(json.dumps(dict(status='passed',
            original_archive_sha256=expected, subsets=restored), indent=2), encoding='utf-8')
        print(json.dumps(restored))


if __name__ == '__main__':
    main()
