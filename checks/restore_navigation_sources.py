"""Restore pinned small external sources; stdlib only, never import experiment code."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Path must stay inside the reproduction checkout: ' + relative)
    return path


def digest(content):
    return hashlib.sha256(content).hexdigest()


def install(root, record, content):
    if digest(content) != record['sha256']:
        raise ValueError('SHA256 mismatch: ' + record['path'])
    target = inside(root, record['path'])
    if target.exists():
        if digest(target.read_bytes()) != record['sha256']:
            raise ValueError('Refusing to overwrite different existing file: ' + record['path'])
        return 'already_verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(content)
    return 'restored'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true', help='Fetch missing pinned files; default only checks')
    parser.add_argument('--habitat-lab', action='store_true', help='Also restore the verified 16 MB Habitat-Lab source archive')
    args = parser.parse_args()
    records = json.loads((ROOT/'setup/reproduction/external_sources.json').read_text(encoding='utf-8'))['files']
    rows = []
    for record in records:
        if record['group'] == 'habitat' and not args.habitat_lab:
            continue
        path = inside(ROOT, record['path'])
        if path.exists():
            if digest(path.read_bytes()) != record['sha256']:
                raise ValueError('Existing file hash mismatch; nothing overwritten: ' + record['path'])
            status = 'verified'
        elif not args.restore:
            status = 'missing'
        else:
            if 'repository_source' in record:
                content = inside(ROOT, record['repository_source']).read_bytes()
            else:
                request = Request(record['url'], headers={'User-Agent': 'project2-reproduction/1'})
                with urlopen(request, timeout=45) as response:
                    content = response.read(32*1024*1024+1)
                if len(content) > 32*1024*1024:
                    raise ValueError('Unexpected source download size')
            status = install(ROOT, record, content)
        rows.append(dict(path=record['path'], status=status))
    print(json.dumps(rows, indent=2))
    return int(any(row['status'] == 'missing' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
