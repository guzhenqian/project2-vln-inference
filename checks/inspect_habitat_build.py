"""Read-only comparison of installed Habitat binary and package-cache records."""
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


prefix = Path(sys.prefix)
print('Prefix:', prefix)
for record in sorted((prefix / 'conda-meta').glob('habitat-sim-*.json')):
    data = json.loads(record.read_text())
    if data['name'] != 'habitat-sim':
        continue
    print('Record:', record.name)
    print('Build:', data['build'])
    source = data.get('link', {}).get('source')
    print('Cache:', source)
    for name in data.get('files', []):
        if 'habitat_sim_bindings' in name and name.endswith('.so'):
            actual = prefix / name
            print('Binary:', name)
            print('Installed SHA256:', sha256(actual) if actual.exists() else 'MISSING')
            cached = Path(source) / name if source else None
            print('Cached SHA256:', sha256(cached) if cached and cached.exists() else 'MISSING')
