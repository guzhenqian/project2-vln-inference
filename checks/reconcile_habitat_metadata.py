"""Quarantine a stale ordinary-build record only after verifying headless binary."""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    expected_prefix = Path.home() / '.local/share/project2-habitat/mamba/envs/vlnce-017'
    prefix = Path(sys.prefix).resolve()
    assert prefix == expected_prefix.resolve(), str(prefix)
    commit = '856d4b08c1a2632626bf0d205bf46471a99502b7'
    headless_build = 'py3.6_headless_linux_' + commit
    ordinary_build = 'py3.6_linux_' + commit
    metadata = prefix / 'conda-meta'
    new_record = metadata / ('habitat-sim-0.1.7-' + headless_build + '.json')
    old_record = metadata / ('habitat-sim-0.1.7-' + ordinary_build + '.json')
    new_data = json.loads(new_record.read_text())
    assert new_data['name'] == 'habitat-sim' and new_data['build'] == headless_build
    binding_files = [f for f in new_data['files']
                     if 'habitat_sim_bindings' in f and f.endswith('.so')]
    assert len(binding_files) == 1, binding_files
    cached = Path(new_data['link']['source']) / binding_files[0]
    installed = prefix / binding_files[0]
    assert installed.is_file() and cached.is_file()
    installed_hash = digest(installed)
    assert installed_hash == digest(cached), 'Headless cache/binary mismatch; no changes made'
    print('Verified headless binary SHA256:', installed_hash)
    if not old_record.exists():
        print('No stale ordinary-build record. Nothing to move.')
        return
    old_data = json.loads(old_record.read_text())
    assert old_data['name'] == 'habitat-sim' and old_data['build'] == ordinary_build
    print('Stale record:', old_record)
    if not args.apply:
        print('Read-only check passed. Use --apply to quarantine the stale record.')
        return
    project = Path(__file__).resolve().parents[1]
    backup = project / 'tmp/habitat_setup' / (
        'metadata_reconcile_' + datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True, exist_ok=False)
    # Preserve the original JSON for recovery; do not uninstall shared files.
    shutil.move(str(old_record), str(backup / old_record.name))
    report = {'binary_sha256': installed_hash, 'active_record': str(new_record),
              'quarantined_record': str(backup / old_record.name)}
    (backup / 'result.json').write_text(json.dumps(report, indent=2))
    print('Stale record backed up:', backup)
    print('HABITAT_METADATA_RECONCILED')


if __name__ == '__main__':
    main()
