"""Unpack the verified official Habitat-Lab core source without training extras."""
import hashlib
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
archive_path = root / 'tmp/habitat_setup/habitat-lab-v0.1.7-source.zip'
expected_hash = '36afe986d81e905332c440fade63da7c0b5b4246125cab6b56c11a6219d82c8b'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == expected_hash
destination = root / 'tmp/habitat_setup/sources'
source_root = destination / 'habitat-lab-0.1.7'
marker = source_root / '.project2_source_sha256'
if source_root.exists():
    assert marker.is_file() and marker.read_text().strip() == expected_hash, (
        'Existing source is not marked complete; inspect it before retrying')
else:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(archive_path)) as archive:
        base = destination.resolve()
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            assert info.filename.startswith('habitat-lab-0.1.7/')
            assert base in target.parents and '..' not in Path(info.filename).parts
            assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'Unexpected symlink'
        archive.extractall(str(destination))
    marker.write_text(expected_hash + '\n')
assert (source_root / 'habitat/version.py').is_file()
print('Verified Habitat-Lab source:', source_root)
