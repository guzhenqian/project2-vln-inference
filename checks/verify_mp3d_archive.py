"""Validate the Habitat-only ZIP and record a local SHA256 for backup checks."""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--headers', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    header_text = args.headers.read_text()
    sizes = re.findall(r'^Content-Length:\s*(\d+)\s*$', header_text, re.I | re.M)
    if not sizes:
        raise RuntimeError('No Content-Length in server response; inspect the saved headers')
    expected_size = int(sizes[-1])
    if expected_size < 1024 * 1024:
        raise RuntimeError('Unexpectedly small Habitat archive; inspect the server response')
    partial = Path(str(args.archive) + '.part')
    candidate = args.archive if args.archive.exists() else partial
    actual_size = candidate.stat().st_size if candidate.exists() else 0
    if actual_size > expected_size:
        raise RuntimeError('Local file exceeds remote size; preserve it for inspection')
    if args.preflight:
        free = shutil.disk_usage(str(args.archive.parent)).free
        remaining = expected_size - actual_size
        print('Archive size: {:.2f} GB / {:.2f} GiB'.format(expected_size / 1e9, expected_size / 2**30))
        print('Local bytes:', actual_size, 'Remaining bytes:', remaining)
        print('Available disk: {:.1f} GiB'.format(free / 2**30))
        if free < remaining + 1024**3:
            raise RuntimeError('Insufficient disk space for archive plus 1 GiB reserve')
        (args.headers.parent / 'transfer_required.txt').write_text('yes' if remaining else 'no')
        return
    if not candidate.is_file() or actual_size != expected_size:
        raise RuntimeError('Downloaded size does not match server Content-Length')
    print('Verifying ZIP member CRCs; this can take several minutes.', flush=True)
    with zipfile.ZipFile(str(candidate)) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError('ZIP CRC failure in ' + broken)
        members = archive.infolist()
        scenes = sorted({Path(item.filename).stem for item in members
                         if item.filename.endswith('.glb')})
        if not scenes:
            raise RuntimeError('No GLB scene files in Habitat archive')
        uncompressed = sum(item.file_size for item in members)
    print('Computing local SHA256 for backup verification.', flush=True)
    digest = hashlib.sha256()
    with candidate.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if candidate == partial:
        if args.archive.exists():
            raise RuntimeError('Final archive appeared during validation; refusing overwrite')
        candidate.rename(args.archive)
    checksum = digest.hexdigest()
    Path(str(args.archive) + '.sha256').write_text(checksum + '  ' + args.archive.name + '\n')
    etags = re.findall(r'^ETag:\s*(.+?)\s*$', header_text, re.I | re.M)
    report = {
        'validated_at_utc': datetime.utcnow().isoformat() + 'Z',
        'source': 'https://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip',
        'size_bytes': actual_size, 'sha256': checksum,
        'checksum_scope': 'locally computed; no publisher SHA256 was provided',
        'etag': etags[-1] if etags else None,
        'zip_crc_check': 'passed', 'scene_count': len(scenes), 'scene_ids': scenes,
        'uncompressed_size_bytes': uncompressed, 'extracted': False,
    }
    Path(str(args.archive) + '.manifest.json').write_text(json.dumps(report, indent=2))
    print('ZIP valid. Scenes:', len(scenes), 'Uncompressed size: {:.2f} GiB'.format(uncompressed / 2**30))
    print('SHA256:', checksum)


if __name__ == '__main__':
    main()
