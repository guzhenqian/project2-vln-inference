"""Clear an obsolete GNU_STACK executable flag in isolated CPU torch libraries."""
import hashlib
import json
import shutil
import struct
import sys
from datetime import datetime
from pathlib import Path

prefix = Path(sys.prefix).resolve()
expected = Path.home() / '.local/share/project2-habitat/mamba/envs/vlnce-017'
assert prefix == expected.resolve()
root = Path(__file__).resolve().parents[1]
backup = root / 'tmp/habitat_setup' / ('torch_execstack_' + datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ'))
libdir = prefix / 'lib/python3.6/site-packages/torch/lib'
records = []
for path in sorted(libdir.glob('*.so*')):
    if not path.is_file() or path.is_symlink():
        continue
    with path.open('rb') as stream:
        header = stream.read(64)
        if header[:6] != b'\x7fELF\x02\x01':
            continue
        phoff = struct.unpack_from('<Q', header, 32)[0]
        phsize, phnum = struct.unpack_from('<HH', header, 54)
        assert phsize >= 56
        offsets = []
        for index in range(phnum):
            offset = phoff + phsize * index
            stream.seek(offset)
            ptype, flags = struct.unpack('<II', stream.read(8))
            if ptype == 0x6474e551 and flags & 1:
                offsets.append((offset + 4, flags))
    if not offsets:
        continue
    backup.mkdir(parents=True, exist_ok=True)
    saved = backup / path.name
    shutil.copy2(str(path), str(saved))
    original_hash = hashlib.sha256(saved.read_bytes()).hexdigest()
    with path.open('r+b') as stream:
        for offset, flags in offsets:
            stream.seek(offset)
            stream.write(struct.pack('<I', flags & ~1))
    expected_bytes = bytearray(saved.read_bytes())
    for offset, flags in offsets:
        struct.pack_into('<I', expected_bytes, offset, flags & ~1)
    after_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert hashlib.sha256(expected_bytes).hexdigest() == after_hash
    records.append({'path': str(path), 'backup': str(saved), 'before_sha256': original_hash,
                    'after_sha256': after_hash, 'flag_offsets': offsets})
    print('Cleared executable stack flag:', path.name)
if records:
    (backup / 'repair.json').write_text(json.dumps(records, indent=2))
    print('Recovery copies:', backup)
else:
    print('No torch libraries require an executable stack; no changes made.')
print('TORCH_EXECSTACK_REPAIR_READY')
