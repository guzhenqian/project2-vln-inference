"""Seal first compression experiment; preserve previously frozen quality evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


prior=json.loads((ROOT/'reports/navigation_quality_20261030_freeze.json').read_text())
for name,record in prior['files'].items():
    assert digest(ROOT/name)==record['sha256'],name
print('PREVIOUS_QUALITY_FILES_UNCHANGED',len(prior['files']),flush=True)
report=json.loads((ROOT/'reports/navigation_compression_20261030.json').read_text())
assert not report['decision']['adopted'] and report['runtime_valid']
folders={ROOT/v['path'] for group in ['fp16','awq'] for v in report[group].values()}
folders.add(ROOT/json.loads((ROOT/report['fp16']['B']['path']/'provenance.json').read_text())['prefix_run'])
folders.add(ROOT/'data/calibration/navigation_awq_v1')
folders.add(ROOT/'data/baseline/navigation_awq_calibration_v1')
folders.update((ROOT/'data/baseline').glob('navigation_*_v1_*'))
files=set()
for folder in folders:
    files.update(p for p in folder.rglob('*') if p.is_file() and 'processed' not in p.parts and '__pycache__' not in p.parts)
for directory,patterns in {
    'checks':['*navigation*.py'], 'scripts':['*navigation*.sh','*simplevln*fp16*.sh','*simplevln*awq*.sh'],
    'setup':['navigation-compression*.json','navigation-quant*.txt'],
    'reports':['navigation_compression_20261030.*','navigation_replay_*.json','navigation_awq_*.json'],
    'logs':['navigation_awq_quant_*.log'],
}.items():
    for pattern in patterns:files.update((ROOT/directory).glob(pattern))
for name in ['config.json','conversion_audit.json']:
    files.add(ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_AWQ'/name)
files.add(ROOT/'models/SimpleVLN_Qwen3VL_4B_RL_AWQ_CT/project2_quantization.json')
records={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
output=dict(date='2026-10-31',status='completed first AWQ; not adopted due observed quality regression',
    previous_quality_files_verified_unchanged=len(prior['files']),files=records,
    weight_verification='Full converted weight SHA computed once at conversion; original source SHA retained from download; no redundant large-weight read',
    awq_weight_sha256=report['weights']['awq_sha256'])
(ROOT/'reports/navigation_compression_20261031_freeze.json').write_text(json.dumps(output,indent=2))
print('NAVIGATION_COMPRESSION_FROZEN',len(records),flush=True)
