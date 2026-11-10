"""Add post-run source linkage without changing the original numeric gate report."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'checks/navigation_precision_kernels.py'
original=ROOT/'reports/navigation_precision_kernel_gate_20261110.json'
target=ROOT/'reports/navigation_precision_kernel_gate_linked_20261110.json'
assert source.stat().st_mtime_ns <= original.stat().st_mtime_ns, 'Kernel changed after the gate; rerun required'
data=json.loads(original.read_text(encoding='utf-8'))
data.update(kernel_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            original_numeric_report_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            linkage_note='Source hash added after numerical run; kernel file unchanged since before original report. Original report retained.')
with target.open('x',encoding='utf-8') as stream:
    json.dump(data,stream,indent=2)
print('KERNEL_GATE_LINKED',data['kernel_source_sha256'])
