#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
out="logs/navigation_fp16_control_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "${out}"
printf '%s\n' "${out}" > tmp/navigation_fp16_latest.txt
for group in A_diagnostic B_confirmation; do
  env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 bash scripts/run_habitat_task_python.sh \
    checks/run_simplevln_diagnostic.py --policy memory \
    --manifest "data/baseline/navigation_fp16_v1_${group}/manifest.json" \
    --output "${out}/${group}" > "${out}/${group}.log" 2>&1
done
printf 'FP16_CONTROL_COMPLETE %s\n' "${out}"
