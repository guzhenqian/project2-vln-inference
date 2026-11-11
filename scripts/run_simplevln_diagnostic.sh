#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${project_root}"
suite="${project_root}/logs/simplevln_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "${suite}"
exec > >(tee "${suite}/suite.log") 2>&1
printf 'Navigation checkpoint DEV + scene-disjoint check: %s\n' "${suite}"
for split in dev holdout; do
  env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 \
    bash scripts/run_habitat_task_python.sh checks/run_simplevln_diagnostic.py \
    --policy memory --manifest "data/baseline/simplevln_4b_sft_${split}/manifest.json" \
    --output "${suite}/${split}"
done
printf 'SIMPLEVLN_DIAGNOSTIC_SUITE_READY %s\n' "${suite}"
