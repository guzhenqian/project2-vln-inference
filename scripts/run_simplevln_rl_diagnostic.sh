#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${project_root}"
suite="${project_root}/logs/simplevln_rl_diagnostic_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "${suite}"
exec > >(tee "${suite}/suite.log") 2>&1
printf 'Same-interface RL comparison and fresh-scene confirmation: %s\n' "${suite}"
for split in compare confirmation; do
  env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 \
    bash scripts/run_habitat_task_python.sh checks/run_simplevln_diagnostic.py \
    --policy memory --manifest "data/baseline/simplevln_4b_rl_${split}/manifest.json" \
    --output "${suite}/${split}"
done
printf 'SIMPLEVLN_RL_DIAGNOSTIC_SUITE_READY %s\n' "${suite}"
