#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${project_root}"
stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
suite="${project_root}/logs/navigation_diagnostic_${stamp}"
mkdir -p -- "${suite}"
exec > >(tee "${suite}/suite.log") 2>&1
printf 'Fixed one-candidate suite: %s\n' "${suite}"
for spec in memory_dev rgb_holdout memory_holdout; do
  case "${spec}" in
    memory_dev) policy=memory; manifest=memory_policy_v3_dev ;;
    rgb_holdout) policy=rgb; manifest=rgb_v1_holdout ;;
    memory_holdout) policy=memory; manifest=memory_policy_v3_holdout ;;
  esac
  printf 'START %s\n' "${spec}"
  env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 \
    bash scripts/run_habitat_task_python.sh checks/run_navigation_diagnostic.py \
    --policy "${policy}" --manifest "data/baseline/${manifest}/manifest.json" \
    --output "${suite}/${spec}"
done
printf 'NAVIGATION_DIAGNOSTIC_SUITE_READY %s\n' "${suite}"
