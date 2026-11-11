#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
prefix="$(cat tmp/navigation_fp16_latest.txt)/B_confirmation"
out="logs/navigation_fp16_B_resumed_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "${out}"
printf '%s\n' "$out" > tmp/navigation_fp16_B_latest.txt
env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 bash scripts/run_habitat_task_python.sh \
  checks/resume_navigation_run.py --policy memory \
  --manifest data/baseline/navigation_fp16_v1_B_confirmation/manifest.json \
  --prefix "$prefix" --output "$out" > "$out/run.log" 2>&1
printf 'FP16_B_RESUMED_COMPLETE %s\n' "$out"
