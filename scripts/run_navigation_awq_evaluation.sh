#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
out="logs/navigation_awq_eval_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "${out}"
printf '%s\n' "${out}" > tmp/navigation_awq_eval_latest.txt
for group in A_diagnostic B_confirmation; do
  env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 bash scripts/run_habitat_task_python.sh \
    checks/run_simplevln_diagnostic.py --policy memory \
    --manifest "data/baseline/navigation_awq_v1_${group}/manifest.json" \
    --output "${out}/${group}" > "${out}/${group}.log" 2>&1
done
bash scripts/run_habitat_task_python.sh checks/benchmark_navigation_replay.py \
  --model SimpleVLN-Qwen3VL-4B-RL-AWQ --output reports/navigation_replay_awq.json > "${out}/replay.log" 2>&1
bash scripts/run_habitat_task_python.sh checks/audit_navigation_compression.py --awq-suite "$out" > "${out}/audit.log" 2>&1
printf 'AWQ_EVALUATION_COMPLETE %s\n' "${out}"
