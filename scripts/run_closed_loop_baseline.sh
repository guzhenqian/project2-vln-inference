#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${project_root}/logs/closed_loop_v1_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "${run_dir}"
exec > >(tee "${run_dir}/run.log") 2>&1
printf 'Fixed six-trajectory Qwen2B BF16 development baseline. Output: %s\n' "${run_dir}"
exec env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 \
  bash "${project_root}/scripts/run_habitat_task_python.sh" \
  "${project_root}/checks/run_closed_loop_baseline.py" --output "${run_dir}"
