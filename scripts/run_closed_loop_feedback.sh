#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${project_root}/logs/closed_loop_feedback_v2_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p -- "${run_dir}"
exec > >(tee "${run_dir}/run.log") 2>&1
printf 'Six fixed tasks; change only previous-action execution feedback. Output: %s\n' "${run_dir}"
exec env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 \
  bash "${project_root}/scripts/run_habitat_task_python.sh" \
  "${project_root}/checks/run_closed_loop_feedback.py" --output "${run_dir}"
