#!/usr/bin/env bash
# Resolve the minimal Habitat-Lab core task dependencies without installing.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_runtime="${HOME}/.local/share/project2-habitat"
habitat_env="${habitat_runtime}/mamba/envs/vlnce-017"
habitat_manager="${habitat_runtime}/bin/micromamba"
run_dir="${project_root}/tmp/habitat_setup/task_plan_$(date -u +%Y%m%dT%H%M%SZ)_$$"
[[ -x "${habitat_manager}" && -x "${habitat_env}/bin/python" ]] || {
  printf 'Missing existing Habitat environment.\n' >&2; exit 1;
}
mkdir -p -- "${run_dir}"
exec > >(tee "${run_dir}/resolve.log") 2>&1
trap 'printf "HABITAT_TASK_ENV_PLAN_FAILED: see %s/resolve.log\n" "${run_dir}" >&2' ERR
"${habitat_env}/bin/python" -m pip --isolated freeze > "${run_dir}/pip_before.txt"
printf 'Planning Habitat-Lab core task dependencies only. No packages will be installed.\n'
printf 'Existing package versions are frozen; no model or training dependencies requested.\n'
printf 'OpenCV is handled separately as a compatible PyPI wheel to avoid the Conda FFmpeg conflict.\n'
plan_command=(
  "${habitat_manager}" install --no-rc
  --root-prefix "${habitat_runtime}/mamba" --prefix "${habitat_env}"
  --override-channels --channel conda-forge --channel aihabitat
  --channel-priority flexible --freeze-installed --dry-run
  'python=3.6.15' 'numpy=1.19.5'
  'habitat-sim=0.1.7=py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7'
  'gym=0.17.3' 'yacs=0.1.8' 'pyyaml=5.4.1' 'pickle5=0.0.11'
)
printf '%q ' "${plan_command[@]}" > "${run_dir}/command.txt"
printf '\n' >> "${run_dir}/command.txt"
"${plan_command[@]}"
printf 'HABITAT_TASK_ENV_PLAN_READY\nLog: %s/resolve.log\n' "${run_dir}"
printf 'Resolution only. Habitat-Lab import, task execution, SR/SPL and model baseline remain untested.\n'
