#!/usr/bin/env bash
# Add missing CPU torch dependencies only to the isolated simulator environment.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_python="${HOME}/.local/share/project2-habitat/mamba/envs/vlnce-017/bin/python"
run_dir="${project_root}/tmp/habitat_setup/task_torch_$(date -u +%Y%m%dT%H%M%SZ)_$$"
[[ -x "${habitat_python}" ]] || { printf 'Missing Habitat Python.\n' >&2; exit 1; }
mkdir -p -- "${run_dir}"
exec > >(tee "${run_dir}/repair.log") 2>&1
trap 'printf "HABITAT_TASK_TORCH_FAILED: see %s/repair.log\n" "${run_dir}" >&2' ERR
core_before="${1:-${run_dir}/core_before.json}"
if [[ $# -eq 0 ]]; then
  "${habitat_python}" "${project_root}/checks/check_habitat_task_imports.py" --save-core "${core_before}"
fi
[[ -f "${core_before}" ]] || { printf 'Missing core package snapshot.\n' >&2; exit 1; }
"${habitat_python}" -c 'import typing_extensions; print("Existing typing_extensions import passed")'
"${habitat_python}" -m pip --isolated freeze > "${run_dir}/pip_before.txt"
cp -- "${project_root}/setup/habitat-task-torch-requirements.txt" "${run_dir}/requirements.txt"
printf 'Installing CPU-only torch1.10.2 and Python3.6 dataclasses (~199 MB).\n'
"${habitat_python}" -m pip --isolated install --disable-pip-version-check \
  --index-url https://pypi.org/simple --only-binary=:all: --no-deps --require-hashes \
  -r "${run_dir}/requirements.txt"
"${habitat_python}" -m pip --isolated freeze > "${run_dir}/pip_after.txt"
"${habitat_python}" "${project_root}/checks/repair_torch_execstack.py"
bash "${project_root}/scripts/run_habitat_task_python.sh" \
  "${project_root}/checks/check_habitat_task_imports.py" --core-before "${core_before}"
printf 'Log: %s/repair.log\n' "${run_dir}"
