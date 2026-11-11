#!/usr/bin/env bash
# Install only the reviewed task-core dependencies; no simulator/model runs.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_runtime="${HOME}/.local/share/project2-habitat"
habitat_env="${habitat_runtime}/mamba/envs/vlnce-017"
habitat_python="${habitat_env}/bin/python"
habitat_manager="${habitat_runtime}/bin/micromamba"
run_dir="${project_root}/tmp/habitat_setup/task_install_$(date -u +%Y%m%dT%H%M%SZ)_$$"
[[ -x "${habitat_python}" && -x "${habitat_manager}" ]] || {
  printf 'Missing isolated Habitat environment.\n' >&2; exit 1;
}
mkdir -p -- "${run_dir}"
exec > >(tee "${run_dir}/install.log") 2>&1
trap 'printf "HABITAT_TASK_INSTALL_FAILED: see %s/install.log\n" "${run_dir}" >&2' ERR
"${habitat_python}" "${project_root}/checks/check_habitat_task_imports.py" \
  --save-core "${run_dir}/core_before.json"
"${habitat_python}" -m pip --isolated freeze > "${run_dir}/pip_before.txt"
printf 'Adding reviewed task dependencies; existing Conda packages are frozen.\n'
"${habitat_manager}" install --no-rc --root-prefix "${habitat_runtime}/mamba" \
  --prefix "${habitat_env}" --override-channels --channel conda-forge --channel aihabitat \
  --channel-priority flexible --freeze-installed --yes \
  'python=3.6.15' 'numpy=1.19.5' \
  'habitat-sim=0.1.7=py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7' \
  'gym=0.17.3' 'yacs=0.1.8' 'pyyaml=5.4.1' 'pickle5=0.0.11'
# Only this official Linux cp36-abi3 wheel is accepted. --no-deps protects NumPy.
printf '%s\n' 'opencv-python==4.5.5.64 --hash=sha256:b293ced62f4360d9f11cf72ae7e9df95320ff7bf5b834d87546f844e838c0c35' \
  > "${run_dir}/opencv-requirements.txt"
"${habitat_python}" -m pip --isolated install --disable-pip-version-check \
  --index-url https://pypi.org/simple --only-binary=:all: --no-deps --require-hashes \
  -r "${run_dir}/opencv-requirements.txt"
"${habitat_python}" "${project_root}/checks/prepare_habitat_lab_source.py"
"${habitat_python}" -m pip --isolated freeze > "${run_dir}/pip_after.txt"
bash "${project_root}/scripts/repair_habitat_task_torch.sh" "${run_dir}/core_before.json"
printf 'Log: %s/install.log\nNo simulator, model, or navigation experiment was executed.\n' "${run_dir}"
