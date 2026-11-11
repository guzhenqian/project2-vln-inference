#!/usr/bin/env bash
# Install the simulator separately from the model API and WebUI environments.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_runtime="${HOME}/.local/share/project2-habitat"
habitat_manager="${habitat_runtime}/bin/micromamba"
habitat_env="${habitat_runtime}/mamba/envs/vlnce-017"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
install_dir="${project_root}/tmp/habitat_setup/install_${run_id}"

[[ -x "${habitat_manager}" ]] || {
  printf 'Missing manager. Run setup_habitat_manager.sh first.\n' >&2
  exit 1
}
mkdir -p -- "${install_dir}"
exec > >(tee "${install_dir}/install.log") 2>&1
trap 'printf "HABITAT_ENV_STEP_FAILED: see %s/install.log\n" "${install_dir}" >&2' ERR

"${habitat_manager}" --version > "${install_dir}/manager_version.txt"
if [[ -e "${habitat_env}" ]]; then
  printf 'Existing target detected. Skipping installation; checking its packages and imports.\n'
  [[ -f "${habitat_env}/conda-meta/history" ]] || {
    printf 'Incomplete target: inspect it before retrying; no files were removed.\n' >&2
    exit 1
  }
else
  install_command=(
    "${habitat_manager}" create --no-rc
    --root-prefix "${habitat_runtime}/mamba"
    --prefix "${habitat_env}"
    --override-channels --channel aihabitat --channel conda-forge
    --channel-priority flexible --yes
    'python=3.6' 'habitat-sim=0.1.7=py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7' headless 'imageio=2.9.0'
  )
  printf '%q ' "${install_command[@]}" > "${install_dir}/command.txt"
  printf '\n' >> "${install_dir}/command.txt"
  "${install_command[@]}"
fi

"${habitat_manager}" list --no-rc --root-prefix "${habitat_runtime}/mamba" \
  --prefix "${habitat_env}" --json > "${install_dir}/packages.json"
"${habitat_manager}" list --no-rc --root-prefix "${habitat_runtime}/mamba" \
  --prefix "${habitat_env}" --explicit > "${install_dir}/packages_explicit.txt"

"${habitat_env}/bin/python" - "${install_dir}/packages.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    package_data = json.load(f)
# micromamba 2.10 wraps the records in a "packages" object.
package_records = (package_data['packages']
                   if isinstance(package_data, dict) else package_data)
if not isinstance(package_records, list):
    raise ValueError('Expected a list of package records')
packages = {p['name']: p for p in package_records}
assert sys.version_info[:2] == (3, 6), sys.version
assert packages['habitat-sim']['version'] == '0.1.7'
assert 'headless' in packages['habitat-sim']['build_string'], (
    'Simulator binary is not headless: ' + packages['habitat-sim']['build_string'])
assert 'headless' in packages['habitat-sim-mutex']['build_string']
print('Python:', sys.version.split()[0])
print('Habitat package:', packages['habitat-sim']['version'])
print('Habitat build:', packages['habitat-sim']['build_string'])
print('Mutex:', packages['habitat-sim-mutex']['build_string'])
import habitat_sim
import numpy
import imageio
print('habitat_sim import:', habitat_sim.__file__)
print('numpy:', numpy.__version__)
print('imageio:', imageio.__version__)
PY

# pip repairs can differ from conda metadata; preserve the runtime package view.
"${habitat_env}/bin/python" -m pip --isolated freeze > "${install_dir}/pip_freeze.txt"

printf 'HABITAT_SIM_IMPORT_READY\nEnvironment: %s\nLog: %s/install.log\n' \
  "${habitat_env}" "${install_dir}"
printf 'Simulator installation and imports passed. Scene rendering and navigation are not yet tested.\n'
