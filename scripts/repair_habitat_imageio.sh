#!/usr/bin/env bash
# Repair only the incompatible imageio package in the isolated Habitat env.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_env="${HOME}/.local/share/project2-habitat/mamba/envs/vlnce-017"
habitat_python="${habitat_env}/bin/python"
repair_dir="${project_root}/tmp/habitat_setup/imageio_repair_$(date -u +%Y%m%dT%H%M%SZ)_$$"
[[ -x "${habitat_python}" ]] || { printf 'Missing Habitat Python.\n' >&2; exit 1; }
mkdir -p -- "${repair_dir}"
exec > >(tee "${repair_dir}/repair.log") 2>&1
trap 'printf "HABITAT_IMAGEIO_REPAIR_FAILED: see %s/repair.log\n" "${repair_dir}" >&2' ERR

"${habitat_python}" - "${habitat_env}" <<'PY'
import os
import sys
assert sys.version_info[:2] == (3, 6), sys.version
assert os.path.realpath(sys.prefix) == os.path.realpath(sys.argv[1]), sys.prefix
print('Repair target:', sys.prefix)
PY
"${habitat_python}" -m pip --isolated freeze > "${repair_dir}/pip_before.txt"
# Official PyPI wheel hash: https://pypi.org/project/imageio/2.9.0/
printf '%s\n' 'imageio==2.9.0 --hash=sha256:3604d751f03002e8e0e7650aa71d8d9148144a87daf17cb1f3228e80747f2e6b' \
  > "${repair_dir}/requirements.txt"
"${habitat_python}" -m pip --isolated install --disable-pip-version-check \
  --index-url https://pypi.org/simple --only-binary=:all: --no-deps \
  --require-hashes -r "${repair_dir}/requirements.txt"
"${habitat_python}" -m pip --isolated freeze > "${repair_dir}/pip_after.txt"
"${habitat_python}" -c 'import imageio; assert imageio.__version__ == "2.9.0"; print("imageio:", imageio.__version__)'
printf 'Imageio repair complete. Rechecking Habitat imports.\n'
bash "${project_root}/scripts/install_habitat_environment.sh"
