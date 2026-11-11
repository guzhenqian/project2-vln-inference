#!/usr/bin/env bash
# Replace only the 0.1.7 simulator binary, keeping existing dependencies.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_runtime="${HOME}/.local/share/project2-habitat"
habitat_env="${habitat_runtime}/mamba/envs/vlnce-017"
habitat_manager="${habitat_runtime}/bin/micromamba"
repair_dir="${project_root}/tmp/habitat_setup/headless_repair_$(date -u +%Y%m%dT%H%M%SZ)_$$"
[[ -f "${habitat_env}/conda-meta/history" ]] || { printf 'Missing installed environment.\n' >&2; exit 1; }
mkdir -p -- "${repair_dir}"
exec > >(tee "${repair_dir}/repair.log") 2>&1
trap 'printf "HABITAT_HEADLESS_REPAIR_FAILED: see %s/repair.log\n" "${repair_dir}" >&2' ERR
cp -- "${habitat_env}/conda-meta/history" "${repair_dir}/history_before.txt"
"${habitat_env}/bin/python" -m pip --isolated freeze > "${repair_dir}/pip_before.txt"
# Exact official aihabitat build and checksum from the 0.1.7 release API.
printf '%s\n' '@EXPLICIT' \
  'https://conda.anaconda.org/aihabitat/linux-64/habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.tar.bz2#d47114f26f252b7c20a85c7b4f238999' \
  > "${repair_dir}/explicit.txt"
"${habitat_manager}" list --no-rc --root-prefix "${habitat_runtime}/mamba" \
  --prefix "${habitat_env}" --json > "${repair_dir}/packages_before.json"
printf 'Replacing only habitat-sim with the official Python 3.6 headless build (~128 MB).\n'
if [[ -f "${habitat_env}/conda-meta/habitat-sim-0.1.7-py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7.json" ]]; then
  printf 'Headless record already exists; verify binary instead of reinstalling.\n'
else
"${habitat_manager}" install --no-rc --root-prefix "${habitat_runtime}/mamba" \
  --prefix "${habitat_env}" --no-deps --yes --file "${repair_dir}/explicit.txt"
fi
# Explicit installs may leave both build records. Verify the actual binary
# before moving only the superseded ordinary-build metadata into a backup.
"${habitat_env}/bin/python" "${project_root}/checks/reconcile_habitat_metadata.py" --apply
cp -- "${habitat_env}/conda-meta/history" "${repair_dir}/history_after.txt"
"${habitat_manager}" list --no-rc --root-prefix "${habitat_runtime}/mamba" \
  --prefix "${habitat_env}" --json > "${repair_dir}/packages_after.json"
printf 'Checking repaired build and imports (no rendering yet).\n'
bash "${project_root}/scripts/install_habitat_environment.sh"
printf 'HABITAT_HEADLESS_PACKAGE_READY\nLog: %s/repair.log\n' "${repair_dir}"
