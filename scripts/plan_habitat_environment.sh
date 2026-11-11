#!/usr/bin/env bash
# Resolve the official VLN-CE simulator dependencies without installing packages.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_runtime="${HOME}/.local/share/project2-habitat"
habitat_manager="${habitat_runtime}/bin/micromamba"
habitat_env="${habitat_runtime}/mamba/envs/vlnce-017"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
plan_dir="${project_root}/tmp/habitat_setup/resolve_${run_id}"

[[ -x "${habitat_manager}" ]] || {
  printf 'Run setup_habitat_manager.sh first.\n' >&2
  exit 1
}
if [[ -e "${habitat_env}" ]]; then
  printf 'Target already exists: %s. Inspect it before planning a new environment.\n' "${habitat_env}" >&2
  exit 1
fi
mkdir -p -- "${plan_dir}"
exec > >(tee "${plan_dir}/resolve.log") 2>&1
trap 'printf "HABITAT_ENV_PLAN_FAILED: see %s/resolve.log\n" "${plan_dir}" >&2' ERR

printf 'Manager version: '
"${habitat_manager}" --version
printf 'Resolving Python 3.6 + Habitat-Sim 0.1.7 headless (no installation).\n'
printf 'The first run downloads channel metadata; this can take several minutes.\n'
plan_command=(
  "${habitat_manager}" create
  --no-rc
  --root-prefix "${habitat_runtime}/mamba"
  --prefix "${habitat_env}"
  --override-channels
  --channel aihabitat
  --channel conda-forge
  --channel-priority flexible
  --dry-run
  'python=3.6'
  'habitat-sim=0.1.7=py3.6_headless_linux_856d4b08c1a2632626bf0d205bf46471a99502b7'
  headless
  'imageio=2.9.0'
)
printf '%q ' "${plan_command[@]}" > "${plan_dir}/command.txt"
printf '\n' >> "${plan_dir}/command.txt"
"${plan_command[@]}"
printf 'HABITAT_ENV_PLAN_READY\nLog: %s/resolve.log\n' "${plan_dir}"
printf 'Dependency resolution passed. Package installation, imports and rendering remain untested.\n'
