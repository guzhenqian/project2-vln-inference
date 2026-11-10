#!/usr/bin/env bash
# Habitat-Lab core source-mode entry point, isolated from model/WebUI envs.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_python="${HOME}/.local/share/project2-habitat/mamba/envs/vlnce-017/bin/python"
source_dir="${project_root}/tmp/habitat_setup/sources/habitat-lab-0.1.7"
[[ -f "${source_dir}/.project2_source_sha256" ]] || {
  printf 'Task source not prepared. Run install_habitat_task_environment.sh first.\n' >&2; exit 1;
}
exec env PYTHONNOUSERSITE=1 PYTHONPATH="${source_dir}" "${habitat_python}" "$@"
