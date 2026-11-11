#!/usr/bin/env bash
# One public-scene RGB frame; no model requests or navigation evaluation.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_python="${HOME}/.local/share/project2-habitat/mamba/envs/vlnce-017/bin/python"
run_dir="${project_root}/tmp/habitat_setup/render_$(date -u +%Y%m%dT%H%M%SZ)_$$"
cache_dir="${project_root}/tmp/habitat_setup/public_scenes"
archive="${cache_dir}/habitat-test-scenes.zip"
scene_url='https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip'
render_mode=cuda
extra_args=()
for option in "$@"; do
  case "${option}" in
    --software) render_mode=software ;;
    --action-smoke) extra_args+=(--action-smoke) ;;
    *) printf 'Usage: bash scripts/check_habitat_render.sh [--software] [--action-smoke]\n' >&2; exit 2 ;;
  esac
done
[[ -x "${habitat_python}" ]] || { printf 'Missing Habitat Python.\n' >&2; exit 1; }
mkdir -p -- "${run_dir}" "${cache_dir}"
exec > >(tee "${run_dir}/render.log") 2>&1
trap 'printf "HABITAT_RENDER_FAILED: see %s/render.log\n" "${run_dir}" >&2' ERR
printf 'Run directory: %s\nScene source: %s\n' "${run_dir}" "${scene_url}"
printf '%s\n' "${scene_url}" > "${run_dir}/scene_source.txt"
printf 'Requested render mode: %s\n' "${render_mode}"
if [[ ! -f "${archive}" ]]; then
  printf 'Downloading official public test scenes (first run only).\n'
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 600 \
    "${scene_url}" --output "${run_dir}/habitat-test-scenes.zip.part"
  "${habitat_python}" -m zipfile -t "${run_dir}/habitat-test-scenes.zip.part"
  mv -- "${run_dir}/habitat-test-scenes.zip.part" "${archive}"
fi
sha256sum "${archive}" > "${run_dir}/archive_sha256.txt"
{ nvidia-smi || true; } > "${run_dir}/nvidia-smi.txt" 2>&1
# Headless EGL test; changes DISPLAY only for this child process.
render_env=(env -u DISPLAY PYTHONUNBUFFERED=1)
if [[ "${render_mode}" == software ]]; then
  render_env+=(LIBGL_ALWAYS_SOFTWARE=1)
fi
"${render_env[@]}" "${habitat_python}" \
  "${project_root}/checks/check_habitat_render.py" \
  --archive "${archive}" --output "${run_dir}" --render-mode "${render_mode}" "${extra_args[@]}"
printf 'Image: %s/rgb.png\nLog: %s/render.log\n' "${run_dir}" "${run_dir}"
