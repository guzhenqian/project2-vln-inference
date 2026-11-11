#!/usr/bin/env bash
# Preview one official CE starting pose in a real MP3D scene, using software EGL.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_python="${HOME}/.local/share/project2-habitat/mamba/envs/vlnce-017/bin/python"
run_dir="${project_root}/tmp/habitat_setup/mp3d_preview_$(date -u +%Y%m%dT%H%M%SZ)_$$"
scene="${project_root}/data/scene_datasets/mp3d/e9zR4mvMWw7/e9zR4mvMWw7.glb"
episode="${project_root}/data/datasets/R2R_VLNCE_v1-3_smoke/episode_188.json"
[[ -x "${habitat_python}" && -f "${scene}" && -f "${episode}" ]] || {
  printf 'Missing prepared smoke inputs or Habitat Python.\n' >&2; exit 1;
}
mkdir -p -- "${run_dir}"
exec > >(tee "${run_dir}/render.log") 2>&1
trap 'printf "MP3D_EPISODE_PREVIEW_FAILED: see %s/render.log\n" "${run_dir}" >&2' ERR
printf 'Preview: official val_seen episode 188; software renderer; no navigation evaluation.\n'
cp -- "${episode}" "${run_dir}/episode.json"
env -u DISPLAY LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1 "${habitat_python}" \
  "${project_root}/checks/check_habitat_render.py" \
  --scene "${scene}" --episode "${episode}" --render-mode software --output "${run_dir}"
printf 'MP3D_EPISODE_PREVIEW_READY\nImage: %s/rgb.png\nLog: %s/render.log\n' "${run_dir}" "${run_dir}"
