#!/usr/bin/env bash
# Authorized Matterport3D Habitat archive only; never invokes the full downloader.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_python="${HOME}/.local/share/project2-habitat/mamba/envs/vlnce-017/bin/python"
download_dir="${project_root}/data/scene_datasets/mp3d/downloads"
run_dir="${project_root}/tmp/habitat_setup/mp3d_download_$(date -u +%Y%m%dT%H%M%SZ)_$$"
archive="${download_dir}/mp3d_habitat.zip"
scene_url='https://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip'
curl_network=()
if [[ $# -eq 1 && "${1}" == --direct ]]; then
  curl_network=(--noproxy '*')
elif [[ $# -ne 0 ]]; then
  printf 'Usage: bash scripts/download_mp3d_habitat.sh [--direct]\n' >&2
  exit 2
fi
[[ -x "${habitat_python}" ]] || { printf 'Missing isolated Habitat Python.\n' >&2; exit 1; }
command -v curl >/dev/null
mkdir -p -- "${download_dir}" "${run_dir}"
exec > >(tee "${run_dir}/download.log") 2>&1
trap 'printf "MP3D_DOWNLOAD_INCOMPLETE: retained partial file; see %s/download.log\n" "${run_dir}" >&2' ERR
printf 'Downloading only the official Habitat archive; no extraction or experiments.\n'
printf 'Source: %s\nDestination: %s\n' "${scene_url}" "${archive}"
printf 'Local partial files are retained for the next run.\n'
if [[ ${#curl_network[@]} -gt 0 ]]; then
  printf 'Network mode: bypass curl proxies. OS/VPN/TUN routing must be switched separately.\n'
else
  printf 'Network mode: current shell/system routing.\n'
fi
curl "${curl_network[@]}" --fail --location --head --retry 2 --connect-timeout 20 --max-time 60 \
  --dump-header "${run_dir}/http_headers.txt" --output /dev/null "${scene_url}"
"${habitat_python}" "${project_root}/checks/verify_mp3d_archive.py" \
  --archive "${archive}" --headers "${run_dir}/http_headers.txt" --preflight
if [[ ! -f "${archive}" && "$(cat "${run_dir}/transfer_required.txt")" == yes ]]; then
  curl "${curl_network[@]}" --fail --location --retry 3 --retry-delay 5 --connect-timeout 20 \
    --speed-limit 1024 --speed-time 120 --continue-at - \
    --output "${archive}.part" "${scene_url}"
fi
"${habitat_python}" "${project_root}/checks/verify_mp3d_archive.py" \
  --archive "${archive}" --headers "${run_dir}/http_headers.txt"
printf 'MP3D_HABITAT_ARCHIVE_READY\nArchive: %s\nBackup checksum: %s.sha256\nLog: %s/download.log\n' \
  "${archive}" "${archive}" "${run_dir}"
