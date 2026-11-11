#!/usr/bin/env bash
# Install only the environment manager; Habitat packages are a separate step.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
habitat_runtime="${HOME}/.local/share/project2-habitat"
habitat_manager="${habitat_runtime}/bin/micromamba"
habitat_package_url="https://micro.mamba.pm/api/micromamba/linux-64/latest"
run_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
setup_dir="${project_root}/tmp/habitat_setup/manager_${run_id}"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf 'This script requires Linux x86_64 (run inside WSL Ubuntu).\n' >&2
  exit 1
fi
for utility in curl tar bzip2 sha256sum; do
  command -v "${utility}" >/dev/null || {
    printf 'Missing utility: %s\n' "${utility}" >&2
    exit 1
  }
done
mkdir -p -- "${setup_dir}" "${habitat_runtime}/bin"
exec > >(tee "${setup_dir}/setup.log") 2>&1
trap 'printf "SETUP_FAILED: see %s/setup.log\n" "${setup_dir}" >&2' ERR

if [[ -e "${habitat_manager}" ]]; then
  [[ -x "${habitat_manager}" ]] || {
    printf 'Existing manager is not executable; refusing to replace it.\n' >&2
    exit 1
  }
  printf 'Reusing existing manager: %s\n' "${habitat_manager}"
else
  printf 'Downloading official Linux package...\n'
  curl --fail --location --retry 2 --connect-timeout 20 --max-time 180 \
    --output "${setup_dir}/micromamba.tar.bz2" \
    --write-out '%{url_effective}\n' "${habitat_package_url}" \
    > "${setup_dir}/download_url.txt"
  sha256sum "${setup_dir}/micromamba.tar.bz2" > "${setup_dir}/archive.sha256"
  tar -tjf "${setup_dir}/micromamba.tar.bz2" bin/micromamba
  tar -xjf "${setup_dir}/micromamba.tar.bz2" -C "${setup_dir}" bin/micromamba
  chmod u+x "${setup_dir}/bin/micromamba"
  "${setup_dir}/bin/micromamba" --version
  cp --no-clobber -- "${setup_dir}/bin/micromamba" "${habitat_manager}"
fi

"${habitat_manager}" --version | tee "${setup_dir}/version.txt"
sha256sum "${habitat_manager}" > "${setup_dir}/binary.sha256"
printf 'HABITAT_MANAGER_READY\nManager: %s\nFuture environment root: %s/mamba\nLog: %s/setup.log\n' \
  "${habitat_manager}" "${habitat_runtime}" "${setup_dir}"
printf 'Only micromamba was installed. Habitat and Python environments have not been created.\n'
