#!/usr/bin/env bash
# 项目2：用独立环境启动老师原 Gradio 页面。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
webui_python="${HOME}/.venvs/project2-webui/bin/python"
webui_source="${project_root}/Qwen3VL-main/Qwen3VL-main/app/vln_webui.py"
if [[ ! -x "${webui_python}" || ! -f "${webui_source}" ]]; then
  printf 'FAIL: WebUI environment or source is missing.\n' >&2
  exit 1
fi

export GRADIO_ANALYTICS_ENABLED=False
export PYTHONUNBUFFERED=1
mkdir -p -- "${project_root}/logs"
log_path="${project_root}/logs/webui_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
launch_command=(
  "${webui_python}" "${webui_source}"
  --host 127.0.0.1 --port 7860 --api-url http://127.0.0.1:20000
)

{
  printf 'Project2: teacher Gradio WebUI\n'
  printf 'Python: %s\n' "${webui_python}"
  printf 'Source: %s\n' "${webui_source}"
  printf 'Business API: http://127.0.0.1:20000\n'
  printf 'Open in Windows browser: http://localhost:7860\n'
  printf 'Log file: %s\n' "${log_path}"
  printf 'Keep the model, business API and this WebUI terminal running.\n'

  "${webui_python}" - <<'PY' || exit 1
import json
from urllib.request import urlopen

try:
    with urlopen("http://127.0.0.1:20000/health", timeout=10) as response:
        health = json.load(response)
except Exception as error:
    raise SystemExit(f"FAIL: business API is not reachable: {error}")
if health.get("status") != "healthy":
    raise SystemExit(f"FAIL: unexpected business health response: {health}")
print("Business API check:", health.get("status"), health.get("model"))
print("Note: this is a connectivity check; image and navigation checks are separate.")
PY

  printf 'Command: '
  printf '%q ' "${launch_command[@]}"
  printf '\n'
  "${launch_command[@]}"
} 2>&1 | tee "${log_path}"
