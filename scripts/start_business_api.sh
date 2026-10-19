#!/usr/bin/env bash
# 项目2：启动老师的 FastAPI 业务层，连接当前 LMDeploy 模型后端。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_env="${HOME}/.venvs/project2"
source_dir="${project_root}/Qwen3VL-main/Qwen3VL-main"

if [[ ! -x "${project_env}/bin/python" || ! -f "${source_dir}/app/main.py" ]]; then
  printf 'Project2 environment or business source is missing.\n' >&2
  exit 1
fi
source "${project_env}/bin/activate"
cd -- "${source_dir}"

# app/config.py 在导入时读取环境变量，故需在启动 Python 前设置。
export MODEL_PATH="${project_root}/models/Qwen3-VL-2B-Instruct"
export TENSOR_PARALLEL_SIZE=1
export MAX_MODEL_LEN=2048
export LMDEPLOY_BASE_URL="http://127.0.0.1:23333"
export HOST="127.0.0.1"
export PORT=20000
export PROJECT2_DIAGNOSTICS=1
export PROJECT2_NAV_PROMPT_PROFILE="${PROJECT2_NAV_PROMPT_PROFILE:-baseline_v1}"
case "${PROJECT2_NAV_PROMPT_PROFILE}" in
  baseline_v1|grounded_stop_v2) ;;
  *) printf 'Unknown PROJECT2_NAV_PROMPT_PROFILE: %s\n' "${PROJECT2_NAV_PROMPT_PROFILE}" >&2; exit 1 ;;
esac

mkdir -p -- "${project_root}/logs"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${project_root}/logs/business_api_${run_id}_$$.log"
launch_command=(
  "${project_env}/bin/python" -m uvicorn app.main:app
  --host "${HOST}" --port "${PORT}" --log-level info
)

{
  printf 'Project2: teacher FastAPI business service\n'
  printf 'Business source: %s\n' "${source_dir}"
  printf 'Model directory: %s\n' "${MODEL_PATH}"
  printf 'Backend URL: %s\n' "${LMDEPLOY_BASE_URL}"
  printf 'Business URL: http://%s:%s\n' "${HOST}" "${PORT}"
  printf 'Navigation diagnostics: enabled (raw model text and parsed result in this log)\n'
  printf 'Navigation prompt profile: %s\n' "${PROJECT2_NAV_PROMPT_PROFILE}"
  printf 'Log file: %s\n' "${log_path}"
  printf 'Keep this terminal open while the business service is running.\n'

  # 老师原 /health 仅检查客户端对象是否存在；先核实后端确实可达。
  python - <<'PY' || exit 1
import json
import os
from urllib.request import urlopen

url = os.environ["LMDEPLOY_BASE_URL"] + "/v1/models"
try:
    with urlopen(url, timeout=10) as response:
        models = json.load(response).get("data", [])
except Exception as error:
    raise SystemExit(f"FAIL: LMDeploy is not reachable. Keep the model server running. {error}")

ids = [model.get("id") for model in models]
if "Qwen3-VL-2B-Instruct" not in ids:
    raise SystemExit(f"FAIL: unexpected backend model list: {ids}")
print("Backend check: Qwen3-VL-2B-Instruct is available.")
PY

  printf 'Command: '
  printf '%q ' "${launch_command[@]}"
  printf '\n'
  "${launch_command[@]}"
} 2>&1 | tee "${log_path}"
