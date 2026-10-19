#!/usr/bin/env bash
# 项目2：首次启动老师方案中的 LMDeploy 2B 单卡后端。
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_env="${HOME}/.venvs/project2"
model_path="${project_root}/models/Qwen3-VL-2B-Instruct"

if [[ ! -x "${project_env}/bin/lmdeploy" ]]; then
  printf 'Project2 LMDeploy is missing: %s\n' "${project_env}/bin/lmdeploy" >&2
  exit 1
fi
if [[ ! -f "${model_path}/model.safetensors" ]]; then
  printf 'Model weights are missing: %s\n' "${model_path}" >&2
  exit 1
fi

source "${project_env}/bin/activate"
cd -- "${project_root}"
mkdir -p -- "${project_root}/logs"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${project_root}/logs/lmdeploy_2b_${run_id}_$$.log"

# 首次运行的容量配置：一张 GPU、单请求、上下文上限 2048 token。
# KV 缓存使用加载权重后空闲显存的 20%，为视觉计算与桌面程序留空间。
# 这是待验证的启动配置，后续基线必须记录这些值。
launch_command=(
  "${project_env}/bin/lmdeploy" serve api_server "${model_path}"
  --backend pytorch
  --model-name Qwen3-VL-2B-Instruct
  --server-name 127.0.0.1
  --server-port 23333
  --tp 1
  --dtype bfloat16
  --session-len 2048
  --max-batch-size 1
  --max-concurrent-requests 1
  --vision-max-batch-size 1
  --max-prefill-token-num 2048
  --cache-max-entry-count 0.2
  --distributed-executor-backend mp
  --disable-metrics
  --log-level INFO
)

{
  printf 'Project2: LMDeploy 2B single-GPU startup\n'
  printf 'Model directory: %s\n' "${model_path}"
  printf 'Log file: %s\n' "${log_path}"
  printf 'API URL: http://127.0.0.1:23333\n'
  printf 'Keep this terminal open while the server is running.\n'
  python --version
  nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
  printf 'Command: '
  printf '%q ' "${launch_command[@]}"
  printf '\n'
  "${launch_command[@]}"
} 2>&1 | tee "${log_path}"
