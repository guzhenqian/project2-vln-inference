#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "${project_root}"
model_path="${project_root}/models/SimpleVLN_Qwen3VL_4B_SFT_lmdeploy"
[[ -f "${model_path}/serving_adapter.json" ]] || exit 1
mkdir -p logs
exec "${HOME}/.venvs/project2/bin/lmdeploy" serve api_server "${model_path}" \
  --backend pytorch --model-name SimpleVLN-Qwen3VL-4B-SFT \
  --server-name 127.0.0.1 --server-port 23333 --tp 1 --dtype bfloat16 \
  --session-len 4096 --max-batch-size 1 --max-concurrent-requests 1 \
  --vision-max-batch-size 1 --max-prefill-token-num 1024 --cache-max-entry-count 0.35 \
  --distributed-executor-backend mp --disable-metrics --log-level INFO \
  > >(tee "logs/simplevln_4b_$(date -u +%Y%m%dT%H%M%SZ)_$$.log") 2>&1
