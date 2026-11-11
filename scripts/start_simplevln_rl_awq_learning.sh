#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
exec "${HOME}/.venvs/project2/bin/lmdeploy" serve api_server "${PWD}/models/SimpleVLN_Qwen3VL_4B_RL_AWQ_learning_20261104" \
  --backend pytorch --model-name SimpleVLN-Qwen3VL-4B-RL-AWQ-learning-20261104 \
  --server-name 127.0.0.1 --server-port 23333 --tp 1 --dtype float16 \
  --session-len 4096 --max-batch-size 1 --max-concurrent-requests 1 \
  --vision-max-batch-size 1 --max-prefill-token-num 1024 --cache-max-entry-count 0.35 \
  --distributed-executor-backend mp --disable-metrics --log-level INFO
