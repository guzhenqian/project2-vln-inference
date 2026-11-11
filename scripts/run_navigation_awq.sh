#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
out="logs/navigation_awq_quant_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
printf '%s\n' "$out" > tmp/navigation_awq_quant_latest.txt
exec "${HOME}/.venvs/project2-quant312/bin/python" -u checks/quantize_navigation_awq.py > "$out" 2>&1
