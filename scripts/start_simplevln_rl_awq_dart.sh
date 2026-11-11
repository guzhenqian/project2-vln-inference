#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
keep_ratio="${1:-1.0}"
if (( $# > 1 )); then
  printf 'Usage: bash scripts/start_simplevln_rl_awq_dart.sh [1.0|0.75]\n' >&2
  exit 2
fi
case "$keep_ratio" in
  1.0|0.75) ;;
  *) printf 'Expected keep ratio 1.0 or 0.75\n' >&2; exit 2 ;;
esac
mkdir -p logs
console_log="${PWD}/logs/simplevln_rl_awq_dart_keep${keep_ratio}_$(date -u +%Y%m%dT%H%M%S%NZ)_$$.log"
export PROJECT2_AWQ_DART_CONSOLE_LOG="$console_log"
exec "${HOME}/.venvs/project2/bin/python" -B -u checks/serve_navigation_awq_dart.py \
  --run --keep-ratio "$keep_ratio" --profile navigation > >(tee -a "$console_log") 2>&1
