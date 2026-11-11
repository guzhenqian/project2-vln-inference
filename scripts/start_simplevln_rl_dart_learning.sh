#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
keep_ratio="${1:-1.0}"
profile="${2:-short}"
usage() {
  printf 'Usage: bash scripts/start_simplevln_rl_dart_learning.sh [1.0|0.75] [short|navigation]\n' >&2
}
if (( $# > 2 )); then
  usage
  exit 2
fi
case "$keep_ratio" in
  1.0|0.75) ;;
  *) usage; exit 2 ;;
esac
case "$profile" in
  short|navigation) ;;
  *) usage; exit 2 ;;
esac
mkdir -p logs
console_log="${PWD}/logs/simplevln_rl_dart_${profile}_keep${keep_ratio}_$(date -u +%Y%m%dT%H%M%S%NZ)_$$.log"
export PROJECT2_DART_CONSOLE_LOG="$console_log"
exec "${HOME}/.venvs/project2/bin/python" -u checks/serve_navigation_dart_learning.py \
  --keep-ratio "$keep_ratio" --profile "$profile" > >(tee -a "$console_log") 2>&1
