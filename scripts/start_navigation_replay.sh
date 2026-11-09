#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
exec python3 checks/serve_navigation_replay.py --run "${1:-logs/learning_baseline6_20261031_153519}" --port "${2:-8766}"
