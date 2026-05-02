#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
RUNS_JSON="${RUNS_JSON:-artifacts/runs/trajectory/trajectory_final_runs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/trajectory/final_charts}"
DIRECTION="${DIRECTION:-human_to_robot}"
LEVEL="${LEVEL:-task}"
DPI="${DPI:-400}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/trajectory/export_final_trajectory_charts.py \
  --runs-json "$RUNS_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --direction "$DIRECTION" \
  --level "$LEVEL" \
  --dpi "$DPI" \
  "$@"
