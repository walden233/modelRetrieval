#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
RUNS_JSON="${RUNS_JSON:-artifacts/retrieval/system_final_runs.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/retrieval/final_charts}"
LEVEL="${LEVEL:-scene}"
DPI="${DPI:-400}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/retrieval/export_retrieval_system_charts.py \
  --runs-json "$RUNS_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --level "$LEVEL" \
  --dpi "$DPI" \
  "$@"
