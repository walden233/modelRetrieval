#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
EVAL_DIR="${EVAL_DIR:-artifacts/semantic/rh20t/cfg2/evaluation/pair}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/semantic/rh20t/cfg2/final_charts}"
DIRECTION="${DIRECTION:-human_to_robot}"
LEVEL="${LEVEL:-pair}"
DPI="${DPI:-400}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/semantic/export_final_semantic_charts.py \
  --eval-dir "$EVAL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --direction "$DIRECTION" \
  --level "$LEVEL" \
  --dpi "$DPI" \
  "$@"
