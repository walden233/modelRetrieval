#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
CONFIG="${CONFIG:-configs/video/vjepa_rh20t_baseline.json}"
CSV_PATH="${CSV_PATH:-}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME="${RESUME:-}"

cd "$ROOT_DIR"

ARGS=(--config "$CONFIG")
if [[ -n "$CSV_PATH" ]]; then
  ARGS+=(--csv-path "$CSV_PATH")
fi
if [[ -n "$DATA_ROOT" ]]; then
  ARGS+=(--data-root "$DATA_ROOT")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

exec "$PYTHON_BIN" runs/video/train_video.py "${ARGS[@]}" "$@"
