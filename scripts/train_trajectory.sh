#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
CONFIG="${CONFIG:-configs/trajectory/T1_trajectory_baseline_scene.json}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

cd "$ROOT_DIR"

ARGS=(--config "$CONFIG")
if [[ -n "$DATA_ROOT" ]]; then
  ARGS+=(--data-root "$DATA_ROOT")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

exec "$PYTHON_BIN" runs/trajectory/train_trajectory.py "${ARGS[@]}" "$@"
