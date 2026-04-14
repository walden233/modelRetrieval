#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/ttt/miniconda3/envs/torch2/bin/python"

DEFAULT_CONFIG="$ROOT_DIR/configs/trajectory/augment_fair.json"
DEFAULT_CHECKPOINT="$ROOT_DIR/artifacts/runs/trajectory/trajectory_augment_fair_20260413_234320/best_model.pth"

if [[ $# -eq 0 ]]; then
  exec "$PYTHON_BIN" \
    "$ROOT_DIR/tools/evaluate_retrieval.py" \
    --config "$DEFAULT_CONFIG" \
    --checkpoint "$DEFAULT_CHECKPOINT"
fi

exec "$PYTHON_BIN" \
  "$ROOT_DIR/tools/evaluate_retrieval.py" \
  "$@"
