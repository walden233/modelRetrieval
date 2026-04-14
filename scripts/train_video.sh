#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/home/ttt/miniconda3/envs/torch2/bin/python"

exec "$PYTHON_BIN" \
  "$ROOT_DIR/tools/train_video.py" \
  --config "$ROOT_DIR/configs/video/vjepa_whirl.json" \
  "$@"
