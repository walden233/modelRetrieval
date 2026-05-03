#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
CONFIG="${CONFIG:-configs/retrieval/library_rh20t_cfg2.json}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/retrieval/build_retrieval_library.py \
  --config "$CONFIG" \
  "$@"
