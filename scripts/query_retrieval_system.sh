#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
LIBRARY_DIR="${LIBRARY_DIR:-artifacts/retrieval/rh20t_cfg2_v1}"
CONFIG="${CONFIG:-configs/retrieval/system_default.json}"
TOP_K="${TOP_K:-10}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/retrieval/query_retrieval_system.py \
  --library-dir "$LIBRARY_DIR" \
  --config "$CONFIG" \
  --top-k "$TOP_K" \
  "$@"
