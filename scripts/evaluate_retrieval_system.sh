#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
LIBRARY_DIR="${LIBRARY_DIR:-artifacts/retrieval/rh20t_cfg2_v1}"
CONFIG="${CONFIG:-configs/retrieval/system_eval_full_modalities.json}"
LEVEL="${LEVEL:-mixed}"
REQUIRE_MODALITIES="${REQUIRE_MODALITIES:-video,trajectory,semantic_text}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/retrieval/rh20t_cfg2_v1/eval/full_modalities_mixed}"
TOP_K="${TOP_K:-10}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/retrieval/evaluate_retrieval_system.py \
  --library-dir "$LIBRARY_DIR" \
  --config "$CONFIG" \
  --level "$LEVEL" \
  --require-modalities "$REQUIRE_MODALITIES" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  "$@"
