#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
RUN="${RUN:-artifacts/runs/video/video_vjepa_rh20t_baseline_20260429_145608}"
CONFIG="${CONFIG:-$RUN/params.json}"
CHECKPOINT="${CHECKPOINT:-$RUN/best_model.pth}"
SPLIT="${SPLIT:-test}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$RUN/split_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN/final_test}"
TOP_K="${TOP_K:-5}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/video/evaluate_video.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --top-k "$TOP_K" \
  "$@"
