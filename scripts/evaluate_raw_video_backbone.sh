#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
CONFIG="${CONFIG:-configs/video/vjepa_rh20t_baseline.json}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/video/RAW_VJEPA_backbone}"
SPLIT="${SPLIT:-test}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-artifacts/runs/video/video_vjepa_rh20t_baseline_20260429_145608/split_manifest.json}"
TOP_K="${TOP_K:-5}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" runs/video/evaluate_raw_video_backbone.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --split "$SPLIT" \
  --split-manifest "$SPLIT_MANIFEST" \
  --top-k "$TOP_K" \
  "$@"
