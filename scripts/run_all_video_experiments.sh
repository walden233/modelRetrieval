#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-torch2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/video}"
FINAL_CHART_ROOT="${FINAL_CHART_ROOT:-$OUTPUT_ROOT/final_charts}"
TOP_K="${TOP_K:-5}"
SKIP_CONDA="${SKIP_CONDA:-0}"
RUN_E6_DUAL_HEAD_DUPLICATE="${RUN_E6_DUAL_HEAD_DUPLICATE:-0}"

cd "$ROOT_DIR"

exec env \
  CONDA_ENV="$CONDA_ENV" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  FINAL_CHART_ROOT="$FINAL_CHART_ROOT" \
  TOP_K="$TOP_K" \
  SKIP_CONDA="$SKIP_CONDA" \
  RUN_E6_DUAL_HEAD_DUPLICATE="$RUN_E6_DUAL_HEAD_DUPLICATE" \
  bash runs/video/run_all_video_experiments.sh "$@"
