#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-torch2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/trajectory}"
FINAL_CHART_ROOT="${FINAL_CHART_ROOT:-$OUTPUT_ROOT/final_charts}"
TOP_K="${TOP_K:-5}"
SKIP_CONDA="${SKIP_CONDA:-0}"

cd "$ROOT_DIR"

exec env \
  CONDA_ENV="$CONDA_ENV" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  FINAL_CHART_ROOT="$FINAL_CHART_ROOT" \
  TOP_K="$TOP_K" \
  SKIP_CONDA="$SKIP_CONDA" \
  bash runs/trajectory/run_all_trajectory_experiments.sh "$@"
