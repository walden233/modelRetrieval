#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
CONFIG="${CONFIG:-configs/experiments/paper_baseline_cfg2_cfg3.json}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" runs/run_experiment_pipeline.py --config "$CONFIG" "$@"
