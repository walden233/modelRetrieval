#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
MANIFEST="${MANIFEST:-artifacts/experiments/paper_baseline_cfg2_cfg3/pipeline_manifest.json}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" runs/export_paper_bundle.py --manifest "$MANIFEST" "$@"
