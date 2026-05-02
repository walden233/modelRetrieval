#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
ANNOTATIONS="${ANNOTATIONS:-artifacts/semantic/rh20t/cfg2/annotations/normalized_annotations.jsonl}"
GALLERY_ANNOTATIONS="${GALLERY_ANNOTATIONS:-}"
QUERY_ROLE="${QUERY_ROLE:-human}"
GALLERY_ROLE="${GALLERY_ROLE:-robot}"
POSITIVE_KEY="${POSITIVE_KEY:-pair_id}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/semantic/rh20t/cfg2/evaluation/pair}"
TOP_K="${TOP_K:-10}"

cd "$ROOT_DIR"

ARGS=(
  runs/semantic/evaluate_semantic_retrieval.py
  --annotations "$ANNOTATIONS"
  --query-role "$QUERY_ROLE"
  --gallery-role "$GALLERY_ROLE"
  --positive-key "$POSITIVE_KEY"
  --output-dir "$OUTPUT_DIR"
  --top-k "$TOP_K"
)

if [[ -n "$GALLERY_ANNOTATIONS" ]]; then
  ARGS+=(--gallery-annotations "$GALLERY_ANNOTATIONS")
fi

exec "$PYTHON_BIN" "${ARGS[@]}" "$@"
