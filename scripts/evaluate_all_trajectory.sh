#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
RUNS_JSON="${RUNS_JSON:-artifacts/runs/trajectory/trajectory_final_runs.json}"
SPLIT="${SPLIT:-test}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-final_test}"
TOP_K="${TOP_K:-5}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_model.pth}"
SKIP_MISSING_CHECKPOINT="${SKIP_MISSING_CHECKPOINT:-1}"

cd "$ROOT_DIR"

mapfile -t RUN_ENTRIES < <(
  "$PYTHON_BIN" - "$RUNS_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit(f"runs json must be an object: {path}")

for key, value in payload.items():
    if isinstance(value, str):
        run_path = value
    elif isinstance(value, dict):
        run_path = value.get("run_path") or value.get("path") or value.get("run_dir")
    else:
        run_path = None
    if not run_path:
        raise SystemExit(f"Invalid run entry for {key}: {value!r}")
    print(f"{key}\t{run_path}")
PY
)

for entry in "${RUN_ENTRIES[@]}"; do
  key="${entry%%$'\t'*}"
  run="${entry#*$'\t'}"
  config="$run/params.json"
  checkpoint="$run/$CHECKPOINT_NAME"
  split_manifest="$run/split_manifest.json"
  output_dir="$run/$OUTPUT_SUBDIR"

  if [[ ! -f "$checkpoint" ]]; then
    if [[ "$SKIP_MISSING_CHECKPOINT" == "1" ]]; then
      echo "[$key] skip: missing checkpoint $checkpoint"
      continue
    fi
    echo "[$key] missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -f "$config" ]]; then
    echo "[$key] missing params.json: $config" >&2
    exit 1
  fi
  if [[ ! -f "$split_manifest" ]]; then
    echo "[$key] missing split_manifest.json: $split_manifest" >&2
    exit 1
  fi

  echo "[$key] evaluate trajectory: $run -> $output_dir"
  "$PYTHON_BIN" runs/trajectory/evaluate_retrieval.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split "$SPLIT" \
    --split-manifest "$split_manifest" \
    --output-dir "$output_dir" \
    --top-k "$TOP_K" \
    "$@"
done
