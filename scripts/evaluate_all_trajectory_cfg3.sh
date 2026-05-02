#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
RUNS_JSON="${RUNS_JSON:-artifacts/runs/trajectory/trajectory_final_runs.json}"
CFG3_DATA_ROOT="${CFG3_DATA_ROOT:-dataset/RH20T_subset/RH20T_cfg3}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-cfg3_all_test}"
CHART_OUTPUT_DIR="${CHART_OUTPUT_DIR:-artifacts/runs/trajectory/cfg3_final_charts}"
TOP_K="${TOP_K:-5}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_model.pth}"
DIRECTION="${DIRECTION:-human_to_robot}"
LEVEL="${LEVEL:-task}"
DPI="${DPI:-400}"

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
  output_dir="$run/$OUTPUT_SUBDIR"

  if [[ ! -f "$config" ]]; then
    echo "[$key] missing params.json: $config" >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[$key] missing checkpoint: $checkpoint" >&2
    exit 1
  fi

  echo "[$key] evaluate trajectory cfg3 all-test: $run -> $output_dir"
  "$PYTHON_BIN" runs/trajectory/evaluate_retrieval.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --data-root "$CFG3_DATA_ROOT" \
    --all-as-test \
    --split test \
    --output-dir "$output_dir" \
    --top-k "$TOP_K" \
    "$@"
done

"$PYTHON_BIN" runs/trajectory/export_final_trajectory_charts.py \
  --runs-json "$RUNS_JSON" \
  --output-dir "$CHART_OUTPUT_DIR" \
  --metrics-subdir "$OUTPUT_SUBDIR" \
  --metrics-filename metrics_comparison.png \
  --curves-filename curves_comparison.png \
  --direction "$DIRECTION" \
  --level "$LEVEL" \
  --dpi "$DPI"

echo "cfg3 trajectory metrics chart: $CHART_OUTPUT_DIR/metrics_comparison.png"
