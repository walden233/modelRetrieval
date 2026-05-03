#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ttt/miniconda3/envs/torch2/bin/python}"
LIBRARY_DIR="${LIBRARY_DIR:-artifacts/retrieval/rh20t_cfg3_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/retrieval/rh20t_cfg3_v1/eval/ablation_uniform}"
LEVEL="${LEVEL:-mixed}"
REQUIRE_MODALITIES="${REQUIRE_MODALITIES:-video,trajectory,semantic_text}"
TOP_K="${TOP_K:-10}"

cd "$ROOT_DIR"

declare -a LABELS=(
  "Video"
  "Trajectory"
  "Semantic"
  "Video+Trajectory"
  "Video+Semantic"
  "Trajectory+Semantic"
  "Video+Trajectory+Semantic"
)

declare -a CONFIGS=(
  "configs/retrieval/ablation/video_only_uniform.json"
  "configs/retrieval/ablation/trajectory_only_uniform.json"
  "configs/retrieval/ablation/semantic_text_only_uniform.json"
  "configs/retrieval/ablation/video_trajectory_uniform.json"
  "configs/retrieval/ablation/video_semantic_text_uniform.json"
  "configs/retrieval/ablation/trajectory_semantic_text_uniform.json"
  "configs/retrieval/ablation/video_trajectory_semantic_text_uniform.json"
)

declare -a DIRS=(
  "video_only"
  "trajectory_only"
  "semantic_text_only"
  "video_trajectory"
  "video_semantic_text"
  "trajectory_semantic_text"
  "video_trajectory_semantic_text"
)

mkdir -p "$OUTPUT_ROOT"
RUNS_JSON="$OUTPUT_ROOT/ablation_runs.json"
printf '{\n' > "$RUNS_JSON"

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  config="${CONFIGS[$index]}"
  output_dir="$OUTPUT_ROOT/${DIRS[$index]}"
  echo "Running ablation: $label -> $output_dir"
  "$PYTHON_BIN" runs/retrieval/evaluate_retrieval_system.py \
    --library-dir "$LIBRARY_DIR" \
    --config "$config" \
    --level "$LEVEL" \
    --require-modalities "$REQUIRE_MODALITIES" \
    --output-dir "$output_dir" \
    --top-k "$TOP_K"

  comma=","
  if [[ "$index" -eq "$((${#LABELS[@]} - 1))" ]]; then
    comma=""
  fi
  printf '  "%s": "%s"%s\n' "$label" "$output_dir" "$comma" >> "$RUNS_JSON"
done

printf '}\n' >> "$RUNS_JSON"

echo "Ablation runs JSON: $RUNS_JSON"
echo "Export charts with:"
echo "RUNS_JSON=$RUNS_JSON OUTPUT_DIR=$OUTPUT_ROOT/charts LEVEL=scene scripts/export_retrieval_system_charts.sh"
