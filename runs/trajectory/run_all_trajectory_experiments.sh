#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-torch2}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/trajectory}"
FINAL_CHART_ROOT="${FINAL_CHART_ROOT:-${OUTPUT_ROOT}/final_charts}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${OUTPUT_ROOT}/run_logs_$(date +%Y%m%d_%H%M%S)}"
TOP_K="${TOP_K:-5}"

mkdir -p "${OUTPUT_ROOT}" "${FINAL_CHART_ROOT}" "${RUN_LOG_ROOT}"

if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda init script not found: ${CONDA_SH}" >&2
    echo "Set CONDA_SH=/path/to/conda.sh or SKIP_CONDA=1." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

declare -A RUN_DIRS
declare -A RUN_CONFIGS

config_experiment_name() {
  python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["experiment_name"])' "$1"
}

latest_run_dir() {
  local experiment_name="$1"
  local run_dir
  run_dir="$(
    find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "${experiment_name}_*" -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
  if [[ -z "${run_dir}" ]]; then
    echo "No run directory found for experiment_name=${experiment_name}" >&2
    exit 1
  fi
  echo "${run_dir}"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file missing: ${path}" >&2
    exit 1
  fi
}

train_and_eval() {
  local label="$1"
  local config="$2"
  local experiment_name
  local run_dir

  experiment_name="$(config_experiment_name "${config}")"

  echo "===== ${label}: train ${config} ====="
  python runs/trajectory/train_trajectory.py --config "${config}" 2>&1 | tee "${RUN_LOG_ROOT}/${label}_train.log"

  run_dir="$(latest_run_dir "${experiment_name}")"
  require_file "${run_dir}/best_model.pth"
  require_file "${run_dir}/split_manifest.json"

  echo "===== ${label}: evaluate ${run_dir} ====="
  python runs/trajectory/evaluate_retrieval.py \
    --config "${config}" \
    --checkpoint "${run_dir}/best_model.pth" \
    --split test \
    --top-k "${TOP_K}" \
    --split-manifest "${run_dir}/split_manifest.json" \
    --output-dir "${run_dir}/final_test" \
    2>&1 | tee "${RUN_LOG_ROOT}/${label}_eval.log"

  require_file "${run_dir}/final_test/metrics.json"
  RUN_DIRS["${label}"]="${run_dir}"
  RUN_CONFIGS["${label}"]="${config}"
}

select_best_run() {
  python - "$@" <<'PY'
import json
import sys
from pathlib import Path

best = None
items = sys.argv[1:]
if len(items) % 3 != 0:
    raise SystemExit("select_best_run expects LABEL CONFIG RUN_DIR triples")

for index in range(0, len(items), 3):
    label, config, run_dir = items[index:index + 3]
    metrics_path = Path(run_dir) / "final_test" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    score = metrics["human_to_robot"]["task"]["MRR"]
    candidate = (score, label, config, run_dir)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is None:
    raise SystemExit("No candidate runs were provided")

score, label, config, run_dir = best
print(f"{label}\t{config}\t{run_dir}\t{score}")
PY
}

echo "Output root: ${OUTPUT_ROOT}"
echo "Final chart root: ${FINAL_CHART_ROOT}"
echo "Log root: ${RUN_LOG_ROOT}"

train_and_eval "T1" "configs/trajectory/T1_trajectory_baseline_scene.json"
train_and_eval "T2" "configs/trajectory/T2_trajectory_augment_scene.json"
train_and_eval "T3" "configs/trajectory/T3_trajectory_two_stage_scene.json"
train_and_eval "T4" "configs/trajectory/T4_trajectory_task_heldout.json"
train_and_eval "T5" "configs/trajectory/T5_trajectory_21_keypoints_scene.json"

BEST_LINE="$(
  select_best_run \
    "T1" "${RUN_CONFIGS[T1]}" "${RUN_DIRS[T1]}" \
    "T2" "${RUN_CONFIGS[T2]}" "${RUN_DIRS[T2]}" \
    "T3" "${RUN_CONFIGS[T3]}" "${RUN_DIRS[T3]}" \
    "T4" "${RUN_CONFIGS[T4]}" "${RUN_DIRS[T4]}" \
    "T5" "${RUN_CONFIGS[T5]}" "${RUN_DIRS[T5]}"
)"
IFS=$'\t' read -r BEST_LABEL BEST_CONFIG BEST_RUN_DIR BEST_MRR <<< "${BEST_LINE}"

FINAL_RUNS_JSON="${OUTPUT_ROOT}/trajectory_final_runs.json"
cat > "${FINAL_RUNS_JSON}" <<EOF
{
  "T1": {
    "run_path": "${RUN_DIRS[T1]}",
    "label": "T1_Baseline"
  },
  "T2": {
    "run_path": "${RUN_DIRS[T2]}",
    "label": "T2_Augment"
  },
  "T3": {
    "run_path": "${RUN_DIRS[T3]}",
    "label": "T3_TwoStage"
  },
  "T4": {
    "run_path": "${RUN_DIRS[T4]}",
    "label": "T4_TaskHeld"
  },
  "T5": {
    "run_path": "${RUN_DIRS[T5]}",
    "label": "T5_21Keypoints"
  }
}
EOF

python runs/trajectory/export_final_trajectory_charts.py \
  --runs-json "${FINAL_RUNS_JSON}" \
  --output-dir "${FINAL_CHART_ROOT}" \
  2>&1 | tee "${RUN_LOG_ROOT}/final_charts.log"

cat > "${OUTPUT_ROOT}/trajectory_all_experiments_summary.json" <<EOF
{
  "best_label": "${BEST_LABEL}",
  "best_config": "${BEST_CONFIG}",
  "best_run_dir": "${BEST_RUN_DIR}",
  "best_h2r_task_mrr": ${BEST_MRR},
  "final_runs_json": "${FINAL_RUNS_JSON}",
  "final_chart_root": "${FINAL_CHART_ROOT}",
  "log_root": "${RUN_LOG_ROOT}"
}
EOF

echo "All trajectory experiments finished."
echo "Best: ${BEST_LABEL}, H2R task MRR=${BEST_MRR}, run=${BEST_RUN_DIR}"
echo "Summary: ${OUTPUT_ROOT}/trajectory_all_experiments_summary.json"
