#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV="${CONDA_ENV:-torch2}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/runs/video}"
FIGURE_ROOT="${FIGURE_ROOT:-${OUTPUT_ROOT}/figures}"
RUN_LOG_ROOT="${RUN_LOG_ROOT:-${OUTPUT_ROOT}/run_logs_$(date +%Y%m%d_%H%M%S)}"
TOP_K="${TOP_K:-5}"

# E6 reuses E1 as dual_head by default because it is the same baseline config.
# Set RUN_E6_DUAL_HEAD_DUPLICATE=1 to train another dual_head run for E6.
RUN_E6_DUAL_HEAD_DUPLICATE="${RUN_E6_DUAL_HEAD_DUPLICATE:-0}"

mkdir -p "${OUTPUT_ROOT}" "${FIGURE_ROOT}" "${RUN_LOG_ROOT}"

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
  python runs/video/train_video.py --config "${config}" 2>&1 | tee "${RUN_LOG_ROOT}/${label}_train.log"

  run_dir="$(latest_run_dir "${experiment_name}")"
  require_file "${run_dir}/best_model.pth"
  require_file "${run_dir}/split_manifest.json"

  echo "===== ${label}: evaluate ${run_dir} ====="
  python runs/video/evaluate_video.py \
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

build_figures() {
  local name="$1"
  shift
  echo "===== figures: ${name} ====="
  python runs/video/build_video_figures.py "$@" \
    --output-dir "${FIGURE_ROOT}/${name}" \
    2>&1 | tee "${RUN_LOG_ROOT}/figures_${name}.log"
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
echo "Figure root: ${FIGURE_ROOT}"
echo "Log root: ${RUN_LOG_ROOT}"

# Recommended order from VIDEO_ENCODER_TRAINING_EXPERIMENT_SCHEDULE.md.
train_and_eval "E1" "configs/video/vjepa_rh20t_baseline.json"
build_figures "E1" \
  --eval-dir "E1=${RUN_DIRS[E1]}/final_test" \
  --run-dir "E1=${RUN_DIRS[E1]}"

train_and_eval "E2" "configs/video/videomae_rh20t_baseline.json"
build_figures "backbone" \
  --eval-dir "E1=${RUN_DIRS[E1]}/final_test" \
  --eval-dir "E2=${RUN_DIRS[E2]}/final_test" \
  --run-dir "E1=${RUN_DIRS[E1]}" \
  --run-dir "E2=${RUN_DIRS[E2]}"

train_and_eval "E5" "configs/video/vjepa_rh20t_info_nce.json"
build_figures "loss" \
  --eval-dir "E1_multi_positive=${RUN_DIRS[E1]}/final_test" \
  --eval-dir "E5_info_nce=${RUN_DIRS[E5]}/final_test" \
  --run-dir "E1_multi_positive=${RUN_DIRS[E1]}" \
  --run-dir "E5_info_nce=${RUN_DIRS[E5]}"

train_and_eval "E4" "configs/video/vjepa_rh20t_intra.json"
build_figures "intra" \
  --eval-dir "E1_no_intra=${RUN_DIRS[E1]}/final_test" \
  --eval-dir "E4_intra=${RUN_DIRS[E4]}/final_test" \
  --run-dir "E1_no_intra=${RUN_DIRS[E1]}" \
  --run-dir "E4_intra=${RUN_DIRS[E4]}"

train_and_eval "E6_shared" "configs/video/vjepa_rh20t_shared.json"
if [[ "${RUN_E6_DUAL_HEAD_DUPLICATE}" == "1" ]]; then
  train_and_eval "E6_dual_head" "configs/video/vjepa_rh20t_baseline.json"
else
  RUN_DIRS["E6_dual_head"]="${RUN_DIRS[E1]}"
  RUN_CONFIGS["E6_dual_head"]="${RUN_CONFIGS[E1]}"
fi
train_and_eval "E6_dual_encoder" "configs/video/vjepa_rh20t_dual_encoder.json"
build_figures "encoder_mode" \
  --eval-dir "E6_shared=${RUN_DIRS[E6_shared]}/final_test" \
  --eval-dir "E6_dual_head=${RUN_DIRS[E6_dual_head]}/final_test" \
  --eval-dir "E6_dual_encoder=${RUN_DIRS[E6_dual_encoder]}/final_test"

train_and_eval "E3" "configs/video/vjepa_rh20t_task_heldout.json"
build_figures "split_generalization" \
  --eval-dir "E1_scene=${RUN_DIRS[E1]}/final_test" \
  --eval-dir "E3_task=${RUN_DIRS[E3]}/final_test"

build_figures "all_experiments" \
  --eval-dir "E1=${RUN_DIRS[E1]}/final_test" \
  --eval-dir "E2=${RUN_DIRS[E2]}/final_test" \
  --eval-dir "E3=${RUN_DIRS[E3]}/final_test" \
  --eval-dir "E4=${RUN_DIRS[E4]}/final_test" \
  --eval-dir "E5=${RUN_DIRS[E5]}/final_test" \
  --eval-dir "E6_shared=${RUN_DIRS[E6_shared]}/final_test" \
  --eval-dir "E6_dual_head=${RUN_DIRS[E6_dual_head]}/final_test" \
  --eval-dir "E6_dual_encoder=${RUN_DIRS[E6_dual_encoder]}/final_test"

BEST_LINE="$(
  select_best_run \
    "E1" "${RUN_CONFIGS[E1]}" "${RUN_DIRS[E1]}" \
    "E2" "${RUN_CONFIGS[E2]}" "${RUN_DIRS[E2]}" \
    "E3" "${RUN_CONFIGS[E3]}" "${RUN_DIRS[E3]}" \
    "E4" "${RUN_CONFIGS[E4]}" "${RUN_DIRS[E4]}" \
    "E5" "${RUN_CONFIGS[E5]}" "${RUN_DIRS[E5]}" \
    "E6_shared" "${RUN_CONFIGS[E6_shared]}" "${RUN_DIRS[E6_shared]}" \
    "E6_dual_head" "${RUN_CONFIGS[E6_dual_head]}" "${RUN_DIRS[E6_dual_head]}" \
    "E6_dual_encoder" "${RUN_CONFIGS[E6_dual_encoder]}" "${RUN_DIRS[E6_dual_encoder]}"
)"
IFS=$'\t' read -r BEST_LABEL BEST_CONFIG BEST_RUN_DIR BEST_MRR <<< "${BEST_LINE}"

echo "===== E7: best=${BEST_LABEL}, H2R task MRR=${BEST_MRR}, run=${BEST_RUN_DIR} ====="
python runs/video/evaluate_video.py \
  --config "${BEST_CONFIG}" \
  --checkpoint "${BEST_RUN_DIR}/best_model.pth" \
  --split test \
  --top-k "${TOP_K}" \
  --split-manifest "${BEST_RUN_DIR}/split_manifest.json" \
  --output-dir "${BEST_RUN_DIR}/final_test" \
  2>&1 | tee "${RUN_LOG_ROOT}/E7_eval.log"

python runs/video/export_video_embeddings.py \
  --config "${BEST_CONFIG}" \
  --checkpoint "${BEST_RUN_DIR}/best_model.pth" \
  --split test \
  --split-manifest "${BEST_RUN_DIR}/split_manifest.json" \
  --output "${BEST_RUN_DIR}/final_test/video_embeddings.json" \
  2>&1 | tee "${RUN_LOG_ROOT}/E7_export_embeddings.log"

build_figures "final" \
  --eval-dir "BEST=${BEST_RUN_DIR}/final_test" \
  --run-dir "BEST=${BEST_RUN_DIR}"

cat > "${OUTPUT_ROOT}/video_all_experiments_summary.json" <<EOF
{
  "best_label": "${BEST_LABEL}",
  "best_config": "${BEST_CONFIG}",
  "best_run_dir": "${BEST_RUN_DIR}",
  "best_h2r_task_mrr": ${BEST_MRR},
  "figure_root": "${FIGURE_ROOT}",
  "log_root": "${RUN_LOG_ROOT}"
}
EOF

echo "All video experiments finished."
echo "Summary: ${OUTPUT_ROOT}/video_all_experiments_summary.json"
