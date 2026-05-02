# cfg2 模型评估 cfg3 全量测试集

更新时间：2026-05-01

本文档记录如何使用 `RH20T_cfg2` 训练得到的模型，在 `RH20T_cfg3` 上做单模态检索评估。这里的 cfg3 全部作为 test，不参与训练、验证或 checkpoint 选择。

## 1. 核心原则

- 使用 cfg2 run 目录下的 `params.json`，不要随手换原始 config。`params.json` 记录了 checkpoint 实际使用的模型结构和训练参数。
- 使用 `--data-root dataset/RH20T_subset/RH20T_cfg3` 覆盖数据路径。
- 使用 `--all-as-test` 将 cfg3 全部放入 test split。
- 不要传 `--split-manifest`。cfg2 run 目录下的 `split_manifest.json` 属于 cfg2，不能用于 cfg3。
- 不要同时使用 `--all-as-test` 和 `--split-manifest`。

## 2. 轨迹模型评估 cfg3

一键评估 `trajectory_final_runs.json` 中所有轨迹模型，并生成 cfg3 指标对比图：

```bash
bash scripts/evaluate_all_trajectory_cfg3.sh
```

默认读取：

```text
artifacts/runs/trajectory/trajectory_final_runs.json
```

默认输出：

```text
artifacts/runs/trajectory/<RUN>/cfg3_all_test/
artifacts/runs/trajectory/cfg3_final_charts/metrics_comparison.png
artifacts/runs/trajectory/cfg3_final_charts/metrics_comparison_data.json
```

常用变量：

```bash
RUNS_JSON=artifacts/runs/trajectory/trajectory_final_runs.json \
CFG3_DATA_ROOT=dataset/RH20T_subset/RH20T_cfg3 \
OUTPUT_SUBDIR=cfg3_all_test \
CHART_OUTPUT_DIR=artifacts/runs/trajectory/cfg3_final_charts \
bash scripts/evaluate_all_trajectory_cfg3.sh
```

推荐命令：

```bash
RUN=artifacts/runs/trajectory/<CFG2_RUN_NAME>

python runs/trajectory/evaluate_retrieval.py \
  --config "$RUN/params.json" \
  --checkpoint "$RUN/best_model.pth" \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --all-as-test \
  --split test \
  --output-dir "$RUN/cfg3_all_test"
```

如果要评估 21 keypoints 模型，checkpoint 对应的 `params.json` 必须满足：

```json
{
  "use_6_keypoints": false,
  "model_params": {
    "human_input_dim": 63
  }
}
```

可选配置模板：

- `configs/trajectory/T1_trajectory_cfg3_all_test_eval.json`：6 keypoints 轨迹模型。
- `configs/trajectory/T5_trajectory_cfg3_all_test_eval.json`：21 keypoints 轨迹模型。

只有当模板中的模型结构与 checkpoint 完全一致时，才建议使用模板：

```bash
python runs/trajectory/evaluate_retrieval.py \
  --config configs/trajectory/T1_trajectory_cfg3_all_test_eval.json \
  --checkpoint artifacts/runs/trajectory/<CFG2_RUN_NAME>/best_model.pth \
  --split test \
  --output-dir artifacts/runs/trajectory/<CFG2_RUN_NAME>/cfg3_all_test
```

## 3. 视频模型评估 cfg3

一键评估 `video_final_runs.json` 中所有视频模型，并生成 cfg3 指标对比图：

```bash
bash scripts/evaluate_all_video_cfg3.sh
```

默认读取：

```text
artifacts/runs/video/video_final_runs.json
```

默认输出：

```text
artifacts/runs/video/<RUN>/cfg3_all_test/
artifacts/runs/video/cfg3_final_charts/metrics_comparison.png
artifacts/runs/video/cfg3_final_charts/metrics_comparison_data.json
```

常用变量：

```bash
RUNS_JSON=artifacts/runs/video/video_final_runs.json \
CFG3_DATA_ROOT=dataset/RH20T_subset/RH20T_cfg3 \
OUTPUT_SUBDIR=cfg3_all_test \
CHART_OUTPUT_DIR=artifacts/runs/video/cfg3_final_charts \
bash scripts/evaluate_all_video_cfg3.sh
```

说明：

- 视频脚本会正常评估 trained runs。
- 如果 `video_final_runs.json` 中包含 raw backbone entries，脚本会自动调用 `runs/video/evaluate_raw_video_backbone.py`，同样输出到该 raw run 的 `cfg3_all_test/`。

推荐命令：

```bash
RUN=artifacts/runs/video/<CFG2_RUN_NAME>

python runs/video/evaluate_video.py \
  --config "$RUN/params.json" \
  --checkpoint "$RUN/best_model.pth" \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --all-as-test \
  --split test \
  --output-dir "$RUN/cfg3_all_test"
```

视频评估会继续使用 `$RUN/params.json` 中的采样配置，例如：

- `num_frames`
- `sampling_strategy`
- `max_pairs_per_scene`
- `eval_batch_size`
- backbone 与 encoder mode

这能保证 cfg2 test 和 cfg3 test 的模型与采样口径尽量一致。

可选配置模板：

- `configs/video/vjepa_rh20t_cfg3_all_test_eval.json`
- `configs/video/videomae_rh20t_cfg3_all_test_eval.json`

只有当模板中的 backbone 和 encoder 结构与 checkpoint 完全一致时，才建议使用模板：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_cfg3_all_test_eval.json \
  --checkpoint artifacts/runs/video/<CFG2_RUN_NAME>/best_model.pth \
  --split test \
  --output-dir artifacts/runs/video/<CFG2_RUN_NAME>/cfg3_all_test
```

## 4. 输出文件

轨迹和视频评估都会输出同类产物：

```text
artifacts/runs/<modality>/<CFG2_RUN_NAME>/cfg3_all_test/
  metrics.json
  cases.json
  metadata.json
  similarity_matrix.npy
  human_embeddings.npy
  robot_embeddings.npy
  similarity_heatmap.png
  task_scene_sorted_similarity_heatmap.png
  run_info.json
```

其中：

- `metrics.json`：双向 `human_to_robot` / `robot_to_human`，以及 `scene` / `task` 两级指标。
- `cases.json`：Top-K 检索案例。
- `metadata.json`：query/gallery 的 task、scene、camera、路径信息。
- `similarity_matrix.npy`：human embedding 与 robot embedding 的相似度矩阵。
- `run_info.json`：本次评估使用的数据根目录和 split 配置。

## 5. 结果解释

cfg2 -> cfg3 是跨配置迁移评估，不应直接等同于 cfg2 内部 test 指标。

报告结果时建议同时说明：

- cfg2 训练 run 名称。
- 使用的是 `best_model.pth` 还是 `last_model.pth`。
- cfg3 是否全量作为 test，本项目命令中为 `--all-as-test`。
- `valid_queries` 数量。
- task 数量与每个 task 的正样本数量分布。
- 主指标使用 `human_to_robot.task`，辅助报告 `robot_to_human.task`。

特别注意 `NDCG@10`：

- `R@K` 和 `MRR` 更关注是否至少命中一个正样本。
- `NDCG@10` 关注 top10 内多个正样本的整体排序质量。
- cfg3 全量 test 的 task 分布可能不同于 cfg2 test，因此解释 `NDCG@10` 时应同时查看 `metadata.json`。

## 6. 快速检查 cfg3 数据规模

轨迹：

```bash
python - <<'PY'
from bise.data import RH20TTrajectoryDataset
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset

dataset = RH20TTrajectoryDataset("dataset/RH20T_subset/RH20T_cfg3", use_6_keypoints=True)
splits = split_trajectory_dataset(dataset, {"all_as_test": True})
manifest = build_split_manifest(splits)
print({name: len(values) for name, values in manifest.items()})
PY
```

视频：

```bash
python - <<'PY'
from bise.data.rh20t.scanner import scan_task_scenes

tasks = scan_task_scenes("dataset/RH20T_subset/RH20T_cfg3")
scenes = sum(len(task) for task in tasks)
pairs = sum(len(scene.video_pairs) for task in tasks for scene in task)
print({"tasks": len(tasks), "scenes_with_video": scenes, "video_pairs": pairs})
PY
```
