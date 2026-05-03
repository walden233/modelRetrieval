# 项目常用入口命令

更新时间：2026-05-03

本文档只整理常用入口。优先使用 `scripts/*.sh`，它们已经把常用路径和变量前置，适合复现实验和减少手写长命令。

## 0. 环境

```bash
cd /home/ttt/BISE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch2
```

语义 VLM 如需调用 API：

```bash
export ZHIPU_API_KEY="你的 key"
```

通用变量：

```bash
PYTHON_BIN=/home/ttt/miniconda3/envs/torch2/bin/python
```

## 1. 视频模态

### 1.1 训练单个视频实验

默认配置是 V-JEPA baseline：

```bash
scripts/train_video.sh
```

指定配置：

```bash
CONFIG=configs/video/vjepa_rh20t_intra.json scripts/train_video.sh
CONFIG=configs/video/videomae_rh20t_baseline.json scripts/train_video.sh
```

覆盖数据或输出目录：

```bash
DATA_ROOT=dataset/RH20T_subset/RH20T_cfg2 OUTPUT_DIR=artifacts/runs/video/my_run scripts/train_video.sh
```

### 1.2 一键跑视频实验组

```bash
scripts/run_all_video_experiments.sh
```

### 1.3 评估单个视频 run

默认评估 `RUN/final_test`：

```bash
scripts/evaluate_video.sh
```

指定 run：

```bash
RUN=artifacts/runs/video/video_vjepa_rh20t_baseline_20260429_145608 scripts/evaluate_video.sh
```

### 1.4 评估 final_runs 中所有视频模型

```bash
scripts/evaluate_all_video.sh
```

### 1.5 cfg3 all-test 视频评估

单个 run：

```bash
RUN=artifacts/runs/video/video_vjepa_rh20t_baseline_20260429_145608 scripts/cfg3_video_test.sh
```

final_runs 全部模型：

```bash
scripts/evaluate_all_video_cfg3.sh
```

### 1.6 视频最终图表

```bash
scripts/export_video_final_charts.sh
```

常用变量：

```bash
RUNS_JSON=artifacts/runs/video/video_final_runs.json \
OUTPUT_DIR=artifacts/runs/video/final_charts \
LEVEL=task \
scripts/export_video_final_charts.sh
```

## 2. 轨迹模态

### 2.1 训练单个轨迹实验

默认配置是 T1 baseline：

```bash
scripts/train_trajectory.sh
```

指定配置：

```bash
CONFIG=configs/trajectory/T2_trajectory_augment_scene.json scripts/train_trajectory.sh
CONFIG=configs/trajectory/T4_trajectory_task_heldout.json scripts/train_trajectory.sh
```

### 2.2 一键跑轨迹实验组

```bash
scripts/run_all_trajectory_experiments.sh
```

### 2.3 评估单个轨迹 run

```bash
scripts/evaluate_trajectory.sh
```

指定 run：

```bash
RUN=artifacts/runs/trajectory/T1_trajectory_baseline_scene_20260430_224542 scripts/evaluate_trajectory.sh
```

### 2.4 评估 final_runs 中所有轨迹模型

```bash
scripts/evaluate_all_trajectory.sh
```

### 2.5 cfg3 all-test 轨迹评估

单个 run：

```bash
RUN=artifacts/runs/trajectory/T1_trajectory_baseline_scene_20260430_224542 scripts/cfg3_trajectory_test.sh
```

final_runs 全部模型：

```bash
scripts/evaluate_all_trajectory_cfg3.sh
```

### 2.6 轨迹最终图表

```bash
scripts/export_trajectory_final_charts.sh
```

## 3. 语义模态

### 3.1 构建语义 manifest

```bash
python runs/semantic/build_semantic_manifest.py \
  --config configs/semantic/pipeline_v1.json \
  --dataset-type rh20t \
  --data-root dataset/RH20T_subset/RH20T_cfg2 \
  --scenes-per-task 2
```

### 3.2 运行语义标注

同步小批量调试：

```bash
python runs/semantic/run_semantic_annotation.py \
  --config configs/semantic/pipeline_v1.json \
  --start-index 0 \
  --end-index 8
```

正式运行：

```bash
python runs/semantic/run_semantic_annotation.py \
  --config configs/semantic/pipeline_v1.json
```

### 3.3 语义检索评估

```bash
scripts/evaluate_semantic_retrieval.sh
```

常用变量：

```bash
ANNOTATIONS=artifacts/semantic/rh20t/cfg2/annotations/normalized_annotations.jsonl \
POSITIVE_KEY=pair_id \
OUTPUT_DIR=artifacts/semantic/rh20t/cfg2/evaluation/pair \
scripts/evaluate_semantic_retrieval.sh
```

### 3.4 语义最终图表

```bash
scripts/export_semantic_final_charts.sh
```

### 3.5 构建语义 FAISS 索引

默认索引 `text_embedding`：

```bash
python runs/semantic/build_index.py \
  --features artifacts/semantic/rh20t/cfg2/feature_store/semantic_features.json \
  --output artifacts/semantic/rh20t/cfg2/index/text_embedding.faiss
```

## 4. 三模态统一检索系统

### 4.1 构建检索库

默认 cfg2：

```bash
scripts/build_retrieval_library.sh
```

指定 cfg3：

```bash
CONFIG=configs/retrieval/library_rh20t_cfg3.json scripts/build_retrieval_library.sh
```

常用覆盖：

```bash
CONFIG=configs/retrieval/library_rh20t_cfg2.json \
scripts/build_retrieval_library.sh \
  --scenes-per-task 0 \
  --cameras-per-scene 2
```

默认输出：

```text
artifacts/retrieval/rh20t_cfg2_v1
```

### 4.2 单条 query 检索

使用检索库内 human eval query：

```bash
scripts/query_retrieval_system.sh --query-id "<query_id>"
```

使用外部语义 embedding：

```bash
scripts/query_retrieval_system.sh \
  --semantic-feature query_semantic_feature.json \
  --top-k 10
```

使用外部 human video：

```bash
scripts/query_retrieval_system.sh \
  --video-path /path/to/human.mp4 \
  --video-config configs/video/vjepa_rh20t_baseline.json \
  --video-checkpoint artifacts/runs/video/video_vjepa_rh20t_baseline_20260429_145608/best_model.pth \
  --top-k 10
```

### 4.3 评估检索系统

默认评估三模态完整 query：

```bash
scripts/evaluate_retrieval_system.sh
```

只评估 video 输入：

```bash
CONFIG=configs/retrieval/ablation/video_only_uniform.json \
REQUIRE_MODALITIES=video \
OUTPUT_DIR=artifacts/retrieval/rh20t_cfg2_v1/eval/video_only_scene \
LEVEL=scene \
scripts/evaluate_retrieval_system.sh
```

### 4.4 检索系统消融实验

一键跑 Video / Trajectory / Semantic / 双模态 / 三模态均匀权重消融：

```bash
scripts/run_retrieval_system_ablation.sh
```

默认输出：

```text
artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform
```

### 4.5 检索系统图表

消融图表：

```bash
RUNS_JSON=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/ablation_runs.json \
OUTPUT_DIR=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/charts \
LEVEL=scene \
scripts/export_retrieval_system_charts.sh
```

## 5. 常见 final chart 汇总

视频：

```bash
scripts/export_video_final_charts.sh
```

轨迹：

```bash
scripts/export_trajectory_final_charts.sh
```

语义：

```bash
scripts/export_semantic_final_charts.sh
```

检索系统：

```bash
scripts/export_retrieval_system_charts.sh
```

从 final chart 的 JSON 数据恢复 PNG：

```bash
python runs/restore_charts_from_json.py --root artifacts --overwrite
```

只恢复指定 JSON 到指定目录：

```bash
python runs/restore_charts_from_json.py \
  --input artifacts/runs/video/final_charts/video_metrics_comparison_data.json \
  --output-dir /tmp/restored_charts \
  --overwrite
```

## 6. 常见产物位置

视频：

```text
artifacts/runs/video/<run>/
artifacts/runs/video/<run>/final_test/
artifacts/runs/video/final_charts/
```

轨迹：

```text
artifacts/runs/trajectory/<run>/
artifacts/runs/trajectory/<run>/final_test/
artifacts/runs/trajectory/final_charts/
```

语义：

```text
artifacts/semantic/rh20t/cfg2/
artifacts/semantic/rh20t/cfg2/evaluation/pair/
artifacts/semantic/rh20t/cfg2/final_charts/
```

统一检索：

```text
artifacts/retrieval/rh20t_cfg2_v1/
artifacts/retrieval/rh20t_cfg2_v1/eval/
artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/
```

## 7. 文档索引

视频实验：

```text
docs/video/VIDEO_ENCODER_TRAINING_EXPERIMENT_SCHEDULE.md
```

轨迹实验：

```text
docs/trajectory/TRAJECTORY_ENCODER_TRAINING_EXPERIMENT_SCHEDULE.md
```

语义实验：

```text
docs/semantic/SEMANTIC_SINGLE_MODAL_EXPERIMENT_SCHEDULE.md
```

统一检索系统：

```text
docs/multimodal/UNIFIED_RETRIEVAL_SYSTEM_PLAN.md
docs/multimodal/RETRIEVAL_SYSTEM_ABLATION_EXPERIMENTS.md
```
