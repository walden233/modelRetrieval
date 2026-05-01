# 单轨迹编码器训练与实验计划

更新时间：2026-04-30

本文档面向当前 `configs/trajectory`、`src/bise/modalities/trajectory/models`、`runs/trajectory` 代码，目标是检查轨迹单模态模块的硬伤与不足，并参考视频模态的实验组织方式，给出只做必要实验的训练与评估计划。本文只覆盖轨迹模态，不涉及视频、语义或多模态融合。

## 1. 当前代码检查结论

当前轨迹代码可以运行基础训练。2026-04-30 已补齐 split manifest 与评估产物闭环；后续正式实验应统一走新的 `split_manifest.json` 和 `final_test/` 产物。

### 1.1 已具备能力

- `CrossModalTrajectoryModel` 已实现 human trajectory encoder、robot TCP encoder、共享 projection head、跨域相似度温度参数。
- `trajectory_symmetric_contrastive_loss` 支持 scene/task 多正样本软标签。
- `train_trajectory.py` 支持三种训练模式：`standard`、`augment`、`two_stage`。
- `trainer.py` 支持轨迹旋转/噪声增强、intra-modal consistency、two-stage pretrain + finetune。
- `evaluate_retrieval_grouped` 支持把同 scene/task 的多条 camera 轨迹聚合成 label-level 检索指标。
- `save_run_artifacts` 会保存 `params.json`、`best_metrics.json`、`curves.json`、`curves.png`。

### 1.2 已修复的关键硬伤

1. **训练/评估 split 不可复现且不一致。**

   已修复：`train_trajectory.py` 现在按 scene/task split 生成 train/val/test，并保存 `split_manifest.json`。`evaluate_retrieval.py` 支持 `--split`、`--split-manifest` 和 checkpoint 同目录自动读取 manifest。

2. **没有 scene/task split 语义。**

   已修复：新增轨迹 split 工具，支持 `scene` 与 `task` 两种粒度。轨迹数据的样本单位是 scene，RH20T cfg2 当前为 434 scenes / 109 tasks，每个 task 约 3-4 scenes。正式实验至少需要：

   - `scene split`：测试同 task 下未见 scene 的跨域检索。
   - `task-held-out split`：测试训练未见 task 的泛化。

3. **配置存在历史实验混杂。**

   `baseline.json` 与 `baseline_fair.json` 基本重复；`augment.json` 与 `augment_fair.json` 的评估语义不同；`two_stage.json` 使用 `use_6_keypoints=false` 和 `human_input_dim=63`，与 fair configs 的 6 keypoints 设置不在同一比较条件下。正式实验应以 `_fair` 系列或重新整理后的 config 为主。

4. **评估指标结构弱于视频模态。**

   已修复：新增 `evaluate_trajectory_retrieval`，输出双向 scene/task 指标、NDCG@10、valid queries、similarity matrix、embeddings 和 metadata。旧 `evaluate_retrieval_grouped` 保留兼容。

   - robot-to-human 指标。
   - scene/task 两套指标同时输出。
   - NDCG@10。
   - valid queries。
   - similarity matrix / embeddings / cases。

5. **最终评估入口产物不足。**

   已修复：`runs/trajectory/evaluate_retrieval.py` 可输出 `metrics.json`、`cases.json`、`metadata.json`、`similarity_matrix.npy`、`human_embeddings.npy`、`robot_embeddings.npy`、`similarity_heatmap.png` 和 `task_scene_sorted_similarity_heatmap.png`。

6. **曲线命名和内容还不够严谨。**

   `standard` 模式下 `history` 包含 `train_loss_inter/train_loss_intra` 但不追加值，通常不影响画图；但 two-stage 模式把 pretrain 阶段和 finetune 阶段写入同一 `train_loss` 序列，后续图表解释需要明确阶段边界。

7. **增强随机性没有统一 seed 控制。**

   `augmentations.py` 使用 `torch.rand` / `torch.randn_like`，训练复现依赖全局随机状态。当前 `train_trajectory.py` 只给 split generator 设 seed，没有统一设置 `torch.manual_seed`、`numpy seed` 和 dataloader worker seed。

8. **Dataset 容错方式可能隐藏坏样本。**

   `RH20TTrajectoryDataset.__getitem__` 在读取失败或空轨迹时递归返回下一个样本。这会让某个 index 实际对应另一个 scene，split 和指标解释会变得不严格。正式实验前建议在 dataset 初始化阶段过滤坏样本，而不是 `__getitem__` 时替换。

## 2. 数据与任务定义

当前正式数据：

| 数据集 | 路径 | scene 数 | task 数 | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| RH20T cfg2 trajectory | `dataset/RH20T_subset/RH20T_cfg2` | 434 | 109 | 单轨迹正式实验 |

轨迹样本单位：

- Dataset item 是一个 scene。
- 一个 scene 内包含多个 camera 的 human hand trajectory 和 robot TCP trajectory。
- collate 后，一个 batch 内会展开成多条 human trajectories 和多条 robot trajectories。

主评估目标：

- Human trajectory query 检索 Robot TCP candidate。
- 默认主指标：`human_to_robot.task.MRR`。
- 辅助看 scene-level 指标，用于判断模型是否只学到局部 scene 对齐。

正式指标建议与视频模态对齐：

- `R@1`
- `R@5`
- `R@10`
- `Mean Rank`
- `MRR`
- `Mean Percentage Rank`
- `NDCG@10`
- `valid_queries`

建议同时输出：

- `human_to_robot.scene`
- `human_to_robot.task`
- `robot_to_human.scene`
- `robot_to_human.task`

## 3. 训练前必须补齐的工程项

这些是正式实验前的必要工作。当前已在代码中落地，正式实验应直接使用本节定义的入口。

### P0.1 固定 split manifest

轨迹 split 工具已对齐视频模态：

- 支持 `split.unit = scene | task`。
- 支持 `split.seed` 与 `split.ratios`。
- 支持 `split.manifest_path`。
- 训练时保存 `split_manifest.json`。
- 评估时显式或自动读取 checkpoint 同目录的 `split_manifest.json`。

建议 manifest 格式：

```json
{
  "train": ["task_0001/scene_1", "..."],
  "val": ["task_0002/scene_3", "..."],
  "test": ["task_0008/scene_2", "..."]
}
```

### P0.2 改造训练入口

`runs/trajectory/train_trajectory.py` 已补齐：

- `run_dir` 在 split 后保存 `split_manifest.json`。
- 使用 `train / val / test` 三份 split，但训练阶段只用 train/val。
- best checkpoint 仍由 val 主指标选择。
- 不在训练脚本里自动写 test 产物，正式 test 由 evaluate 脚本统一输出。
- 设置全局随机种子。

### P0.3 改造评估入口

`runs/trajectory/evaluate_retrieval.py` 已补齐：

- `--split train|val|test`
- `--split-manifest`
- `--output-dir`
- 输出 `final_test/metrics.json`
- 输出 `final_test/cases.json`
- 输出 `final_test/similarity_matrix.npy`
- 输出 `final_test/human_embeddings.npy`
- 输出 `final_test/robot_embeddings.npy`
- 输出 `final_test/metadata.json`
- 输出 `final_test/similarity_heatmap.png`
- 输出 `final_test/task_scene_sorted_similarity_heatmap.png`

### P0.4 对齐评估指标

已新增一个 trajectory evaluator，接口接近视频模态的 `evaluate_video_retrieval`：

```python
evaluate_trajectory_retrieval(model, dataloader, device) -> {
  "metrics": {
    "human_to_robot": {"scene": ..., "task": ...},
    "robot_to_human": {"scene": ..., "task": ...}
  },
  "similarity_matrix": ...,
  "human_embeddings": ...,
  "robot_embeddings": ...,
  "metadata": ...
}
```

### P0.5 最终图表脚本

参考视频的 `export_final_video_charts.py`，已新增：

```bash
python runs/trajectory/export_final_trajectory_charts.py \
  --runs-json artifacts/runs/trajectory/trajectory_final_runs.json \
  --output-dir artifacts/runs/trajectory/final_charts
```

输出：

- `trajectory_curves_comparison.png`
- `trajectory_metrics_comparison.png`
- `trajectory_curves_comparison_data.json`
- `trajectory_metrics_comparison_data.json`

说明：

- `configs/trajectory/trajectory_final_runs.json` 是手工汇总时的模板。
- `runs/trajectory/run_all_trajectory_experiments.sh` 会在实验完成后自动生成实际可用的 `artifacts/runs/trajectory/trajectory_final_runs.json`。

## 4. 正式实验设计

原则：只做必要、有明确解释价值、能产出图表的实验。不做大规模超参搜索。

### T1：Baseline 主实验

目的：

- 建立轨迹单模态主结果。
- 使用当前最合理的 fair setting：6 keypoints、task-positive training、scene split。

建议配置：

- 基于 `configs/trajectory/baseline_fair.json`。
- 新建正式配置：`configs/trajectory/T1_trajectory_baseline_scene.json`。
- `experiment_name = T1_trajectory_baseline_scene`
- `split.unit = scene`
- `evaluate_task_positives = false` 可以保留，但最终 evaluator 应同时输出 scene/task。

训练：

```bash
python runs/trajectory/train_trajectory.py \
  --config configs/trajectory/T1_trajectory_baseline_scene.json
```

最终评估：

```bash
python runs/trajectory/evaluate_retrieval.py \
  --config configs/trajectory/T1_trajectory_baseline_scene.json \
  --checkpoint artifacts/runs/trajectory/<T1_RUN>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/trajectory/<T1_RUN>/split_manifest.json \
  --output-dir artifacts/runs/trajectory/<T1_RUN>/final_test
```

图表产出：

- `curves.png`
- `curves.json`
- `final_test/metrics.json`
- `final_test/similarity_heatmap.png`
- `final_test/task_scene_sorted_similarity_heatmap.png`

### T2：Augment / intra loss 对照

目的：

- 判断轨迹旋转/噪声增强和 intra-modal consistency 是否有效。
- 对照 T1。

建议配置：

- 基于 `configs/trajectory/augment_fair.json`。
- 新建：`configs/trajectory/T2_trajectory_augment_scene.json`。
- 保持与 T1 相同 split seed 和 ratios。
- `intra_loss_weight = 0.3`。
- `intra_task_positive_weight = 0.25`。

训练与评估同 T1。

图表产出：

- T1 vs T2 指标对照柱状图。
- T1 vs T2 train loss / val MRR 曲线。
- 若 T2 提升 task MRR，说明增强一致性对跨任务语义轨迹有帮助。

### T3：Two-stage 对照

目的：

- 判断先做 intra-modal pretrain 再做 cross-domain finetune 是否优于直接训练。
- 对照 T1/T2。

建议配置：

- 基于 `configs/trajectory/two_stage_fair.json`。
- 新建：`configs/trajectory/T3_trajectory_two_stage_scene.json`。
- 保持 `use_6_keypoints=true`。
- 不使用旧 `two_stage.json` 作为正式对照，因为它使用 21 keypoints，和 fair setting 不可比。

图表产出：

- T1/T2/T3 指标对照柱状图。
- Two-stage 曲线图，标注 pretrain 与 finetune 阶段。

### T4：Task-held-out 泛化

目的：

- 测试训练未见 task 的泛化能力。
- 对照 T1 的 scene split。

建议配置：

- 基于 T1。
- 新建：`configs/trajectory/T4_trajectory_task_heldout.json`。
- `split.unit = task`。

图表产出：

- T1 scene split vs T4 task split 指标对照。
- 失败案例表：从 `cases.json` 中筛选高分错误样本。

解释重点：

- 如果 T4 明显低于 T1，说明轨迹模型更依赖见过 task 的运动模式，对未见 task 泛化有限。

### T5：Keypoint 输入维度对照

目的：

- 判断 6 keypoints 是否足够，还是完整 21 keypoints 更有价值。
- 这是唯一建议保留的输入表示对照。

建议配置：

- 基于 T1。
- 新建：`configs/trajectory/T5_trajectory_21_keypoints_scene.json`。
- `use_6_keypoints = false`
- `human_input_dim = 63`
- 其他训练配置与 T1 对齐。

图表产出：

- T1 6-keypoints vs T5 21-keypoints 指标对照。
- 参数/训练耗时记录。

是否必要：

- 如果时间紧，T5 优先级低于 T1-T4。
- 如果论文需要解释为什么用 6 keypoints，则 T5 必做。

## 5. 实验总表

| 编号 | 实验 | 核心变量 | 目的 | 必要性 |
| :--- | :--- | :--- | :--- | :--- |
| T1 | Baseline scene split | standard + 6 keypoints + scene split | 主结果 | 必做 |
| T2 | Augment / intra | 加旋转/噪声增强和 intra loss | 判断增强是否有收益 | 必做 |
| T3 | Two-stage | intra pretrain + cross-domain finetune | 判断两阶段是否必要 | 必做 |
| T4 | Task-held-out | split 从 scene 改为 task | 测试未见任务泛化 | 必做 |
| T5 | 21 keypoints | 6 keypoints vs 21 keypoints | 判断输入表示是否影响结果 | 可选 |

## 6. 推荐执行顺序

| 顺序 | 实验 | 原因 |
| :--- | :--- | :--- |
| 1 | T1 | 建立可复现主基线。 |
| 2 | T2 | 验证增强和 intra loss，直接影响最终训练策略。 |
| 3 | T3 | 验证 two-stage 是否值得保留。 |
| 4 | T4 | 补充泛化分析。 |
| 5 | T5 | 如需要解释 keypoint 选择再执行。 |

### 6.1 一键顺序运行

推荐直接执行：

```bash
bash runs/trajectory/run_all_trajectory_experiments.sh
```

默认行为：

- 自动激活 `torch2` 环境；如已手工激活环境，可设置 `SKIP_CONDA=1`。
- 按 T1 -> T2 -> T3 -> T4 -> T5 顺序训练。
- 每个实验训练结束后立即用本 run 的 `split_manifest.json` 做 test 评估。
- 每个 run 输出 `final_test/metrics.json`、`cases.json`、embedding、similarity matrix 和两张 heatmap。
- 自动生成 `artifacts/runs/trajectory/trajectory_final_runs.json`。
- 自动生成 `artifacts/runs/trajectory/final_charts/trajectory_curves_comparison.png` 和 `trajectory_metrics_comparison.png`。

可选参数：

```bash
CONDA_ENV=torch2 \
OUTPUT_ROOT=artifacts/runs/trajectory \
TOP_K=5 \
bash runs/trajectory/run_all_trajectory_experiments.sh
```

如果需要单独运行某个实验，使用：

```bash
python runs/trajectory/train_trajectory.py \
  --config configs/trajectory/T1_trajectory_baseline_scene.json

python runs/trajectory/evaluate_retrieval.py \
  --config configs/trajectory/T1_trajectory_baseline_scene.json \
  --checkpoint artifacts/runs/trajectory/<T1_RUN>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/trajectory/<T1_RUN>/split_manifest.json \
  --output-dir artifacts/runs/trajectory/<T1_RUN>/final_test
```

T2-T5 只需替换 config 和 run 目录：

- `configs/trajectory/T2_trajectory_augment_scene.json`
- `configs/trajectory/T3_trajectory_two_stage_scene.json`
- `configs/trajectory/T4_trajectory_task_heldout.json`
- `configs/trajectory/T5_trajectory_21_keypoints_scene.json`

## 7. 最终结果呈现

最终报告建议包含：

- 主结果表：T1-T4 的 H2R/R2H task-level 指标。
- 消融表：T1 vs T2、T1 vs T3、T1 vs T4、T1 vs T5。
- 曲线图：`trajectory_curves_comparison.png`。
- 指标图：`trajectory_metrics_comparison.png`。
- 相似度热图：T1 最佳模型的 `task_scene_sorted_similarity_heatmap.png`。
- Top-K 成功/失败案例：从 `cases.json` 中各选 3-5 个。

建议最终文件结构：

```text
artifacts/runs/trajectory/<RUN>/
  best_model.pth
  last_model.pth
  params.json
  best_metrics.json
  curves.json
  curves.png
  split_manifest.json
  final_test/
    metrics.json
    cases.json
    metadata.json
    similarity_matrix.npy
    human_embeddings.npy
    robot_embeddings.npy
    similarity_heatmap.png
    task_scene_sorted_similarity_heatmap.png

artifacts/runs/trajectory/final_charts/
  trajectory_curves_comparison.png
  trajectory_curves_comparison_data.json
  trajectory_metrics_comparison.png
  trajectory_metrics_comparison_data.json
```

## 8. 当前不建议扩展的实验

- 不做大规模 learning rate / batch size 搜索。
- 不做更多 keypoint 子集搜索。
- 不做 memory bank，除非后续证明 batch 内正负样本不足严重影响指标。
- 不做 cfg2 -> cfg3 迁移，除非单数据集结果已经稳定。
- 不做和视频/语义/多模态融合相关的实验。

## 9. 结论

轨迹模态目前的模型、split 和评估闭环已经具备正式实验入口。执行前重点检查：

1. 是否使用 T1-T5 正式配置，而不是历史 `_fair` 配置。
2. 每次 test 是否显式传入对应 run 的 `split_manifest.json`。
3. `final_test/metrics.json` 是否包含双向 scene/task 指标。
4. 最终图表是否同时保存 PNG 和对应 data JSON。

按 `run_all_trajectory_experiments.sh` 执行 T1-T5 后，即可形成一组足够支撑论文分析的单轨迹实验。
