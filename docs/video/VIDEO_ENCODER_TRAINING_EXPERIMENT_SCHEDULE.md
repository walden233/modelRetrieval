# 视频编码器训练与实验安排表

更新时间：2026-04-29

本文档面向当前 `src/bise/modalities/video` 与 `runs/video` 代码，目标是整理视频单模态训练与实验计划。`WHIRL debug` 已完成链路验证，因此不再列入正式实验。下面实验仅覆盖视频编码器，不涉及轨迹、语义或多模态融合。

## 1. 当前代码检查结论

当前视频代码没有发现阻断训练的硬伤，已确认以下关键点：

- `RH20TVideoDataset` 已修正 scene 标签碰撞问题，`scene_id = task_id/scene_name`。
- `VideoMAEAdapter` 已修正输入维度问题，兼容 `[B,T,C,H,W]` 与 `[B,C,T,H,W]`。
- `runs/video/train_video.py` 可按配置训练并保存 `best_model.pth`、`last_model.pth`、`params.json`、`best_metrics.json`、`curves.json`、`curves.png`、`split_manifest.json`。
- `curves.png` 由 `curves.json` 生成；未启用 intra loss 时，不绘制 `train_loss_inter` 和 `train_loss_intra`。
- `runs/video/evaluate_video.py` 支持 `--split-manifest` 与 `--output-dir`，可保存 `metrics.json`、`cases.json`、`similarity_matrix.npy`、`human_embeddings.npy`、`robot_embeddings.npy`、`run_info.json`。
- `runs/video/export_video_embeddings.py` 支持 `--split-manifest`，可导出 embedding 与检索案例。
- `runs/video/build_video_figures.py` 统一读取评估目录和训练目录，生成热图、指标柱状图、跨实验汇总表和曲线对比图。
- `torch2` 下全量测试已通过。

训练时仍需注意：

- 对比学习依赖真实 batch 内负样本数量，`gradient_accumulation_steps` 不能替代真实 batch size。
- `dual_encoder` 显存开销明显更高，已单独把 batch size 配成 `2`。
- 当前未加入 memory bank 和 task-balanced sampler，若 task-level 指标低，后续再考虑补。
- 正式训练前需确认 HuggingFace 模型已缓存或当前环境可联网加载。
- 训练完成后必须使用该 run 目录下的 `split_manifest.json` 做最终评估和 embedding 导出；评估脚本会默认从 checkpoint 同目录自动读取，但正式命令中仍显式传入，避免误用随机重切分。

## 2. 数据与主指标

当前正式实验数据使用：

| 数据集 | 配置 | 样本数 | task 数 | scene 数 | 用途 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RH20T cfg2 | `dataset/RH20T_subset/RH20T_cfg2` | 868 | 109 | 434 | 全部正式实验 |

主指标统一看：

- `human_to_robot.task.R@1`
- `human_to_robot.task.R@5`
- `human_to_robot.task.R@10`
- `human_to_robot.task.MRR`
- `human_to_robot.task.NDCG@10`
- `robot_to_human.task.R@1`
- `robot_to_human.task.R@10`
- `robot_to_human.task.MRR`

辅助指标：

- `human_to_robot.scene.R@1`
- `robot_to_human.scene.R@1`
- `Mean Rank`
- `Mean Percentage Rank`

模型选择主指标：

```text
human_to_robot.task.MRR
```

## 3. 实验总表

| 编号 | 实验 | 配置文件 | 核心变量 | 目的 | 主要产物 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| E1 | RH20T V-JEPA 主实验 | `configs/video/vjepa_rh20t_baseline.json` | V-JEPA + dual_head + multi_positive + scene split | 得到视频单模态主结果 | 主结果表、曲线、heatmap、Top-K cases |
| E2 | RH20T VideoMAE 对照 | `configs/video/videomae_rh20t_baseline.json` | backbone 换成 VideoMAE | 比较不同预训练视频表征 | backbone 对照表、柱状图 |
| E3 | RH20T V-JEPA task-held-out 泛化 | `configs/video/vjepa_rh20t_task_heldout.json` | split 从 scene 换成 task | 测试未见任务泛化 | split 泛化表、失败案例 |
| E4 | Intra loss 对照 | `configs/video/vjepa_rh20t_intra.json` | 增加同视频增强一致性损失 | 判断模态内增强一致性是否有收益 | intra loss 对照表、loss 曲线 |
| E5 | InfoNCE vs multi-positive | `configs/video/vjepa_rh20t_info_nce.json` | loss 从 multi_positive 换成 info_nce | 判断多正样本损失是否必要 | loss 对照表、柱状图 |
| E6 | shared / dual_head / dual_encoder | `configs/video/vjepa_rh20t_shared.json`、`configs/video/vjepa_rh20t_baseline.json`、`configs/video/vjepa_rh20t_dual_encoder.json` | encoder_mode | 判断跨域 head 设计是否必要 | encoder mode 对照表 |
| E7 | 最佳模型最终导出 | 使用 E1-E6 最优 run | 不训练 | 固化最终指标、embedding 和案例 | final metrics、embeddings、cases、降维图 |

## 4. 实验细化

### E1：RH20T V-JEPA 主实验

目的：

- 作为视频编码器主结果。
- 使用 `scene split`，避免同一 scene 泄漏到 train/test。
- 默认模型设定为 `V-JEPA + dual_head + multi_positive`。

训练：

```bash
conda activate torch2
python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_baseline.json
```

最终评估：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_baseline.json \
  --checkpoint artifacts/runs/video/<E1_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E1_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E1_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir E1=artifacts/runs/video/<E1_RUN_NAME>/final_test \
  --run-dir E1=artifacts/runs/video/<E1_RUN_NAME> \
  --output-dir artifacts/runs/video/figures/E1
```

图表产出：

- `curves.json`
- `curves.png`
- `final_test/metrics.json`
- `final_test/similarity_matrix.npy`
- `final_test/cases.json`
- 主结果表：`figures/E1/video_metrics_summary.csv`
- 指标柱状图：`figures/E1/E1_task_metrics.png`
- 相似度热图：`figures/E1/E1_similarity_heatmap.png`
- Top-K 检索案例表：`figures/E1/video_cases_summary.csv`

### E2：RH20T VideoMAE backbone 对照

目的：

- 比较 `VideoMAE` 与 `V-JEPA` 的预训练视频表征差异。
- 两者下游训练目标相同，区别是 backbone 与 processor。

训练：

```bash
conda activate torch2
python runs/video/train_video.py \
  --config configs/video/videomae_rh20t_baseline.json
```

最终评估：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/videomae_rh20t_baseline.json \
  --checkpoint artifacts/runs/video/<E2_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E2_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E2_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir E1=artifacts/runs/video/<E1_RUN_NAME>/final_test \
  --eval-dir E2=artifacts/runs/video/<E2_RUN_NAME>/final_test \
  --run-dir E1=artifacts/runs/video/<E1_RUN_NAME> \
  --run-dir E2=artifacts/runs/video/<E2_RUN_NAME> \
  --output-dir artifacts/runs/video/figures/backbone
```

对比对象：

- E1：`V-JEPA`
- E2：`VideoMAE`

图表产出：

- Backbone 对照表：`figures/backbone/video_metrics_summary.csv`
- Backbone 指标柱状图：`figures/backbone/video_metrics_comparison.png`
- 对比字段：`R@1 / R@5 / R@10 / MRR / NDCG@10`

### E3：RH20T V-JEPA task-held-out 泛化

目的：

- 将 split 粒度从 `scene` 改成 `task`。
- 测试训练中完全未见过的 task 是否还能被正确检索。

训练：

```bash
conda activate torch2
python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_task_heldout.json
```

最终评估：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_task_heldout.json \
  --checkpoint artifacts/runs/video/<E3_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E3_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E3_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir E1_scene=artifacts/runs/video/<E1_RUN_NAME>/final_test \
  --eval-dir E3_task=artifacts/runs/video/<E3_RUN_NAME>/final_test \
  --output-dir artifacts/runs/video/figures/split_generalization
```

对比对象：

- E1：`scene split`
- E3：`task split`

图表产出：

- 泛化对照表：`figures/split_generalization/video_metrics_summary.csv`
- 泛化指标柱状图：`figures/split_generalization/video_metrics_comparison.png`
- 失败案例表：`figures/split_generalization/video_cases_summary.csv` 中筛选 `top1_is_positive=false` 的高分样本。

解释重点：

- 若 E3 明显低于 E1，说明模型对已见 task 下的跨域对齐更强，对未见 task 的抽象泛化仍有限。

### E4：Intra loss 对照实验

目的：

- 在跨域对比损失之外，引入同一视频两个增强视图的一致性损失。
- 判断 `L = L_inter + lambda_intra * L_intra` 是否带来收益。

训练：

```bash
conda activate torch2
python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_intra.json
```

最终评估：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_intra.json \
  --checkpoint artifacts/runs/video/<E4_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E4_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E4_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir E1_no_intra=artifacts/runs/video/<E1_RUN_NAME>/final_test \
  --eval-dir E4_intra=artifacts/runs/video/<E4_RUN_NAME>/final_test \
  --run-dir E1_no_intra=artifacts/runs/video/<E1_RUN_NAME> \
  --run-dir E4_intra=artifacts/runs/video/<E4_RUN_NAME> \
  --output-dir artifacts/runs/video/figures/intra
```

对比对象：

- E1：`lambda_intra = 0.0`
- E4：`lambda_intra = 0.2`

图表产出：

- Intra loss 对照表：`figures/intra/video_metrics_summary.csv`
- Loss 曲线图：`figures/intra/video_curves_comparison.png`
- 指标柱状图：`figures/intra/video_metrics_comparison.png`
- 注意：E1 未启用 intra loss，其 `curves.png` 和 `curves.json` 不包含有效的 `train_loss_inter/train_loss_intra` 曲线点。

解释重点：

- 如果 task-level MRR 提升，说明视频增强一致性有助于稳定表征。
- 如果 scene-level 提升但 task-level 不升，可能只是增强了近场景鲁棒性。

### E5：InfoNCE vs multi-positive

目的：

- 验证多正样本损失是否比一对一 InfoNCE 更适合 RH20T 的 task-level 检索。

训练：

```bash
conda activate torch2
python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_info_nce.json
```

最终评估：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_info_nce.json \
  --checkpoint artifacts/runs/video/<E5_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E5_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E5_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir E1_multi_positive=artifacts/runs/video/<E1_RUN_NAME>/final_test \
  --eval-dir E5_info_nce=artifacts/runs/video/<E5_RUN_NAME>/final_test \
  --run-dir E1_multi_positive=artifacts/runs/video/<E1_RUN_NAME> \
  --run-dir E5_info_nce=artifacts/runs/video/<E5_RUN_NAME> \
  --output-dir artifacts/runs/video/figures/loss
```

对比对象：

- E1：`multi_positive_video_contrastive_loss`
- E5：`InfoNCELoss`

图表产出：

- Loss 对照表：`figures/loss/video_metrics_summary.csv`
- Loss 对照柱状图：`figures/loss/video_metrics_comparison.png`
- 训练曲线对比：`figures/loss/video_curves_comparison.png`

解释重点：

- 若 multi-positive 优于 InfoNCE，说明“同 task 多正样本”对跨域任务检索有意义。
- 若 InfoNCE 接近或更好，需要检查 batch 内同 task 样本数量是否太少，导致 multi-positive 没有充分发挥。

### E6：shared / dual_head / dual_encoder

目的：

- 比较三种跨领域视频编码结构。
- 判断是否需要 human / robot 域适配头，以及是否值得使用双 backbone。

训练命令：

```bash
conda activate torch2
python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_shared.json

python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_baseline.json

python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_dual_encoder.json
```

最终评估：

```bash
python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_shared.json \
  --checkpoint artifacts/runs/video/<E6_SHARED_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E6_SHARED_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E6_SHARED_RUN_NAME>/final_test

python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_baseline.json \
  --checkpoint artifacts/runs/video/<E1_OR_E6_DUAL_HEAD_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E1_OR_E6_DUAL_HEAD_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E1_OR_E6_DUAL_HEAD_RUN_NAME>/final_test

python runs/video/evaluate_video.py \
  --config configs/video/vjepa_rh20t_dual_encoder.json \
  --checkpoint artifacts/runs/video/<E6_DUAL_ENCODER_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<E6_DUAL_ENCODER_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<E6_DUAL_ENCODER_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir E6_shared=artifacts/runs/video/<E6_SHARED_RUN_NAME>/final_test \
  --eval-dir E6_dual_head=artifacts/runs/video/<E1_OR_E6_DUAL_HEAD_RUN_NAME>/final_test \
  --eval-dir E6_dual_encoder=artifacts/runs/video/<E6_DUAL_ENCODER_RUN_NAME>/final_test \
  --output-dir artifacts/runs/video/figures/encoder_mode
```

对比对象：

- `shared`
- `dual_head`
- `dual_encoder`

图表产出：

- Encoder mode 对照表：`figures/encoder_mode/video_metrics_summary.csv`
- Encoder mode 柱状图：`figures/encoder_mode/video_metrics_comparison.png`
- 显存/耗时记录表：`video_encoder_mode_cost.csv`

解释重点：

- `dual_head` 若优于 `shared`，说明 domain-specific adapter 有价值。
- `dual_encoder` 若收益不明显，则不建议作为最终模型，因为参数量和显存成本更高。

### E7：最佳模型最终评估与导出

目的：

- 从 E1-E6 中选出最佳视频单模态模型。
- 固化最终指标、embedding、相似度矩阵和案例。

最终评估：

```bash
conda activate torch2
python runs/video/evaluate_video.py \
  --config configs/video/<BEST_CONFIG>.json \
  --checkpoint artifacts/runs/video/<BEST_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<BEST_RUN_NAME>/split_manifest.json \
  --output-dir artifacts/runs/video/<BEST_RUN_NAME>/final_test

python runs/video/build_video_figures.py \
  --eval-dir BEST=artifacts/runs/video/<BEST_RUN_NAME>/final_test \
  --run-dir BEST=artifacts/runs/video/<BEST_RUN_NAME> \
  --output-dir artifacts/runs/video/figures/final
```

导出 embedding：

```bash
python runs/video/export_video_embeddings.py \
  --config configs/video/<BEST_CONFIG>.json \
  --checkpoint artifacts/runs/video/<BEST_RUN_NAME>/best_model.pth \
  --split test \
  --split-manifest artifacts/runs/video/<BEST_RUN_NAME>/split_manifest.json \
  --output artifacts/runs/video/<BEST_RUN_NAME>/final_test/video_embeddings.json
```

最终图表产出：

- 最终主结果表：`figures/final/video_metrics_summary.csv`
- 相似度热图：`figures/final/BEST_similarity_heatmap.png`
- 指标柱状图：`figures/final/BEST_task_metrics.png`
- embedding 原始文件：`final_test/video_embeddings.json`
- Top-K 成功/失败案例表：`figures/final/video_cases_summary.csv`

## 5. 汇总表设计

### 5.1 主结果表

| Experiment | Backbone | Split | Loss | Encoder Mode | Intra | H2R R@1 | H2R R@10 | H2R MRR | H2R NDCG@10 | R2H R@1 | R2H R@10 | R2H MRR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

### 5.2 消融表

| Ablation | Baseline | Variant | Main Metric | Change | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Backbone | E1 | E2 | H2R task MRR |  |  |
| Split | E1 | E3 | H2R task MRR |  |  |
| Intra loss | E1 | E4 | H2R task MRR |  |  |
| Loss | E1 | E5 | H2R task MRR |  |  |
| Encoder mode | E6 shared | E1 dual_head | H2R task MRR |  |  |
| Encoder mode | E1 dual_head | E6 dual_encoder | H2R task MRR |  |  |

## 6. 推荐执行顺序

| 顺序 | 实验 | 原因 |
| :--- | :--- | :--- |
| 1 | E1 | 先建立 V-JEPA 主基线。 |
| 2 | E2 | 立刻完成 backbone 对照，决定后续是否继续以 V-JEPA 为主。 |
| 3 | E5 | 先验证损失函数，因为它影响训练目标解释。 |
| 4 | E4 | 再验证 intra loss 是否值得保留。 |
| 5 | E6 | 比较模型结构，注意 dual_encoder 成本更高。 |
| 6 | E3 | 最后跑 task-held-out 泛化，用于补充分析。 |
| 7 | E7 | 对最佳模型做最终导出。 |

## 7. 停止条件

满足以下条件即可结束视频单模态实验：

- E1-E2 完成，确定 backbone 选择。
- E4-E6 至少完成主要对照，能解释最终模型结构和损失选择。
- E3 完成或明确记录未执行原因。
- E7 完成，产出最终 metrics、embedding、similarity matrix、cases 和 figure 目录。
- 至少完成 4 张表/图：主结果表、消融表、相似度热图、Top-K 案例表。

不建议继续扩展的实验：

- 不做大规模 batch size 搜索。
- 不做大规模 `num_frames` 搜索。
- 不做 memory bank，除非 E5 证明 batch 内正负样本不足严重影响结果。
- 不做视频与轨迹/语义联合实验。
