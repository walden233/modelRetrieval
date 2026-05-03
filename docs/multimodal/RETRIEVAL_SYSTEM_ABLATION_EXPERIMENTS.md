# 三模态检索系统消融实验设计与入口

更新时间：2026-05-03

本文档只定义检索系统的模态消融实验。消融实验不做权重搜索，统一使用组合内均匀权重，目的是回答“某个模态或模态组合是否有贡献”，而不是回答“最优融合权重是多少”。

## 1. 实验原则

- 固定同一个检索库：默认 `artifacts/retrieval/rh20t_cfg2_v1`。
- 固定同一批 query：默认要求 `video,trajectory,semantic_text` 三个模态都完整。
- 固定融合策略：`zscore + weighted_sum + missing_policy=renormalize`。
- 固定权重规则：启用几个模态，就对几个模态均匀赋权。
- 不在 test 上搜索权重，不把权重搜索结果混入消融。

这样每个实验只改变 `enabled_modalities`，指标差异才主要反映模态贡献。

## 2. 配置文件

消融配置位于：

```text
configs/retrieval/ablation/
```

| 实验 | 配置 | 权重 |
|---|---|---|
| Video | `video_only_uniform.json` | video=1.0 |
| Trajectory | `trajectory_only_uniform.json` | trajectory=1.0 |
| Semantic | `semantic_text_only_uniform.json` | semantic_text=1.0 |
| Video+Trajectory | `video_trajectory_uniform.json` | 0.5 / 0.5 |
| Video+Semantic | `video_semantic_text_uniform.json` | 0.5 / 0.5 |
| Trajectory+Semantic | `trajectory_semantic_text_uniform.json` | 0.5 / 0.5 |
| Video+Trajectory+Semantic | `video_trajectory_semantic_text_uniform.json` | 1/3 each |

这些配置只控制“启用哪些模态、怎么融合”。查询样本是否要求三模态完整由评估入口的 `--require-modalities` 控制。

## 3. 一键运行

先确保检索库已经建好：

```bash
scripts/build_retrieval_library.sh
```

运行全部消融：

```bash
scripts/run_retrieval_system_ablation.sh
```

默认输出：

```text
artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/
├── video_only/
├── trajectory_only/
├── semantic_text_only/
├── video_trajectory/
├── video_semantic_text/
├── trajectory_semantic_text/
├── video_trajectory_semantic_text/
└── ablation_runs.json
```

每个子目录包含：

- `metrics.json`
- `cases.json`
- `per_query_results.jsonl`
- `run_info.json`
- `fused_similarity_matrix.npy`
- `<modality>_similarity_matrix.npy`

## 4. 常用变量

默认要求三模态完整 query：

```bash
REQUIRE_MODALITIES=video,trajectory,semantic_text scripts/run_retrieval_system_ablation.sh
```

如果要测试真实缺失模态场景，可以放宽 query 要求，例如只要求 video：

```bash
REQUIRE_MODALITIES=video OUTPUT_ROOT=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_video_available scripts/run_retrieval_system_ablation.sh
```

更换检索库：

```bash
LIBRARY_DIR=artifacts/retrieval/rh20t_cfg3_v1 OUTPUT_ROOT=artifacts/retrieval/rh20t_cfg3_v1/eval/ablation_uniform scripts/run_retrieval_system_ablation.sh
```

更换评价 level：

```bash
LEVEL=scene scripts/run_retrieval_system_ablation.sh
LEVEL=task scripts/run_retrieval_system_ablation.sh
LEVEL=mixed scripts/run_retrieval_system_ablation.sh
```

注意：当前评估脚本总是同时输出 `scene/task/mixed` 三组指标，`LEVEL` 主要记录当前关注的评价设置。

## 5. 单个实验入口

只跑 Video：

```bash
python runs/retrieval/evaluate_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/ablation/video_only_uniform.json \
  --level mixed \
  --require-modalities video,trajectory,semantic_text \
  --output-dir artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/video_only \
  --top-k 10
```

只跑 Video+Semantic：

```bash
python runs/retrieval/evaluate_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/ablation/video_semantic_text_uniform.json \
  --level mixed \
  --require-modalities video,trajectory,semantic_text \
  --output-dir artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/video_semantic_text \
  --top-k 10
```

## 6. 图表导出

一键消融脚本会生成：

```text
artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/ablation_runs.json
```

导出 scene-level 图表：

```bash
RUNS_JSON=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/ablation_runs.json \
OUTPUT_DIR=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/charts \
LEVEL=scene \
scripts/export_retrieval_system_charts.sh
```

导出 task-level 图表：

```bash
RUNS_JSON=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/ablation_runs.json \
OUTPUT_DIR=artifacts/retrieval/rh20t_cfg2_v1/eval/ablation_uniform/charts_task \
LEVEL=task \
scripts/export_retrieval_system_charts.sh
```

输出：

- `system_metrics_comparison.png`
- `system_metrics_comparison_data.json`

## 7. 推荐报告方式

主表建议用 scene-level：

| Method | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Video |  |  |  |  |  |
| Trajectory |  |  |  |  |  |
| Semantic |  |  |  |  |  |
| Video+Trajectory |  |  |  |  |  |
| Video+Semantic |  |  |  |  |  |
| Trajectory+Semantic |  |  |  |  |  |
| Video+Trajectory+Semantic |  |  |  |  |  |

辅助报告 mixed 指标：

- `MixedNDCG@10`
- `SceneHit@1`
- `SceneHit@5`
- `TaskHit@5`
- `TaskOnlyHit@5`

## 8. 和权重搜索的关系

消融实验不使用搜索权重。推荐实验顺序：

1. 用本消融实验确定哪些模态组合值得保留。
2. 只对最有希望的组合做 val 权重搜索。
3. 用 val 得到的固定权重跑 test。
4. 报告中分开写 `Uniform-weight ablation` 和 `Tuned-weight fusion`。

如果每个消融组合都单独调权，消融结论会被调参能力污染，不再是纯粹的模态贡献分析。
