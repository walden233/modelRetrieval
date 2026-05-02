# 语义单模态实验计划与入口说明

本文档只覆盖语义模态自身：VLM 语义描述、结构化标签、文本 embedding 检索。不包含视频、轨迹或跨模态联合实验。

## 1. 本次实现约束

- `configs/semantic/provider_api.json` 不再保存明文 API key，统一通过环境变量读取。
- RH20T manifest 的 `scene_id` 已改为 `task_id/scene_name`，例如 `task_0001/scene_1`，避免不同 task 下的 `scene_1` 在 scene-level 评估中互相误判为正样本。
- 为兼容旧 annotation，评估阶段读取 `scene_id` 正样本时也会自动按 `task_id/scene_id` 归一化。
- 语义检索评估已对齐视频/轨迹输出，支持 `metrics.json`、`summary.json`、`cases.json`、`metadata.json`、`similarity_matrices.npz`、`semantic_embeddings.npz`。
- 语义检索指标包含 `R@1`、`R@5`、`R@10`、`Mean Rank`、`MRR`、`Mean Percentage Rank`、`NDCG@10`、`valid_queries`。
- `runs/semantic/build_index.py` 默认索引字段已改为 `text_embedding`，更符合语义 feature store。

注意：已有旧版 `normalized_annotations.jsonl` 如果仍然只有 `scene_id=scene_1`，当前评估代码会自动用 `task_id/scene_id` 防碰撞；但正式归档仍建议重建 manifest 和 annotation，使产物本身字段一致。

## 2. 环境与 API Key

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch2
export ZHIPU_API_KEY="你的实际 key"
```

`configs/semantic/provider_api.json` 中只保留：

```json
{
  "api_key_env": "ZHIPU_API_KEY"
}
```

不要把真实 key 写入配置文件或文档。

## 3. 推荐实验设计

语义单模态不需要做大量训练实验，核心问题是：VLM 输出中的自然语言描述、结构化标签、二者融合，谁更适合作为跨领域人机语义检索特征。

| 实验 | 目标 | 输入特征 | 正样本定义 | 主要产出 |
|---|---|---|---|---|
| S0 Artifact Check | 检查语义产物是否完整 | annotation / embedding / feature store | 不评估 | 样本数、失败数、embedding 维度 |
| S1 Text Main | 主实验，验证自然语言描述检索能力 | `text_embedding` | `pair_id` | `metrics.json`、`cases.json`、final chart |
| S2 Label Ablation | 验证结构化标签是否足够表达任务 | `label_embedding` | `pair_id` | 与 S1 同图对比 |
| S3 Text+Label Fusion | 验证描述和标签融合是否互补 | normalized `text + label` | `pair_id` | 与 S1/S2 同图对比 |
| S4 Level Analysis | 分析 task/scene/pair 粒度差异 | 三种语义特征 | `task_id` / `scene_id` / `pair_id` | 多粒度指标表 |

最低必要实验是 S1-S3。S4 只用于解释模型到底是在识别同一配对、同一任务，还是同一 task-scoped scene。

## 4. 标准入口

### 4.1 生成 manifest

```bash
python runs/semantic/build_semantic_manifest.py \
  --config configs/semantic/pipeline_v1.json \
  --dataset-type rh20t \
  --data-root dataset/RH20T_subset/RH20T_cfg2 \
  --scenes-per-task 2
```

如需覆盖输出路径：

```bash
python runs/semantic/build_semantic_manifest.py \
  --config configs/semantic/pipeline_v1.json \
  --output artifacts/semantic/rh20t/cfg2/manifests/semantic_manifest.jsonl
```

### 4.2 同步语义标注

```bash
python runs/semantic/run_semantic_annotation.py \
  --config configs/semantic/pipeline_v1.json
```

调试小批量：

```bash
python runs/semantic/run_semantic_annotation.py \
  --config configs/semantic/pipeline_v1.json \
  --start-index 0 \
  --end-index 8
```

### 4.3 检索评估并导出标准产物

推荐使用脚本入口，变量集中在前面，便于改路径：

```bash
scripts/evaluate_semantic_retrieval.sh
```

等价显式命令：

```bash
python runs/semantic/evaluate_semantic_retrieval.py \
  --annotations artifacts/semantic/rh20t/cfg2/annotations/normalized_annotations.jsonl \
  --query-role human \
  --gallery-role robot \
  --positive-key pair_id \
  --output-dir artifacts/semantic/rh20t/cfg2/evaluation/pair \
  --top-k 10
```

输出目录包含：

- `metrics.json`：和视频/轨迹一致的嵌套指标，结构为 `direction -> level -> mode`。
- `summary.json`：包含参数、样本数和兼容旧输出的摘要。
- `cases.json`：每个 query 的 top-k 检索案例。
- `metadata.json`：query/gallery 样本元信息。
- `similarity_matrices.npz`：三种语义模式的人到机、机到人相似度矩阵。
- `semantic_embeddings.npz`：评估时实际使用的 embedding。

### 4.4 导出最终图表

```bash
scripts/export_semantic_final_charts.sh
```

等价显式命令：

```bash
python runs/semantic/export_final_semantic_charts.py \
  --eval-dir artifacts/semantic/rh20t/cfg2/evaluation/pair \
  --output-dir artifacts/semantic/rh20t/cfg2/final_charts \
  --direction human_to_robot \
  --level pair \
  --dpi 400
```

输出：

- `semantic_metrics_comparison.png`：论文可用的柱状对比图。
- `semantic_metrics_comparison_data.json`：绘图使用的原始数据。

### 4.5 构建语义索引

默认索引 `text_embedding`：

```bash
python runs/semantic/build_index.py \
  --features artifacts/semantic/rh20t/cfg2/feature_store/semantic_features.json \
  --output artifacts/semantic/rh20t/cfg2/index/text_embedding.faiss
```

如需索引结构化标签：

```bash
python runs/semantic/build_index.py \
  --features artifacts/semantic/rh20t/cfg2/feature_store/semantic_features.json \
  --field label_embedding \
  --output artifacts/semantic/rh20t/cfg2/index/label_embedding.faiss
```

## 5. 结果记录表

每次正式实验记录以下内容即可：

| 项目 | 记录内容 |
|---|---|
| 数据版本 | RH20T cfg2 / manifest 路径 / annotation 路径 |
| API 模型 | `model_name`、prompt 版本、taxonomy 版本 |
| 样本统计 | query 数、gallery 数、失败样本数 |
| 主指标 | pair-level human_to_robot 的 R@1/R@5/R@10/MRR/NDCG@10 |
| 辅助指标 | robot_to_human、task-level、scene-level |
| 图表 | `semantic_metrics_comparison.png` |
| 失败案例 | `cases.json` 中 top-k 错检样例 |

## 6. 判读原则

- `text_only` 高于 `label_only`：说明 VLM 的自由文本描述保留了更多任务细节，结构化标签过粗。
- `label_only` 高于 `text_only`：说明结构化 taxonomy 对跨领域归一化更稳定，自由描述噪声较大。
- `text_plus_label` 高于二者：说明描述和标签互补，可作为语义单模态最终设置。
- `task-level` 明显高、`pair-level` 低：说明语义特征能识别任务类别，但不足以区分同任务下的具体场景或动作细节。
- `scene-level` 指标依赖 task-scoped scene key；新 manifest 会直接写入该字段，旧 annotation 会在评估时兼容归一化。
