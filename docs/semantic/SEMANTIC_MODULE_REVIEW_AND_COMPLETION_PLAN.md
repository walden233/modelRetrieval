# 语义模块现状评审与补全计划

更新时间：2026-05-02

本文档基于当前 `artifacts/semantic`、`runs/semantic`、`runs/batch`、`src/bise/modalities/semantic`、`configs/semantic` 与 `docs/semantic` 的实际实现，评估语义层 VLM 模块设计是否合理、完成度如何、当前硬伤是什么，以及后续需要补充什么。

## 1. 当前结论

语义模块已经不是占位实现，当前已经具备一条可运行的离线语义处理链路：

```text
manifest 构建
-> VLM 语义标注，sync 或 batch
-> raw response / normalized annotation / cache 持久化
-> text_embedding / label_embedding 生成
-> feature_store 写入
-> human-to-robot / robot-to-human 语义检索评估
```

当前 `artifacts/semantic/rh20t/cfg2` 已有完整 cfg2 产物：

| 产物 | 当前状态 |
| :--- | :--- |
| `manifests/semantic_manifest_v1.jsonl` | 436 行 |
| `annotations/raw_responses.jsonl` | 436 行 |
| `annotations/normalized_annotations.jsonl` | 436 行 |
| `errors/failed_samples.jsonl` | 0 行 |
| `embeddings/text_embedding_v1.npy` | `(436, 1024)` |
| `embeddings/label_embedding_v1.npy` | `(436, 1024)` |
| `feature_store/semantic_features.json` | 436 条 |
| `batch/request_files.json` | 已生成 |
| `batch/jobs.json` | 已存在 |

这对应 218 个 `human/robot` scene pair。当前用 `pair_id` 作为正样本定义时，语义检索结果为：

| 模式 | R@1 | R@5 | R@10 | MRR | Mean Rank |
| :--- | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.1514 | 0.4358 | 0.5596 | 0.2860 | 30.15 |
| label_only | 0.0229 | 0.0642 | 0.1239 | 0.0616 | 93.24 |
| text_plus_label | 0.1239 | 0.4037 | 0.5183 | 0.2554 | 35.38 |

初步判断：

- `text_embedding` 有一定检索能力。
- `label_embedding` 当前很弱，且拉低了 `text_plus_label`。
- 语义模块工程闭环基本成立，但正式实验闭环和指标输出还弱于视频/轨迹模块。
- 有几个必须修的硬伤，尤其是 API key 明文配置、scene_id 冲突、语义评估指标不完整。

已运行语义相关测试：

```bash
pytest -q tests/test_semantic_*.py tests/test_build_semantic_manifest.py
```

结果：

```text
28 passed
```

## 2. 当前设计是否合理

### 2.1 合理的部分

整体分层是合理的：

| 层 | 文件 | 评价 |
| :--- | :--- | :--- |
| Schema | `schemas.py` | 数据结构清晰，manifest / annotation / embedding / eval record 分离合理。 |
| Prompt | `prompts.py` + `configs/semantic/prompt_*.json` | prompt 配置化，支持 description / label / joint。 |
| VLM Client | `vlm_client.py` | 支持 stub 与 OpenAI-compatible API，便于本地测试和远程调用。 |
| Parser | `parser.py` | 支持 JSON 与 code fence，职责清楚。 |
| Normalizer | `normalizer.py` | taxonomy 归一、alias 映射、canonical text 独立出来是正确设计。 |
| Pipeline | `pipeline.py` | sync 处理、缓存、失败记录、增量输出基本完整。 |
| Batch | `batch.py` + `runs/batch/*` | 支持 request shard、submit、sync、ingest，适合大规模离线标注。 |
| Embedder | `embedder.py` | 支持 hash stub 与 transformers embedding，测试和正式处理可分离。 |
| Retrieval Eval | `evaluator.py` + `runs/semantic/evaluate_semantic_retrieval.py` | 能完成 role split、text/label/combined 三种检索评估。 |

当前语义模块采用“单视频语义理解 + human/robot 跨域语义检索”的方式，与项目目标是匹配的。每个 scene 采样一个 camera pair，然后拆成 `robot` 和 `human` 两条记录，这个粒度也合理，避免每个 camera 都调用 VLM 带来成本爆炸。

### 2.2 不够合理的部分

当前设计的主要问题不在代码能否跑，而在“正式实验可信度”和“语义标签可解释性”不足：

- `text_embedding`、`label_embedding`、`combined` 的结果没有形成和视频/轨迹一致的 `final_test/metrics.json`、`cases.json`、图表产物。
- 语义检索没有固定 split / cfg2->cfg3 all-test 规范，当前更像全量离线评估。
- `label_only` 明显弱，说明 taxonomy 和 canonical label text 的区分度不足，不能直接作为有力语义结果。
- prompt 明确要求不要颜色信息，但实际 annotation 中仍出现类似颜色描述，缺少输出后校验。
- 配置里存在明文 API key，这是必须立即处理的安全问题。

## 3. 完成情况

| 模块 | 完成度 | 说明 |
| :--- | :--- | :--- |
| Manifest 构建 | 已完成 | 支持 RH20T / WHIRL，支持 `scenes_per_task`。 |
| Camera pair 采样 | 基本完成 | 支持 preferred camera，但当前 cfg2 配置值与实际 camera id 不匹配。 |
| Sync VLM 标注 | 已完成 | 支持 stub/API、cache、错误记录、断点续跑。 |
| Batch VLM 标注 | 已完成 | 支持生成 batch requests、提交、同步、下载、ingest。 |
| Response parser | 已完成 | 支持 JSON/code fence，失败时抛出明确异常。 |
| Taxonomy normalizer | 基本完成 | 支持 alias、非法值回退；但 taxonomy 粒度偏粗。 |
| Text embedding | 已完成 | 支持 hash 和 transformers。 |
| Feature store | 已完成 | 保存到通用 `FeatureStore`，包含 text/label embedding。 |
| Semantic retrieval | 基本完成 | 支持 text/label/combined、role split、positive key。 |
| Label gold eval | 半完成 | 入口存在，但缺少 gold labels 数据与正式流程。 |
| 图表产出 | 未完成 | 没有类似视频/轨迹的 final charts。 |
| cfg3 评估 | 未完成 | 当前没有语义 cfg2->cfg3 all-test 脚本和文档规范。 |
| 安全配置 | 未完成 | API key 仍在 config 中明文保存。 |

## 4. 当前硬伤

### S0. 明文 API key

严重级别：必须立即修。

`configs/semantic/provider_api.json` 中保存了明文 API key。这个文件如果进入版本控制或共享环境，会直接泄露凭据。

建议改为：

```json
{
  "provider_name": "openai_compatible",
  "base_url": "https://open.bigmodel.cn/api/paas/v4",
  "api_key_env": "ZHIPU_API_KEY",
  "model_name": "glm-4.6v",
  "thinking_type": "disabled",
  "timeout_seconds": 60,
  "max_retries": 3
}
```

并通过环境变量运行：

```bash
export ZHIPU_API_KEY=...
```

### S1. `scene_id` 不包含 task，存在跨 task 碰撞

严重级别：高。

当前 manifest 中：

```text
task_0001 / scene_1 -> scene_id = scene_1
task_0002 / scene_1 -> scene_id = scene_1
```

这会导致如果语义检索使用 `--positive-key scene_id`，不同 task 的同名 scene 会被错误认为是正样本。视频和轨迹模块已经把 scene 表示修成了 task-scoped scene，语义模块也应对齐。

建议：

- `scene_id` 改为 `task_0001/scene_1`。
- 如果还需要原始 scene 名，新增 `scene_name = scene_1`。
- 旧产物需要重新生成 manifest 和 annotations，或者在评估时禁止使用 `positive-key=scene_id`。

### S1. 语义评估指标弱于视频/轨迹

严重级别：高。

当前 `evaluate_semantic_retrieval.py` 只打印 stdout，指标包括：

- `R@1`
- `R@5`
- `R@10`
- `MRR`
- `Mean Rank`

缺少：

- `NDCG@10`
- `Mean Percentage Rank`
- `valid_queries`
- 双向结果统一结构
- `metrics.json`
- `cases.json`
- similarity matrix
- embedding 导出
- metrics comparison 图表

这会导致语义结果无法和视频/轨迹实验直接对齐。

### S1. `build_index.py` 默认字段错误

严重级别：高。

`runs/semantic/build_index.py` 默认：

```text
--field trajectory_embedding
```

但语义 feature store 的主要字段是：

- `text_embedding`
- `label_embedding`

因此直接运行 `build_index.py` 会找不到有效语义 embedding，必须显式传 `--field text_embedding` 或 `--field label_embedding`。默认值应改成 `text_embedding`。

### S1. prompt 的颜色约束没有被后处理校验

严重级别：高。

prompt 已要求不要颜色信息，但当前实际 annotation 中仍可能出现颜色描述，例如描述物体表面颜色。说明仅靠 prompt 约束不够。

建议补一个 description validator：

- 检测常见颜色词。
- 记录到 `errors` 或 `warnings`。
- 可选地触发二次 rewrite prompt。

### S1. label-only 检索很弱

严重级别：高。

当前 cfg2 结果：

```text
label_only R@1 = 0.0229
label_only MRR = 0.0616
```

说明 `capability_tags + task_complexity + environment + scene_category` 这套标签对 pair-level 匹配区分度很低。原因可能包括：

- taxonomy 太粗，很多任务共享同一组标签。
- `environment_tags` 和 `scene_category` 基本低信息量。
- human 和 robot 对同一 pair 的标签不一致，例如一个是 `grasp/place`，另一个是 `grasp/transport`。
- label canonical text 只编码标签，不含对象、目标、动作顺序等关键语义。

短期结论：正式语义检索主结果应优先看 `text_only`，`label_only` 只能作为辅助分析。

### S2. preferred camera 配置疑似无效

严重级别：中。

`configs/semantic/pipeline_v1.json` 中：

```json
"preferred_robot_cam_id": "cam_0",
"preferred_human_cam_id": "cam_0"
```

但 RH20T 文件名实际 camera id 类似：

```text
cam_036422060215
cam_037522062165
cam_104122061850
```

因此 preferred camera 大概率匹配不到，实际走 fallback 的排序第一个 camera。虽然可运行，但配置含义误导。

建议：

- 把默认值改为真实 camera id。
- 或明确为空字符串，表示使用排序第一个 camera。
- manifest 生成 summary 中输出实际选中的 camera 分布。

### S2. `scenes_per_task=2` 限制了语义覆盖

严重级别：中。

当前 manifest 是 436 条，即 218 个 scene pair，符合 `109 tasks * 2 scenes * 2 roles`。这适合降低 VLM 成本，但不代表完整 RH20T cfg2。

正式报告必须说明：

- 语义结果只覆盖每个 task 最多 2 个 scene。
- 不应和视频/轨迹全量 test 结果直接当作同规模实验比较。

### S2. embedding 模型使用方式可能不是最优

严重级别：中。

当前 `TransformersTextEmbedder` 使用 `AutoModel` + mean pooling。对 `BAAI/bge-m3` 来说，这不一定是推荐推理方式。更稳妥的方式是使用官方或 sentence-transformers/FlagEmbedding 风格接口，并确认 normalize 方式。

短期可以先保留，但正式结果需要说明 embedding 实现方式。

### S2. sync/batch 入口分散

严重级别：中。

当前入口分布：

```text
runs/semantic/build_semantic_manifest.py
runs/semantic/run_semantic_annotation.py
runs/batch/build_semantic_batch_requests.py
runs/batch/submit_semantic_batch.py
runs/batch/sync_semantic_batch_jobs.py
runs/batch/ingest_semantic_batch_results.py
runs/semantic/evaluate_semantic_retrieval.py
```

功能上可用，但从实验执行角度不够统一。建议后续补 `scripts/semantic_*.sh` 或 `runs/semantic/run_all_semantic_pipeline.sh`。

### S3. `JsonCache` 每次 set 都重写完整 JSON

严重级别：低到中。

当前 436 条规模问题不大；如果扩展到数万请求，频繁重写 cache JSON 会慢，并且异常中断时可能损坏文件。

后续可以改为：

- JSONL append-only cache。
- SQLite cache。
- 或每 N 条 flush。

## 5. 建议补充内容

### P0. 安全修复

必须先做：

1. 移除 `provider_api.json` 中的明文 key。
2. 改为 `api_key_env`。
3. 检查 git 历史或共享副本是否已经泄露。
4. 必要时轮换 API key。

### P1. 对齐视频/轨迹评估闭环

新增语义最终评估入口，建议命名：

```text
runs/semantic/evaluate_semantic_retrieval.py
```

扩展输出：

```text
artifacts/runs/semantic/<RUN>/
  params.json
  metrics.json
  cases.json
  similarity_matrix.npy
  query_embeddings.npy
  gallery_embeddings.npy
  metadata.json
  semantic_metrics_comparison.png
  semantic_metrics_comparison_data.json
```

指标结构建议对齐：

```json
{
  "human_to_robot": {
    "pair": {},
    "task": {},
    "scene": {}
  },
  "robot_to_human": {
    "pair": {},
    "task": {},
    "scene": {}
  }
}
```

其中每层至少包含：

- `R@1`
- `R@5`
- `R@10`
- `Mean Rank`
- `MRR`
- `Mean Percentage Rank`
- `NDCG@10`
- `valid_queries`

### P1. 修复 scene_id

建议 manifest 生成改为：

```python
scene_name = scene_path.name
scene_id = f"{task_id}/{scene_name}"
```

并在 record metadata 中保留：

```json
{
  "scene_name": "scene_1"
}
```

### P1. 修正 semantic index 默认字段

将：

```bash
--field trajectory_embedding
```

改为：

```bash
--field text_embedding
```

并在文档中明确：

```bash
python runs/semantic/build_index.py \
  --features artifacts/semantic/rh20t/cfg2/feature_store/semantic_features.json \
  --field text_embedding \
  --output artifacts/semantic/rh20t/cfg2/index/text_embedding.faiss
```

### P1. 补最终图表脚本

建议新增：

```text
runs/semantic/export_final_semantic_charts.py
```

输出：

```text
semantic_metrics_comparison.png
semantic_metrics_comparison_data.json
```

图中比较：

- `text_only`
- `label_only`
- `text_plus_label`

指标：

- `R@1`
- `R@5`
- `R@10`
- `MRR`
- `NDCG@10`

### P2. label 体系增强

当前 taxonomy 太粗。建议新增字段：

- `object_category`
- `target_category`
- `action_sequence`
- `spatial_relation`
- `tool_or_container`

并修改 `label_canonical_text`：

```text
action_sequence: grasp -> transport -> place;
object: cup;
target: shelf;
capabilities: grasp, transport, place;
environment: no_obstacle;
scene_category: industrial
```

这样 label embedding 才可能具备 pair-level 区分能力。

### P2. description 后处理校验

建议新增：

```text
src/bise/modalities/semantic/validators.py
```

至少检测：

- 颜色词。
- 过长描述。
- 空对象/空动作。
- 不合法标签。
- human/robot 同 pair 描述差异过大。

输出：

```text
artifacts/semantic/rh20t/cfg2/quality/description_warnings.jsonl
artifacts/semantic/rh20t/cfg2/quality/pair_consistency.jsonl
```

### P2. cfg3 语义处理流程

如果要把语义模块纳入 cfg2 -> cfg3 泛化分析，需要补：

```text
configs/semantic/pipeline_cfg3.json
scripts/semantic_cfg3_build_manifest.sh
scripts/semantic_cfg3_batch_requests.sh
scripts/semantic_cfg3_ingest.sh
scripts/semantic_cfg3_evaluate.sh
```

注意 cfg3 不能复用 cfg2 annotation，必须重新调用 VLM 或至少重新抽样并明确标注成本。

## 6. 推荐后续执行顺序

| 优先级 | 事项 | 原因 |
| :--- | :--- | :--- |
| P0 | 移除明文 API key | 安全问题，必须先处理。 |
| P1 | 修复 `scene_id` task-scoped 问题 | 避免 scene-level 评估错误。 |
| P1 | 扩展 semantic retrieval 输出 metrics/cases/matrix | 和视频/轨迹结果对齐。 |
| P1 | 增加 NDCG@10 / MPR / valid_queries | 指标可比性需要。 |
| P1 | 修正 `build_index.py` 默认字段 | 避免常用命令直接失败。 |
| P1 | 补 semantic final chart 脚本 | 论文图表需要。 |
| P2 | 增强 taxonomy 和 canonical label text | 当前 label-only 太弱。 |
| P2 | 增加 description/label 质量检查 | 防止 prompt 约束失效。 |
| P2 | 补 cfg3 semantic pipeline | 需要跨配置语义泛化时再做。 |

## 7. 当前可用命令

构建 manifest：

```bash
python runs/semantic/build_semantic_manifest.py \
  --config configs/semantic/pipeline_v1.json
```

生成 batch requests：

```bash
python runs/batch/build_semantic_batch_requests.py \
  --config configs/semantic/pipeline_v1.json
```

提交 batch：

```bash
python runs/batch/submit_semantic_batch.py \
  --config configs/semantic/pipeline_v1.json
```

同步 batch：

```bash
python runs/batch/sync_semantic_batch_jobs.py \
  --config configs/semantic/pipeline_v1.json
```

导入 batch 结果：

```bash
python runs/batch/ingest_semantic_batch_results.py \
  --config configs/semantic/pipeline_v1.json
```

语义检索评估：

```bash
python runs/semantic/evaluate_semantic_retrieval.py \
  --annotations artifacts/semantic/rh20t/cfg2/annotations/normalized_annotations.jsonl \
  --positive-key pair_id
```

当前不建议直接运行的命令：

```bash
python runs/semantic/build_index.py \
  --features artifacts/semantic/rh20t/cfg2/feature_store/semantic_features.json \
  --output artifacts/semantic/rh20t/cfg2/index/default.faiss
```

原因是默认 `--field trajectory_embedding` 不适合语义 feature store。应显式指定：

```bash
python runs/semantic/build_index.py \
  --features artifacts/semantic/rh20t/cfg2/feature_store/semantic_features.json \
  --field text_embedding \
  --output artifacts/semantic/rh20t/cfg2/index/text_embedding.faiss
```

## 8. 最小可接受补全标准

如果只做最必要补充，建议完成以下 5 项：

1. `provider_api.json` 改为环境变量读取 key。
2. `scene_id` 改为 task-scoped，并重新生成 cfg2 semantic artifacts。
3. `evaluate_semantic_retrieval.py` 输出 `metrics.json`、`cases.json`、`similarity_matrix.npy`、`metadata.json`。
4. 语义指标补齐 `NDCG@10`、`Mean Percentage Rank`、`valid_queries`。
5. 新增 `export_final_semantic_charts.py`，生成 `semantic_metrics_comparison.png`。

完成这 5 项后，语义模块才能和视频/轨迹模块进入同一套论文结果呈现框架。
