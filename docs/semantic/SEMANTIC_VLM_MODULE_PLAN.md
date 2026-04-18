# 语义层 VLM 模块完整开发方案

更新时间：2026-04-17

本文档基于当前 `BISE` 仓库现状，收敛语义模块范围为两部分：

1. 语义提取层：分别对机器人执行视频和人类演示视频批量生成语义结果。
2. 语义检索层：在语义结果已经提取完成的前提下，基于语义向量做相似度检索与验证。

现阶段重点是离线批量处理现有数据集。对外统一接口暂不实现，但代码结构需要预留后续扩展位。

## 1. 模块目标与边界

### 1.1 语义提取层

单次输入仅包括一个视频：

- `robot_video`
- 或 `human_video`

输出仅包括：

- `task_description`
- `capability_tags`
- `action_slots`
- `text_embedding`
- `label_embedding`

说明：

- `task_description` 是单条规范化任务描述，用于检索和展示。
- `capability_tags` 是受控多标签能力标签。
- `action_slots` 是结构化动作槽位，其中 `object` 和 `target` 必须存在。
- `text_embedding` 是对 `task_description` 编码得到的语义向量。
- `label_embedding` 是对规范标签文本编码得到的语义向量。

语义提取层现阶段不负责：

- 对外 API 的完整实现。
- 多模态融合排序。
- 视频或轨迹 backbone 训练。

### 1.2 语义检索层

前提：

- 所有样本已经完成语义提取。
- 每个样本都已经保存 `task_description`、`capability_tags`、`action_slots`、`text_embedding`、`label_embedding`。

目标：

- 基于语义向量建立检索库。
- 给定查询描述，检索语义最接近的目标视频样本。
- 验证 `text_embedding`、`label_embedding` 及其组合的效果。

## 2. 推荐技术栈

### 2.1 总体原则

当前阶段不需要复杂 agent 框架，也不需要先实现外部接口。建议采用“两套 prompt + taxonomy 约束 + 本地或远程 embedding 模型 + 离线批处理”的最小可行方案。

### 2.2 推荐主栈

| 层 | 推荐方案 | 作用 |
| :--- | :--- | :--- |
| VLM | `Qwen2.5-VL` 或同类视频理解 VLM | 生成任务描述、能力标签、动作槽位 |
| VLM 接入 | 自定义 `vlm_client.py`，兼容本地服务或远程 API | 避免绑定 LangChain 等额外抽象 |
| 文本 embedding | `BAAI/bge-m3` 或同类 embedding 模型 | 生成 `text_embedding` 与 `label_embedding` |
| 配置管理 | `configs/semantic/*.json` | 管理 prompt、taxonomy、provider、批处理参数 |
| 持久化 | `jsonl + parquet + npy` | 存语义结果、评估集和向量结果 |
| 向量数据库 | `FAISS` | 当前阶段离线建索引与相似度检索 |
| 检索 | 仓库现有 `FeatureStore`、`faiss_index.py`、`metrics.py` | 语义索引与相似度检索 |

说明：

- VLM 可以本地部署，也可以直接调 API。
- 不要求必须使用 `vLLM`，只有选择本地部署大模型时才推荐用它做服务层。
- 不建议引入 `LangChain`，当前需求用自定义轻量 client 更合适。
- 当前阶段优先使用 `FAISS`，因为仓库已经有 `faiss_index.py`，且本阶段重点是离线验证，不需要额外引入 `Qdrant` 或 `Milvus` 这类独立服务。
- 当前阶段不建议把“向量数据库对比实验”作为正式实验项；如后续进入在线服务阶段，再评估 `Qdrant` 或 `Milvus`。
- 当前阶段也不建议把“不同 embedding 模型对比”作为正式实验项；如语义检索结果明显不足，再补充一个小规模 embedding 模型对照实验。

## 3. 关键概念

### 3.1 Manifest

`manifest` 是待处理样本清单，不是模型结果表。它的作用是统一管理“哪些视频需要做语义提取”。

当前数据组织特点：

- 一个 `task` 下包含多个 `scene`
- 一个 `scene` 下包含多个相机视频

因此语义提取的基本单元不应是“每个视频”，而应是“每个 `scene` 采样一个 `cam` 对后，再拆成 `robot` / `human` 两条单视频记录”。

推荐策略：

- 每个 `scene` 仅选择一组 `robot_cam` 与 `human_cam`
- 基于该 `cam` 对生成两条语义记录：
  - 一条 `video_role=robot`
  - 一条 `video_role=human`
- 单次 VLM 请求只处理其中一路视频

建议字段：

- `sample_id`
- `pair_id`
- `task_id`
- `scene_id`
- `video_role`
- `cam_id`
- `video_path`
- `paired_video_path`
- `dataset_name`
- `status`
- `description_prompt_version`
- `label_prompt_version`
- `taxonomy_version`

用途：

- 批量处理
- 断点续跑
- 失败重试
- 评估与检索时统一主键
- 控制到 `scene` 级别而不是 `video` 级别的语义生成粒度

### 3.2 Taxonomy

`taxonomy` 是受控标签体系，用来约束 `capability_tags` 的取值范围。

示例：

- `grasp`
- `place`
- `insert`
- `rotate`
- `pour`

用途：

- 限制模型输出范围
- 提高标签一致性
- 便于人工标注对比和 `Precision / Recall / F1` 计算
- 便于生成稳定的 `label_embedding`

## 4. 输出结构设计

### 4.1 task_description

定义：

- 单条简洁、规范、面向任务语义的描述句。

示例：

```text
the robot grasps a cup and places it onto the shelf
```

要求：

- 只保留任务核心动作和对象关系。
- 不追求长文本，不生成段落。
- 尽量避免自由发挥和额外推断。

### 4.2 capability_tags

定义：

- 从 taxonomy 中选择出的多标签结果。

示例：

```json
["grasp", "transport", "place"]
```

要求：

- 只能从 taxonomy 中选取。
- 支持多标签。
- 顺序不重要。

### 4.3 action_slots

定义：

- 对任务关键动作要素的结构化表示。

最小要求：

```json
{
  "object": "cup",
  "target": "shelf"
}
```

说明：

- `object` 必填。
- `target` 必填。
- 其他字段可以按后续需要扩展，例如 `verb`、`tool`、`phase_count`，但当前不作为必须项。

作用：

- 补足仅靠标签无法表达的对象和目标关系。
- 支持规则校验，例如描述中出现的对象和目标是否与槽位一致。
- 为后续对外接口或更细粒度检索预留结构化扩展位。

### 4.4 text_embedding

定义：

- 对 `task_description` 编码得到的向量。

作用：

- 支持基于自然语言描述的语义检索。
- 适合匹配表达更丰富的查询。

### 4.5 label_embedding

定义：

- 对规范标签文本编码得到的向量。

标签文本建议按模板拼接：

```text
capabilities: grasp, transport, place; object: cup; target: shelf
```

作用：

- 支持更稳定、更受控的能力匹配。
- 与 `text_embedding` 互补。

## 5. 模块结构建议

建议在当前目录下补齐：

```text
src/bise/modalities/semantic/
├── __init__.py
├── cache.py
├── prompts.py
├── schemas.py
├── vlm_client.py
├── parser.py
├── normalizer.py
├── embedder.py
├── pipeline.py
└── evaluator.py
```

各文件职责：

| 文件 | 职责 |
| :--- | :--- |
| `vlm_client.py` | 调用本地服务或远程 API |
| `prompts.py` | 生成两套 prompt：描述提取 prompt、标签提取 prompt |
| `parser.py` | 解析 VLM 返回结果并提取结构化 JSON |
| `normalizer.py` | taxonomy 映射、标签清洗、槽位字段校验 |
| `embedder.py` | 生成 `text_embedding` 与 `label_embedding` |
| `pipeline.py` | 批量处理 manifest，串联调用、缓存、落盘 |
| `evaluator.py` | 标签评估、轻量描述人工评审汇总、检索评估 |
| `schemas.py` | 语义结果与评估记录结构定义 |

## 6. 数据组织与管理

### 6.1 目录建议

```text
artifacts/semantic/
├── manifests/
│   └── semantic_manifest_v1.jsonl
├── annotations/
│   ├── raw_responses.jsonl
│   └── normalized_annotations.jsonl
├── embeddings/
│   ├── text_embedding_v1.npy
│   ├── label_embedding_v1.npy
│   └── sample_ids_v1.json
├── eval/
│   ├── label_gold_set_v1.jsonl
│   ├── description_review_set_v1.jsonl
│   └── retrieval_queries_v1.jsonl
└── errors/
    ├── parse_failures.jsonl
    └── failed_samples.jsonl
```

### 6.2 三层数据

建议保持三层分离：

| 层 | 内容 |
| :--- | :--- |
| manifest 层 | 待处理样本清单 |
| annotation 层 | VLM 原始结果和标准化结果 |
| embedding 层 | `text_embedding`、`label_embedding` 和 sample ID 映射 |

### 6.3 与现有检索结构的对接

最终写回 `EmbeddingSample`：

- `text_embedding`
- `label_embedding`
- `metadata.semantic.task_description`
- `metadata.semantic.capability_tags`
- `metadata.semantic.action_slots`

这样可以直接接入仓库已有的 `FeatureStore` 和后续索引构建流程。

## 7. 语义提取层设计

### 7.1 输入形式

当前统一按单视频输入设计：

- 一次请求只处理一个 `robot_video`
- 或一次请求只处理一个 `human_video`

实现上通过 `video_role` 区分：

- `video_role=robot`
- `video_role=human`

### 7.2 Prompt 设计

当前改为两套 prompt：

1. 描述提取 prompt
2. 标签提取 prompt

两套 prompt 的职责分别是：

- 描述提取 prompt：只输出 `task_description`
- 标签提取 prompt：输出 `capability_tags` 与 `action_slots`

taxonomy 约束只作用于标签提取 prompt。

两套 prompt 都必须是单视频表述，不能再使用双视频联合输入的文案。

因此本阶段的 prompt 组合为：

- `description_prompt_v1`
- `label_prompt_with_taxonomy_v1`
- `label_prompt_without_taxonomy_v1`

### 7.3 结果标准化

标准化步骤建议为：

1. 调用描述提取 prompt，生成 `task_description`。
2. 调用标签提取 prompt，生成 `capability_tags` 和 `action_slots`。
3. 将 `capability_tags` 映射到 taxonomy。
4. 检查 `action_slots.object` 和 `action_slots.target` 是否存在。
5. 生成规范标签文本。
6. 生成 `text_embedding` 和 `label_embedding`。

### 7.4 批量处理

现阶段重点是现有数据集批量处理，建议 `pipeline.py` 支持：

- 从 manifest 读取待处理样本
- 以 `scene` 为处理单元，对每个 `scene` 只选一个 `cam` 对
- 基于同一个 `cam` 对生成 `robot` 和 `human` 两条单视频语义记录
- 跳过已处理样本
- 保存失败样本
- 断点续跑
- 缓存同一请求结果

对外统一接口暂不实现，但建议在 `pipeline.py` 之外额外预留一个服务层入口，例如未来可接 `service/semantic_api.py`。

## 8. 语义检索层设计

### 8.1 检索对象

语义检索的核心是对视频样本的语义结果做相似度匹配，而不是直接匹配原视频帧。

索引内容：

- `text_embedding`
- `label_embedding`

### 8.2 查询形式

当前实验中，查询文本主要来自模型生成的人类演示视频描述。

推荐主实验设置：

- 用 `video_role=human` 的 `task_description` 作为 query
- 用 `video_role=robot` 的语义索引作为 gallery
- 检验是否能匹配到对应任务视频

同时建议补一个对照：

- 用机器人视频描述检索机器人语义库

这样可以区分“跨域语义匹配”与“同域语义匹配”。

### 8.3 相似度与组合方式

建议比较三种语义检索方案：

1. 仅 `text_embedding`
2. 仅 `label_embedding`
3. `text_embedding + label_embedding`

组合方式可以先使用简单加权平均或加权和，不需要扩展到全模态融合。

### 8.4 向量数据库选型

当前阶段推荐直接使用 `FAISS`。

原因：

- 当前仓库已经有 `src/bise/retrieval/faiss_index.py`
- 语义检索目前以离线验证为主，不需要独立部署搜索服务
- `FAISS` 足以支持当前规模下的 dense vector 相似度搜索

当前阶段不建议做向量数据库对比实验。

原因：

- 你当前要验证的是语义生成质量和语义检索可行性，不是在线检索基础设施优劣
- 引入 `Qdrant` 或 `Milvus` 会增加部署和维护成本，但不会直接回答“语义层是否有效”
- 当后续需要对外提供统一接口或在线检索服务时，再评估是否迁移到独立向量数据库

## 9. 验证与实验设计

本阶段将“测试与验证”和“实验设计”合并，统一成一套从实现正确性到实验结论的闭环。

### 9.1 工程正确性验证

建议新增：

```text
tests/test_semantic_schemas.py
tests/test_semantic_parser.py
tests/test_semantic_normalizer.py
tests/test_semantic_pipeline.py
```

重点验证：

- schema 是否要求 `action_slots.object` 和 `action_slots.target`
- 描述 prompt 和标签 prompt 是否按预期渲染
- taxonomy 映射是否正确
- 解析失败样本是否进入错误池
- `scene -> cam` 采样策略是否只生成一次语义结果
- embedding 维度是否稳定

### 9.2 VLM 选型实验

目的：

- 比较不同 VLM 在当前单视频语义提取任务上的质量、成本和稳定性。

实验设置：

- 输入单元为“每个 `scene` 采样一个 `cam` 对，并拆成 `robot` / `human` 两条单视频记录”
- 使用同一套 description prompt 和 label prompt
- 使用同一 embedding 模型和同一 `FAISS` 检索后端

指标：

- 标签 `Precision`
- 标签 `Recall`
- 标签 `F1`
- 轻量描述人工评审通过率
- 平均时延
- 失败率

结论目标：

- 选出后续批量处理的主力 VLM

### 9.3 Prompt 消融实验

当前只对标签提取 prompt 做消融，不对描述提取 prompt 做消融。

只比较：

1. `label_prompt_with_taxonomy`
2. `label_prompt_without_taxonomy`

实验设置：

- VLM 固定
- embedding 模型固定
- 检索后端固定为 `FAISS`

指标：

- 标签 `Precision`
- 标签 `Recall`
- 标签 `F1`
- 可选 `Exact Match`
- `R@1`
- `MRR`

结论目标：

- 验证 taxonomy 约束是否提升标签质量及后续语义检索效果

### 9.4 描述质量验证

不做自动文本指标。

仅做轻量人工评审，样本量建议 30 到 50 条，人工判断三个问题：

1. 是否描述了主要动作
2. 是否描述了关键对象
3. 是否存在明显幻觉或错误目标

这部分只作为辅助质检，不作为主结果表核心指标。

### 9.5 语义检索消融实验

只比较：

1. 仅 `text_embedding`
2. 仅 `label_embedding`
3. `text_embedding + label_embedding`

评价方式：

- 查询为人类演示视频生成的 `task_description`
- 检索库为机器人视频语义索引
- 正确匹配目标为对应任务视频或对应任务组

指标：

- `R@1`
- `R@5`
- `R@10`
- `MRR`

说明：

- 当前不把不同向量数据库作为实验变量
- 当前不把不同 embedding 模型作为正式对比实验

补充建议：

- 如果语义检索结果明显不理想，再额外增加一个小规模 embedding 模型补充实验
- 但这不应成为本阶段必须完成的正式实验项

### 9.6 系统效率验证

建议保留：

- 单 `scene` 平均处理时延
- 批量吞吐量
- 缓存命中率
- 失败重试率
- 单 `scene` 平均成本

不做一致性与稳健性指标。

## 10. 实验结果呈现

建议最终实验材料收敛为四张主表。

表 1：VLM 选型结果

| 模型 | Precision | Recall | F1 | 人工描述通过率 | 平均时延 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Model A | TBD | TBD | TBD | TBD | TBD |
| Model B | TBD | TBD | TBD | TBD | TBD |

表 2：Prompt 消融结果

| Prompt 方案 | Precision | Recall | F1 | R@1 | MRR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| label prompt without taxonomy | TBD | TBD | TBD | TBD | TBD |
| label prompt with taxonomy | TBD | TBD | TBD | TBD | TBD |

表 3：语义检索消融结果

| 方案 | R@1 | R@5 | R@10 | MRR |
| :--- | :--- | :--- | :--- | :--- |
| text only | TBD | TBD | TBD | TBD |
| label only | TBD | TBD | TBD | TBD |
| text + label | TBD | TBD | TBD | TBD |

表 4：系统效率结果

| 方案 | 单 scene 时延 | 批量吞吐 | 缓存命中率 | 重试率 | 单 scene 成本 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Model A | TBD | TBD | TBD | TBD | TBD |

## 11. 开发排期建议

### 阶段 1：语义提取基础设施

目标：

- 完成 `schemas.py`、`vlm_client.py`、`parser.py`、`normalizer.py`
- 固定输出字段
- 固定 `action_slots.object` 和 `action_slots.target` 必填约束
- 实现描述 prompt 与标签 prompt 两套模板

### 阶段 2：数据集批量处理

目标：

- 建立 manifest
- 建立 `scene -> cam` 采样规则
- 跑通 RH20T/WHIRL 的批量语义提取
- 保存 annotation 与 embedding 结果

### 阶段 3：标签评估与 prompt 实验

目标：

- 构建几十条样本的人工标签集
- 完成 VLM 选型实验
- 完成 taxonomy 消融实验

### 阶段 4：语义检索验证

目标：

- 基于 `text_embedding` 和 `label_embedding` 建立语义索引
- 完成语义检索消融实验
- 汇总结果表

## 12. 对当前代码的直接建议

1. 扩展 `src/bise/modalities/semantic/schemas.py`，固定输出为 `task_description`、`capability_tags`、`action_slots`、`text_embedding`、`label_embedding`。
2. 将 `action_slots.object` 和 `action_slots.target` 设为必填。
3. 重写 `src/bise/modalities/semantic/prompts.py`，改为两套 prompt：描述提取 prompt 和标签提取 prompt；其中标签提取 prompt 支持 `with_taxonomy` / `without_taxonomy` 两个版本。
4. 重写 `src/bise/modalities/semantic/vlm_client.py`，支持本地部署模型或远程 API。
5. 新增 `normalizer.py` 和 `embedder.py`，将标签结果稳定写入 `label_embedding`。
6. 在 `pipeline.py` 中优先实现以 `scene` 为单位、按单个 `cam` 对采样的批量处理，不急于实现外部接口。
7. 在 `retrieval` 层先使用 `FAISS` 实现纯语义检索实验，不扩展到全模态融合，也不把向量数据库对比作为本阶段任务。

## 13. 最终交付物

建议本阶段最终交付如下：

| 类别 | 交付物 |
| :--- | :--- |
| 代码 | 语义提取层与语义检索层代码 |
| 配置 | taxonomy、prompt、provider 配置 |
| 数据 | manifest、人工标签集、检索查询集 |
| 结果 | annotation 文件、embedding 文件 |
| 评估 | 标签质量表、prompt 消融表、语义检索结果表、效率表 |

按以上范围推进，语义模块会更聚焦：先把“批量生成语义结果 + 基于语义做检索验证”完整跑通，再为后续统一接口与更复杂融合留扩展位。
