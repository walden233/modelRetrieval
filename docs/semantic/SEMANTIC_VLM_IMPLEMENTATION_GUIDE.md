# 语义层 VLM 模块开发实现文档

更新时间：2026-04-17

本文档用于指导语义层 VLM 模块的具体代码实现，直接面向当前仓库结构，不再重复宏观方案说明。范围以 [SEMANTIC_VLM_MODULE_PLAN.md](/home/ttt/BISE/docs/SEMANTIC_VLM_MODULE_PLAN.md) 为准，重点回答：

- 具体要改哪些文件
- 每个文件要实现什么接口
- 数据如何流转
- 先做什么，后做什么
- 每一阶段如何验收

## 1. 实现目标

本阶段只实现两部分：

1. 语义提取层
2. 语义检索层

单次输入：

- `robot_video`
- 或 `human_video`

输出：

- `task_description`
- `capability_tags`
- `action_slots`
- `text_embedding`
- `label_embedding`

处理粒度：

- 以 `scene` 为语义组织单元
- 每个 `scene` 只采样一个 `robot_cam` / `human_cam` 对
- 基于同一个 `cam` 对拆成两条单视频记录：
  - `video_role=robot`
  - `video_role=human`

## 2. 当前代码现状与改造目标

当前 `semantic` 目录还只是占位实现：

- [src/bise/modalities/semantic/schemas.py](/home/ttt/BISE/src/bise/modalities/semantic/schemas.py)
- [src/bise/modalities/semantic/prompts.py](/home/ttt/BISE/src/bise/modalities/semantic/prompts.py)
- [src/bise/modalities/semantic/vlm_client.py](/home/ttt/BISE/src/bise/modalities/semantic/vlm_client.py)
- [src/bise/modalities/semantic/cache.py](/home/ttt/BISE/src/bise/modalities/semantic/cache.py)

改造目标：

- 把占位结构补成完整可跑通的离线批处理模块
- 直接接入已有 `EmbeddingSample`、`FeatureStore`、`faiss_index.py`
- 先支持现有数据集离线批量处理
- 对外服务接口先不实现，只预留结构

## 3. 目标文件清单

### 3.1 需要新增的文件

```text
src/bise/modalities/semantic/
├── parser.py
├── normalizer.py
├── embedder.py
├── pipeline.py
└── sampler.py

tools/
├── build_semantic_manifest.py
├── run_semantic_annotation.py
├── evaluate_semantic.py
└── evaluate_semantic_retrieval.py

tests/
├── test_semantic_schemas.py
├── test_semantic_parser.py
├── test_semantic_normalizer.py
├── test_semantic_sampler.py
├── test_semantic_pipeline.py
└── test_semantic_retrieval.py
```

### 3.2 需要重写或扩展的文件

```text
src/bise/modalities/semantic/
├── schemas.py
├── prompts.py
├── vlm_client.py
└── cache.py
```

### 3.3 需要补充的配置

```text
configs/semantic/
├── provider_stub.json
├── provider_api.json
├── prompt_description_v1.json
├── prompt_label_with_taxonomy_v1.json
├── prompt_label_without_taxonomy_v1.json
├── taxonomy_v1.json
├── pipeline_v1.json
└── retrieval_v1.json
```

## 4. 模块分层与职责

### 4.1 `schemas.py`

负责定义语义模块所有核心数据结构。建议至少包含：

- `SemanticManifestRecord`
- `ActionSlots`
- `SemanticAnnotation`
- `SemanticEmbeddingRecord`
- `LabelEvaluationRecord`
- `DescriptionReviewRecord`

建议结构如下：

```python
@dataclass
class ActionSlots:
    object: str
    target: str
    verb: str = ""
    tool: str = ""


@dataclass
class SemanticAnnotation:
    sample_id: str
    pair_id: str
    task_id: str
    scene_id: str
    video_role: str
    video_path: str
    paired_video_path: str
    cam_id: str
    task_description: str
    capability_tags: list[str]
    action_slots: ActionSlots
    label_canonical_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

实现要求：

- `action_slots.object` 和 `action_slots.target` 必填
- 提供 `to_dict()` / `from_dict()` 能力
- 尽量与 [src/bise/common/schemas.py](/home/ttt/BISE/src/bise/common/schemas.py) 的 `EmbeddingSample` 对齐

### 4.2 `prompts.py`

负责构造两套 prompt：

1. 描述提取 prompt
2. 标签提取 prompt

建议暴露的接口：

```python
def build_description_prompt(...) -> str: ...
def build_label_prompt(..., use_taxonomy: bool) -> str: ...
```

要求：

- 描述 prompt 只要求输出 `task_description`
- 标签 prompt 只要求输出 `capability_tags` 和 `action_slots`
- `with_taxonomy` 和 `without_taxonomy` 仅在标签 prompt 上切换
- prompt 模板不要硬编码长文本，优先从 `configs/semantic/*.json` 读取

### 4.3 `vlm_client.py`

负责统一封装 VLM 调用。

建议类设计：

```python
class VLMClient:
    def annotate_description(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def annotate_labels(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class StubVLMClient(VLMClient): ...
class OpenAICompatibleVLMClient(VLMClient): ...
```

要求：

- 支持描述提取和标签提取两类调用
- 输入 payload 中保留：
  - `video_path`
  - `video_role`
  - `frames`
  - `prompt`
  - `model`
- 不引入 LangChain
- 远程 API 调用失败时，抛出可分类异常，便于 pipeline 记录失败原因

### 4.4 `parser.py`

负责把 VLM 返回内容解析为结构化结果。

建议暴露：

```python
def parse_description_response(raw_response: dict[str, Any]) -> str: ...
def parse_label_response(raw_response: dict[str, Any]) -> dict[str, Any]: ...
```

要求：

- 兼容 code fence 包裹 JSON
- 兼容轻微格式错误
- 解析失败时返回明确错误类型
- 不在 parser 中做 taxonomy 映射

### 4.5 `normalizer.py`

负责标准化标签与槽位。

建议功能：

- `normalize_capability_tags()`
- `normalize_action_slots()`
- `build_label_canonical_text()`
- `validate_annotation()`

要求：

- taxonomy 映射、别名归一在这里完成
- `object` 和 `target` 空值直接判为非法结果
- `label_canonical_text` 统一格式，保证 `label_embedding` 输入稳定

### 4.6 `embedder.py`

负责把文本编码成向量。

建议类设计：

```python
class TextEmbedder:
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...
```

建议暴露：

- `build_text_embedding(task_description)`
- `build_label_embedding(label_canonical_text)`

要求：

- 同一个 embedding 模型同时处理描述文本和标签文本
- 批量编码优先，不做逐条编码
- 维度写入 metadata，便于后续检查

### 4.7 `sampler.py`

负责 `scene -> cam` 采样。

建议接口：

```python
def select_scene_camera_pair(scene_record, strategy_config) -> tuple[str, str]: ...
```

采样要求：

- 每个 `scene` 只选一个 `robot_cam`
- 每个 `scene` 只选一个 `human_cam`
- sampler 输出的是一个 `cam` 对；真正送给 VLM 时仍然是单视频
- 策略先做简单、可复现版本，例如：
  - 优先固定 cam id
  - 若不存在则按排序取第一个

### 4.8 `pipeline.py`

负责把 manifest、采样、VLM、解析、标准化、embedding、落盘串起来。

建议核心入口：

```python
def run_semantic_annotation_pipeline(config: dict[str, Any]) -> None: ...
```

单条样本处理流程：

1. 读取 manifest record
2. 根据 `video_role` 读取对应单视频
3. 生成描述 prompt
4. 调用描述 VLM
5. 解析该单视频的 `task_description`
6. 生成标签 prompt
7. 调用标签 VLM
8. 解析该单视频的 `capability_tags` 和 `action_slots`
9. 做标准化和合法性校验
10. 构造 `label_canonical_text`
11. 生成 `text_embedding`
12. 生成 `label_embedding`
13. 写 annotation
14. 写 embedding 结果
15. 更新 manifest status

## 5. 数据流设计

### 5.1 Manifest 层

建议文件：

- `artifacts/semantic/manifests/semantic_manifest_v1.jsonl`

每条记录至少包含：

```json
{
  "sample_id": "rh20t_task_001_scene_003_robot",
  "pair_id": "rh20t_task_001_scene_003",
  "task_id": "task_001",
  "scene_id": "scene_003",
  "dataset_name": "RH20T",
  "video_role": "robot",
  "cam_id": "cam_0",
  "video_path": "...",
  "paired_video_path": "...",
  "description_prompt_version": "description_prompt_v1",
  "label_prompt_version": "label_prompt_with_taxonomy_v1",
  "taxonomy_version": "taxonomy_v1",
  "status": "pending"
}
```

### 5.2 Annotation 层

建议拆成两类文件：

- `raw_responses.jsonl`
- `normalized_annotations.jsonl`

原始响应记录：

- provider
- model
- prompt version
- request id
- raw text / raw json
- latency
- error message

标准化结果记录：

- `task_description`
- `capability_tags`
- `action_slots`
- `label_canonical_text`
- `status`

### 5.3 Embedding 层

建议文件：

- `text_embedding_v1.npy`
- `label_embedding_v1.npy`
- `sample_ids_v1.json`

要求：

- `sample_ids_v1.json` 与 embedding 行顺序一一对应
- embedding 层生成完成后，同时写入 `EmbeddingSample`

## 6. 与现有检索层的对接

当前仓库已经有：

- [src/bise/common/schemas.py](/home/ttt/BISE/src/bise/common/schemas.py)
- [src/bise/retrieval/feature_store.py](/home/ttt/BISE/src/bise/retrieval/feature_store.py)
- [src/bise/retrieval/faiss_index.py](/home/ttt/BISE/src/bise/retrieval/faiss_index.py)
- [src/bise/retrieval/metrics.py](/home/ttt/BISE/src/bise/retrieval/metrics.py)

落地方式：

1. 把语义结果转成 `EmbeddingSample`
2. 填充：
   - `text_embedding`
   - `label_embedding`
   - `metadata.semantic`
3. 用 `FeatureStore.save()` 保存
4. 用 `faiss_index.py` 建语义索引
5. 用 `metrics.py` 做 `R@1 / R@5 / R@10 / MRR`

建议后续工具分工：

- `evaluate_semantic.py`
  - 标签质量评估
  - 描述人工评审汇总
- `evaluate_semantic_retrieval.py`
  - 纯语义检索评估

## 7. 配置设计

### 7.1 `provider_api.json`

建议字段：

```json
{
  "provider_name": "openai_compatible",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "VLM_API_KEY",
  "model_name": "Qwen2.5-VL-7B-Instruct",
  "timeout_seconds": 120,
  "max_retries": 3
}
```

### 7.2 `pipeline_v1.json`

建议字段：

```json
{
  "dataset_name": "RH20T",
  "manifest_path": "artifacts/semantic/manifests/semantic_manifest_v1.jsonl",
  "raw_response_path": "artifacts/semantic/annotations/raw_responses.jsonl",
  "normalized_annotation_path": "artifacts/semantic/annotations/normalized_annotations.jsonl",
  "sample_ids_path": "artifacts/semantic/embeddings/sample_ids_v1.json",
  "text_embedding_path": "artifacts/semantic/embeddings/text_embedding_v1.npy",
  "label_embedding_path": "artifacts/semantic/embeddings/label_embedding_v1.npy",
  "cache_path": "artifacts/semantic/cache/semantic_cache.json",
  "camera_pair_strategy": "fixed_or_first",
  "preferred_robot_cam_id": "cam_0",
  "preferred_human_cam_id": "cam_0",
  "batch_size": 8,
  "skip_completed": true
}
```

### 7.3 `taxonomy_v1.json`

建议字段：

```json
{
  "version": "taxonomy_v1",
  "allowed_tags": ["grasp", "transport", "place", "insert", "rotate", "pour"],
  "tag_aliases": {
    "pick": "grasp",
    "pick_up": "grasp",
    "put": "place"
  }
}
```

## 8. 缓存与幂等设计

`cache.py` 需要从单纯的 JSON KV 扩展成“可区分请求类型”的缓存。

建议缓存键：

```text
sha256(
  sample_id +
  video_path +
  paired_video_path +
  video_role +
  cam_id +
  model_name +
  prompt_version +
  taxonomy_version +
  request_type
)
```

其中 `request_type` 取值：

- `description`
- `labels`

要求：

- 相同样本、相同 prompt、相同模型重复运行时命中缓存
- 描述请求和标签请求分开缓存
- 失败结果不要直接作为成功缓存写回

## 9. 工具脚本设计

### 9.1 `build_semantic_manifest.py`

职责：

- 扫描数据集
- 以 `scene` 为单元生成 manifest
- 为每条记录填默认 prompt / taxonomy 版本

### 9.2 `run_semantic_annotation.py`

职责：

- 加载 provider、prompt、taxonomy、pipeline 配置
- 执行语义提取全流程
- 写 annotation 和 embedding

### 9.3 `evaluate_semantic.py`

职责：

- 加载人工标签集
- 计算 `Precision / Recall / F1`
- 汇总描述人工评审结果

### 9.4 `evaluate_semantic_retrieval.py`

职责：

- 构建或加载语义索引
- 运行 `text only` / `label only` / `text + label`
- 输出检索指标表

## 10. 测试计划

### 10.1 单元测试

`test_semantic_schemas.py`

- `ActionSlots.object` 为空时报错
- `ActionSlots.target` 为空时报错
- `SemanticAnnotation.to_dict()` 正确

`test_semantic_parser.py`

- 纯 JSON 能解析
- code fence JSON 能解析
- 非法 JSON 抛出解析异常

`test_semantic_normalizer.py`

- `pick` 能映射成 `grasp`
- 空标签被剔除
- 缺失 `object/target` 被识别为非法

`test_semantic_sampler.py`

- 固定 cam 存在时按固定值选
- 固定 cam 不存在时回退到排序第一个

`test_semantic_pipeline.py`

- 单条 manifest 能跑通
- 重跑命中缓存
- 失败样本进入错误文件

`test_semantic_retrieval.py`

- `text only`
- `label only`
- `text + label`

## 11. 开发顺序

### 阶段 1：先补 schema 和 prompt

先做：

- `schemas.py`
- `prompts.py`
- `taxonomy_v1.json`

验收标准：

- 可以稳定构造描述 prompt 和标签 prompt
- `ActionSlots` 必填校验清晰

### 阶段 2：补 VLM 调用和解析链路

再做：

- `vlm_client.py`
- `parser.py`
- `normalizer.py`

验收标准：

- 用 stub provider 能返回结构化结果
- `with_taxonomy` / `without_taxonomy` 标签链路都能跑通

### 阶段 3：补 scene 采样和批处理

再做：

- `sampler.py`
- `build_semantic_manifest.py`
- `pipeline.py`

验收标准：

- RH20T 或 WHIRL 小样本能完成 `scene -> cam` 采样
- 单次运行能产出 annotation 文件

### 阶段 4：补 embedding 和检索

再做：

- `embedder.py`
- `evaluate_semantic_retrieval.py`

验收标准：

- 能生成 `text_embedding` 和 `label_embedding`
- 能用 FAISS 跑纯语义检索

### 阶段 5：补评估脚本

最后做：

- `evaluate_semantic.py`
- 描述人工评审汇总逻辑

验收标准：

- 能产出标签质量表
- 能产出语义检索结果表

## 12. 实施约束

实现过程中需要坚持以下约束：

- 不把语义生成粒度做到每个相机视频
- 不引入 LangChain
- 不在本阶段实现外部服务接口
- 不在本阶段做 embedding 模型对比
- 不在本阶段做向量数据库对比
- 不扩展到视频/轨迹/文本/标签四模态融合

## 13. 本阶段完成定义

满足以下条件，即可认为本阶段开发完成：

1. 可从现有数据集生成 `scene` 级 manifest
2. 可对每个 `scene` 采样一个 `cam` 对并拆成 `robot` / `human` 两条单视频语义记录
3. 可产出 `task_description`、`capability_tags`、`action_slots`
4. 可产出 `text_embedding` 与 `label_embedding`
5. 可将结果写入 `EmbeddingSample`
6. 可基于 `FAISS` 完成纯语义检索评估
7. 可输出标签质量表和语义检索结果表

## 14. 建议的首批实现任务

如果按最小阻塞路径推进，建议直接按下面顺序开工：

1. 重写 [src/bise/modalities/semantic/schemas.py](/home/ttt/BISE/src/bise/modalities/semantic/schemas.py)
2. 重写 [src/bise/modalities/semantic/prompts.py](/home/ttt/BISE/src/bise/modalities/semantic/prompts.py)
3. 新增 `configs/semantic/taxonomy_v1.json`
4. 重写 [src/bise/modalities/semantic/vlm_client.py](/home/ttt/BISE/src/bise/modalities/semantic/vlm_client.py)
5. 新增 `parser.py` 与 `normalizer.py`
6. 新增 `sampler.py` 与 `build_semantic_manifest.py`
7. 新增 `pipeline.py`
8. 新增 `embedder.py`
9. 新增评估脚本与测试

这份文档的目标不是解释“为什么要做语义模块”，而是让开发时可以直接对照文件、接口和阶段任务落代码。
