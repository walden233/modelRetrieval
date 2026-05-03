# 三模态统一检索系统设计与实验计划

更新时间：2026-05-03

本文档重新按“真实检索系统”定义统一检索，而不是把视频、轨迹、语义三个单模态 `final_test` 评估结果简单相加。

当前实现状态：

- 已新增检索系统核心模块：`src/bise/retrieval/system/`。
- 已新增建库脚本：`runs/retrieval/build_retrieval_library.py`。
- 已新增单条 query 脚本：`runs/retrieval/query_retrieval_system.py`。
- 已新增系统评估脚本：`runs/retrieval/evaluate_retrieval_system.py`。
- 已新增图表导出脚本：`runs/retrieval/export_retrieval_system_charts.py`。
- 已新增默认配置：`configs/retrieval/*.json`。
- 已新增常用入口脚本：`scripts/build_retrieval_library.sh`、`scripts/query_retrieval_system.sh`、`scripts/evaluate_retrieval_system.sh`、`scripts/export_retrieval_system_charts.sh`。

系统典型输入是一条人类侧 query，可包含以下任意一种或多种信息：

- 单个人类视频。
- 单个人类轨迹。
- 单个人类语义特征。

三者都允许缺失，但至少要有一个模态。系统输出是机器人侧候选库中的 Top-K 结果。离线实验和评估都围绕这个系统本身展开。

## 1. 核心结论

当前设计方向是合理的，但必须做几处更正：

- 统一检索系统必须和 `<best_video_run>/final_test`、`<best_trajectory_run>/final_test` 解耦。`final_test` 是单模态评估产物，不应作为系统检索库。
- 需要新增“检索库构建脚本”，直接根据指定模型、checkpoint、数据配置和语义产物抽取特征，写入 `artifacts/retrieval/...`。
- 检索库的生产形态应以机器人侧 gallery 特征为主；实验阶段可以同时保存 human/robot 两侧特征，human 侧只用于自动评估。
- `level=scene/task/mixed` 不应把真实标签注入检索排序。它应该定义评估相关性：同 scene 分最高，同 task 但不同 scene 分较低，不同 task 为 0。
- 第一版检索排序使用 late fusion：各模态分别算 query-to-gallery 相似度，再做校准和加权融合。
- 实验主集可以先限制在视频、轨迹、语义都完整的 pair 上，后续再做缺失模态消融和真实系统评估。

## 2. 系统边界

### 2.1 离线建库

离线建库负责把某个 RH20T cfg 数据集中的样本编码成检索库。

输入：

- `cfg<n>`：例如 cfg2 / cfg3。
- `scenes_per_task`：每个 task 取多少 scene，`<=0` 表示全部。
- `cameras_per_scene`：每个 scene 取多少 camera，`<=0` 表示全部。
- 视频模型 config 和 checkpoint。
- 轨迹模型 config 和 checkpoint。
- 语义特征根目录，例如 `artifacts/semantic/rh20t`。
- 输出目录，例如 `artifacts/retrieval/rh20t_cfg2_v1`。

输出：

- 机器人侧视频、轨迹、语义特征。
- 可选的人类侧视频、轨迹、语义特征，用于评估。
- 统一 manifest、feature index、coverage、构建配置和日志。

### 2.2 在线查询

在线查询负责把单条人类输入编码为 query feature，并到机器人 gallery 中检索。

输入可以是：

```text
video_path: optional
trajectory_path or trajectory array: optional
semantic_feature: optional
```

输出：

```text
top_k robot candidates
每个候选的 fused_score
每个候选的 modality_scores
每个候选的 task_id / scene_id / camera_id / source_path
```

### 2.3 离线评估

离线评估模拟在线查询：

- 从检索库中的 human 侧样本逐条取 query。
- gallery 只使用 robot 侧样本。
- 支持只评估三模态完整样本，也支持缺失模态场景。

## 3. 数据组织设计

建议检索库目录：

```text
artifacts/retrieval/rh20t_cfg2_v1/
├── library_config.json
├── build_info.json
├── coverage.json
├── manifests/
│   ├── scenes.jsonl
│   ├── feature_records.jsonl
│   ├── gallery_robot.jsonl
│   └── query_human_eval.jsonl
├── features/
│   ├── video_robot.npy
│   ├── video_human.npy
│   ├── trajectory_robot.npy
│   ├── trajectory_human.npy
│   ├── semantic_text_robot.npy
│   ├── semantic_text_human.npy
│   ├── semantic_label_robot.npy
│   ├── semantic_label_human.npy
│   ├── semantic_combined_robot.npy
│   └── semantic_combined_human.npy
├── indices/
│   ├── video_robot.faiss
│   ├── trajectory_robot.faiss
│   └── semantic_text_robot.faiss
└── eval/
    ├── complete_pairs.json
    └── split_manifest.json
```

第一版可以先不建 FAISS，直接矩阵检索；但目录结构要为 FAISS 保留位置。

### 3.1 `scenes.jsonl`

一行表示一个 scene 级实体：

```json
{
  "entity_key": "rh20t::cfg2::task_0001::task_0001/scene_1",
  "dataset_name": "rh20t",
  "cfg": "cfg2",
  "task_id": "task_0001",
  "scene_id": "task_0001/scene_1",
  "scene_name": "scene_1",
  "scene_path": "dataset/RH20T_subset/RH20T_cfg2/task_0001/scene_1",
  "camera_ids": ["037522062165", "104122061850"],
  "available_modalities": {
    "video": true,
    "trajectory": true,
    "semantic_text": true,
    "semantic_label": true,
    "semantic_combined": true
  }
}
```

### 3.2 `feature_records.jsonl`

一行表示一个具体特征向量。视频和轨迹可以是 camera-level，语义通常是 scene-level 或 selected-camera。

```json
{
  "feature_id": "rh20t::cfg2::task_0001::scene_1::037522062165::robot::video",
  "entity_key": "rh20t::cfg2::task_0001::task_0001/scene_1",
  "domain": "robot",
  "modality": "video",
  "array_path": "features/video_robot.npy",
  "row_index": 0,
  "task_id": "task_0001",
  "scene_id": "task_0001/scene_1",
  "camera_id": "037522062165",
  "source_path": "dataset/RH20T_subset/RH20T_cfg2/task_0001/scene_1/037522062165_robot.mp4",
  "metadata": {}
}
```

### 3.3 `gallery_robot.jsonl`

生产检索库只需要 robot gallery。每行可以对应 scene，也可以对应 scene-camera。推荐第一版用 scene-camera 保存、scene-level 评估时再聚合。

```json
{
  "gallery_id": "rh20t::cfg2::task_0001::scene_1::037522062165::robot",
  "entity_key": "rh20t::cfg2::task_0001::task_0001/scene_1",
  "domain": "robot",
  "task_id": "task_0001",
  "scene_id": "task_0001/scene_1",
  "camera_id": "037522062165",
  "feature_ids": {
    "video": "...",
    "trajectory": "...",
    "semantic_text": "...",
    "semantic_label": "...",
    "semantic_combined": "..."
  }
}
```

### 3.4 `query_human_eval.jsonl`

只用于离线评估。真实在线系统不会要求 query 已经在库里。

```json
{
  "query_id": "rh20t::cfg2::task_0001::scene_1::037522062165::human",
  "entity_key": "rh20t::cfg2::task_0001::task_0001/scene_1",
  "domain": "human",
  "task_id": "task_0001",
  "scene_id": "task_0001/scene_1",
  "camera_id": "037522062165",
  "feature_ids": {
    "video": "...",
    "trajectory": "...",
    "semantic_text": "..."
  }
}
```

## 4. Key 与粒度规范

### 4.1 标准 key

必须统一：

```text
entity_key = <dataset>::<cfg>::<task_id>::<task_id>/<scene_name>
```

示例：

```text
rh20t::cfg2::task_0001::task_0001/scene_1
```

原因：

- `scene_1` 会在不同 task 下重复，不能单独作为 scene key。
- cfg2/cfg3 可能存在同名 task/scene，必须带 cfg。

### 4.2 camera_id 规范

统一去掉 `cam_` 前缀：

```text
cam_037522062165 -> 037522062165
037522062165 -> 037522062165
```

视频文件路径中可保留原名，但 manifest 中统一保存 normalized camera id。

### 4.3 scene-level 与 camera-level

特征保存推荐 camera-level，检索评价推荐 scene-level。

原因：

- 视频和轨迹存在多 camera。
- 语义通常不是全 camera 标注。
- scene-level 可以先建立稳定系统，camera-level 后续作为扩展。

scene-level 聚合方式：

| 聚合 | 含义 | 第一版用途 |
|---|---|---|
| `max` | 同一 scene 下任一 camera 命中即可 | 主设置 |
| `mean` | 多 camera 平均 | 辅助对照 |
| `first` | 固定第一个 camera | debug |

## 5. 建库脚本设计

新增脚本：

```text
runs/retrieval/build_retrieval_library.py
```

建议命令：

```bash
python runs/retrieval/build_retrieval_library.py \
  --cfg 2 \
  --data-root dataset/RH20T_subset/RH20T_cfg2 \
  --output-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --scenes-per-task 0 \
  --cameras-per-scene 2 \
  --video-config configs/video/vjepa_rh20t_baseline.json \
  --video-checkpoint artifacts/runs/video/<best_video_run>/best_model.pth \
  --trajectory-config configs/trajectory/<best_trajectory_config>.json \
  --trajectory-checkpoint artifacts/runs/trajectory/<best_trajectory_run>/best_model.pth \
  --semantic-root artifacts/semantic/rh20t \
  --semantic-cfg cfg2 \
  --include-human \
  --include-robot
```

参数说明：

| 参数 | 含义 |
|---|---|
| `--cfg` | RH20T cfg 编号，例如 2 / 3。 |
| `--data-root` | 数据根目录。 |
| `--output-dir` | 检索库输出目录。 |
| `--scenes-per-task` | 每个 task 取多少 scene，`0` 或负数表示全部。 |
| `--cameras-per-scene` | 每个 scene 取多少 camera，`0` 或负数表示全部。 |
| `--video-config` | 视频编码器 config。 |
| `--video-checkpoint` | 视频编码器 checkpoint。 |
| `--trajectory-config` | 轨迹编码器 config。 |
| `--trajectory-checkpoint` | 轨迹编码器 checkpoint。 |
| `--semantic-root` | 语义产物根目录。 |
| `--semantic-cfg` | 语义 cfg 子目录名，例如 `cfg2`。 |
| `--include-human` | 是否抽取 human 特征。实验需要；生产可关。 |
| `--include-robot` | 是否抽取 robot 特征。检索库必须开。 |
| `--overwrite` | 是否覆盖已有检索库。 |

### 5.1 视频特征提取

不能复用 `<best_video_run>/final_test`，应直接：

1. 构建视频模型并加载 checkpoint。
2. 扫描 `data_root`。
3. 按 `scenes_per_task` 和 `cameras_per_scene` 过滤。
4. 对 human/robot 视频分别编码。
5. 保存到 `features/video_human.npy` 和 `features/video_robot.npy`。
6. 写入 `feature_records.jsonl`。

需要注意：

- 现有视频模型 `forward(human_videos, robot_videos)` 是成对编码接口。建库阶段建议补 `encode_video(pixel_values, domain)` 或 `encode_human/encode_robot` 方法，避免为了编码 robot 还必须提供 human。
- 如果模型是 `shared`，domain 参数可忽略。
- 如果模型是 `dual_head` 或 `dual_encoder`，必须按 domain 走对应分支，否则线上 query 与 gallery 空间不一致。

### 5.2 轨迹特征提取

轨迹建库同样不能依赖 `final_test`。

流程：

1. 构建轨迹模型并加载 checkpoint。
2. 扫描 `human_pose.npy` 和 `tcp_base.npy`。
3. 对 human pose 编码为 human query 特征。
4. 对 robot tcp 编码为 robot gallery 特征。
5. 按 camera 写入 `trajectory_human.npy` 和 `trajectory_robot.npy`。

需要补的方法：

```python
model.encode_human(human_poses, human_mask)
model.encode_robot(tcp_bases, tcp_mask)
```

如果现有 `forward` 只能同时返回两侧 embedding，建库脚本也可以先批量构造 pair 后取两侧输出，但长期建议补独立 encode API。

### 5.3 语义特征导入

语义不重新调用 VLM。直接从已有产物导入：

```text
artifacts/semantic/rh20t/<cfg>/annotations/normalized_annotations.jsonl
artifacts/semantic/rh20t/<cfg>/feature_store/semantic_features.json
artifacts/semantic/rh20t/<cfg>/evaluation/pair/semantic_embeddings.npz
```

推荐优先读取 `normalized_annotations.jsonl` 或 feature store 中的：

- `text_embedding`
- `label_embedding`
- `combined = normalize(text_embedding + label_embedding)`

注意：

- 旧语义产物中 `scene_id` 可能是 `scene_1`，导入时必须归一为 `task_id/scene_id`。
- 语义特征通常只有 selected camera。保存时可以令 `camera_id=null` 或 `camera_id=<selected_cam>`，检索时按 scene 共享给该 scene 的所有 gallery candidate。
- 如果某 scene 没有语义特征，不应阻断建库，只在 coverage 中标记缺失。

## 6. 在线检索系统设计

新增核心模块：

```text
src/bise/retrieval/system/
├── __init__.py
├── schemas.py
├── library.py
├── encoders.py
├── scoring.py
├── query.py
├── evaluator.py
└── figures.py
```

### 6.1 Query 输入结构

```python
@dataclass
class RetrievalQuery:
    query_id: str | None = None
    video_path: str | None = None
    trajectory: Any | None = None
    semantic_text_embedding: list[float] | None = None
    semantic_label_embedding: list[float] | None = None
    semantic_combined_embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

语义 query 第一版只接收已编码特征，不在检索时调用 VLM。这样系统稳定、可复现，也避免评估时受 API 波动影响。

### 6.2 检索配置

示例：

```json
{
  "library_dir": "artifacts/retrieval/rh20t_cfg2_v1",
  "gallery_domain": "robot",
  "candidate_unit": "scene_camera",
  "level": "mixed",
  "scene_task_mixed": {
    "scene_gain": 1.0,
    "task_gain": 0.3
  },
  "modalities": ["video", "trajectory", "semantic_text"],
  "fusion": {
    "method": "weighted_sum",
    "calibration": "zscore",
    "missing_policy": "renormalize",
    "weights": {
      "video": 0.5,
      "trajectory": 0.2,
      "semantic_text": 0.3
    }
  },
  "aggregation": {
    "camera_to_scene": "max"
  },
  "top_k": 10
}
```

### 6.3 缺失模态处理

query 缺失某个模态时：

```text
available_modalities = query_modalities ∩ candidate_modalities ∩ enabled_modalities
```

融合分数：

```text
score(q, g) = sum_{m in available} w_m' * sim_m(q, g)
w_m' = w_m / sum_{m in available} w_m
```

如果某个候选在所有 query 可用模态上都没有特征，则该候选不可排序或分数为 `-inf`。

### 6.4 分数校准

推荐 query-wise 校准：

- `zscore`：主设置。
- `minmax`：辅助对照。
- `rank` / `rrf`：鲁棒性对照。
- `none`：只用于 debug。

校准只对当前 query 的 gallery 分数做，不依赖测试标签。

### 6.5 `level` 的正确含义

`level` 用于评估，不用于向检索排序注入真实标签。

支持：

| level | relevance 定义 | 指标 |
|---|---|---|
| `scene` | 同 `scene_id` 为正样本 | R@K / MRR / NDCG@10 |
| `task` | 同 `task_id` 为正样本 | R@K / MRR / NDCG@10 |
| `mixed` | 同 scene gain=1，同 task gain=0.3，否则 0 | graded NDCG@10 / WeightedHit@K |

`mixed` 设计：

```text
rel(q, g) = 1.0  if same scene
rel(q, g) = 0.3  if same task but different scene
rel(q, g) = 0.0  otherwise
```

可以额外报告：

```text
SceneHit@K
TaskHit@K
MixedNDCG@10
```

这样能体现“命中 scene 更好，只命中 task 也有部分价值”，同时不污染检索排序。

## 7. 需要补充的代码

### 7.1 建库相关

新增：

```text
runs/retrieval/build_retrieval_library.py
src/bise/retrieval/system/library.py
src/bise/retrieval/system/schemas.py
src/bise/retrieval/system/encoders.py
```

职责：

- 扫描 RH20T cfg 数据。
- 根据 `scenes_per_task`、`cameras_per_scene` 采样。
- 调用视频模型编码 human/robot 视频。
- 调用轨迹模型编码 human/robot 轨迹。
- 导入已有语义特征。
- 写 `features/*.npy` 和 `manifests/*.jsonl`。
- 生成 `coverage.json`。

### 7.2 查询相关

新增：

```text
runs/retrieval/query_retrieval_system.py
src/bise/retrieval/system/query.py
src/bise/retrieval/system/scoring.py
```

职责：

- 加载检索库。
- 根据 query 的可用模态编码或读取特征。
- 对 robot gallery 打分。
- 融合排序。
- 输出 Top-K JSON。

命令示例：

```bash
python runs/retrieval/query_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --video-path /path/to/human.mp4 \
  --semantic-feature /path/to/query_semantic.json \
  --config configs/retrieval/system_default.json \
  --top-k 10
```

### 7.3 评估相关

新增：

```text
runs/retrieval/evaluate_retrieval_system.py
src/bise/retrieval/system/evaluator.py
```

职责：

- 从 `query_human_eval.jsonl` 读取 human query。
- 支持 `--require-modalities video,trajectory,semantic_text`，用于只评估完整 pair。
- 支持 `--drop-modality` 或 `--enabled-modalities` 做消融。
- 使用 robot gallery 检索。
- 按 `scene/task/mixed` 计算指标。
- 导出 `metrics.json`、`cases.json`、`per_query_results.jsonl`。

命令示例：

```bash
python runs/retrieval/evaluate_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/system_default.json \
  --level mixed \
  --require-modalities video,trajectory,semantic_text \
  --output-dir artifacts/retrieval/rh20t_cfg2_v1/eval/full_modalities_mixed
```

### 7.4 图表相关

新增：

```text
runs/retrieval/export_retrieval_system_charts.py
src/bise/retrieval/system/figures.py
```

输出：

- `system_metrics_comparison.png`
- `system_metrics_comparison_data.json`
- `modality_coverage.png`
- `fusion_weights.png`
- `missing_modality_ablation.png`

### 7.5 单模态模型 API 需要补强

视频模型：

```python
encode_video(pixel_values, domain: Literal["human", "robot"])
```

轨迹模型：

```python
encode_human(human_poses, human_mask)
encode_robot(tcp_bases, tcp_mask)
```

这样建库和在线查询可以只编码单侧输入，不必伪造成 pair。

## 8. 配置文件建议

新增：

```text
configs/retrieval/library_rh20t_cfg2.json
configs/retrieval/library_rh20t_cfg3.json
configs/retrieval/system_default.json
configs/retrieval/system_eval_full_modalities.json
```

`library_rh20t_cfg2.json` 示例：

```json
{
  "cfg": "cfg2",
  "dataset_name": "rh20t",
  "data_root": "dataset/RH20T_subset/RH20T_cfg2",
  "output_dir": "artifacts/retrieval/rh20t_cfg2_v1",
  "sampling": {
    "scenes_per_task": 0,
    "cameras_per_scene": 2
  },
  "video": {
    "enabled": true,
    "config": "configs/video/vjepa_rh20t_baseline.json",
    "checkpoint": "artifacts/runs/video/<best_video_run>/best_model.pth",
    "batch_size": 4
  },
  "trajectory": {
    "enabled": true,
    "config": "configs/trajectory/<best>.json",
    "checkpoint": "artifacts/runs/trajectory/<best_run>/best_model.pth",
    "batch_size": 16
  },
  "semantic": {
    "enabled": true,
    "root": "artifacts/semantic/rh20t",
    "cfg": "cfg2",
    "modes": ["text", "label", "combined"]
  },
  "domains": {
    "include_human": true,
    "include_robot": true
  }
}
```

## 9. 实验设计

实验对象是“检索系统”，不是单模态模型本身。

### S0：检索库构建与覆盖检查

目标：

- 验证建库脚本能从指定 cfg、模型和语义产物独立生成检索库。
- 统计 video / trajectory / semantic 覆盖情况。

输入：

```text
cfg2
scenes_per_task=0
cameras_per_scene=2
include_human=true
include_robot=true
```

产出：

- `coverage.json`
- `modality_coverage.png`
- `feature_records.jsonl`

必须检查：

- robot gallery 数量。
- human query eval 数量。
- 三模态完整 pair 数量。
- 每个模态缺失原因。

### S1：完整三模态 query 的系统主实验

目标：

- 先在 video/trajectory/semantic 都完整的 human query 上评估系统。

设置：

```text
require_modalities = video, trajectory, semantic_text
gallery_domain = robot
enabled_modalities = video, trajectory, semantic_text
fusion = zscore + weighted_sum
level = scene / task / mixed
```

产出：

- `metrics.json`
- `cases.json`
- `system_metrics_comparison.png`

主报告：

```text
level=mixed
SceneHit@1 / SceneHit@5 / TaskHit@5 / MRR_scene / MixedNDCG@10
```

同时报告标准 scene-level：

```text
R@1 / R@5 / R@10 / MRR / NDCG@10
```

### S2：系统内单模态输入能力

目标：

- 验证系统在只有单个输入模态时也能工作。

实验项：

| ID | Query 可用模态 | Gallery 用同模态 | 目的 |
|---|---|---|---|
| S2-V | video | video | 视频输入检索 |
| S2-T | trajectory | trajectory | 轨迹输入检索 |
| S2-S | semantic_text | semantic_text | 语义输入检索 |

注意：

- 这些不是旧单模态 `final_test` 指标，而是在统一检索库和统一 query 集上重新跑出的系统指标。

### S3：多模态融合消融

目标：

- 比较哪些模态组合真正有收益。

实验项：

| ID | enabled_modalities |
|---|---|
| S3-VS | video + semantic_text |
| S3-VT | video + trajectory |
| S3-TS | trajectory + semantic_text |
| S3-VTS | video + trajectory + semantic_text |

权重：

- 如果有 validation split，权重只在 val 上搜索。
- 如果暂时没有 val，先使用固定均匀权重做 smoke，不作为最终论文结论。

### S4：缺失模态鲁棒性

目标：

- 验证真实系统中 query 模态缺失时仍可检索。

设置：

```text
missing_policy = renormalize
```

实验项：

- 完整三模态 query。
- 去掉 video。
- 去掉 trajectory。
- 去掉 semantic。
- 随机保留每个 query 的 1 个模态。

产出：

- `missing_modality_ablation.png`
- 每种缺失条件的 `metrics.json`。

### S5：level 评价对照

目标：

- 分清系统到底是找到了同一 scene，还是只找到了同一 task。

实验项：

| level | 指标解释 |
|---|---|
| `scene` | 严格同 scene 才算命中 |
| `task` | 同 task 即算命中 |
| `mixed` | 同 scene 高分，同 task 低分 |

产出：

- `level_metrics_comparison.png`
- `level_metrics_comparison_data.json`

### S6：cfg3 检索库迁移

目标：

- 用同一套模型和 cfg2 确定的融合权重构建/评估 cfg3 检索库。

设置：

```text
cfg = cfg3
data_root = dataset/RH20T_subset/RH20T_cfg3
weights = cfg2 selected weights
不在 cfg3 重新调权
```

注意：

- 视频和轨迹可以直接用模型抽特征。
- 语义必须先有 `artifacts/semantic/rh20t/cfg3` 特征，否则 cfg3 三模态实验只能做 video/trajectory 或缺失语义模式。

### S7：真实单条 query smoke

目标：

- 用不在检索库中的单条 human video / trajectory / semantic feature 做一次查询。

产出：

- `query_result.json`
- Top-K 候选路径和单模态贡献分数。

这一步用于验证系统形态，不作为主要实验指标。

## 10. 指标设计

### 10.1 scene-level 标准指标

严格同 scene 为正样本：

- `R@1`
- `R@5`
- `R@10`
- `Mean Rank`
- `MRR`
- `Mean Percentage Rank`
- `NDCG@10`

### 10.2 task-level 标准指标

同 task 为正样本，适合观察语义或粗粒度迁移能力。

### 10.3 mixed-level graded 指标

相关性：

```text
same scene: 1.0
same task only: 0.3
otherwise: 0.0
```

建议指标：

- `MixedNDCG@10`
- `SceneHit@1`
- `SceneHit@5`
- `TaskHit@5`
- `TaskOnlyHit@5`

`TaskOnlyHit@5` 表示 top-5 中没有同 scene，但有同 task，可用于说明“粗粒度正确但细粒度失败”。

## 11. 权重与融合

第一版推荐：

```text
calibration = zscore
fusion = weighted_sum
missing_policy = renormalize
```

默认权重：

```json
{
  "video": 0.5,
  "trajectory": 0.2,
  "semantic_text": 0.3
}
```

后续如果有 val：

```text
step = 0.1
objective = 0.5 * MRR_scene + 0.3 * MixedNDCG@10 + 0.2 * SceneHit@1
```

权重搜索不能用 test。

## 12. 设计风险与更正

### R1：语义特征不是全 camera

可能性：高。

处理：

- 语义按 scene 共享。
- 第一版主评价用 scene-level。
- camera-level 不作为主结果。

### R2：使用 `<best_run>/final_test` 会把评估产物误当检索库

可能性：高。

处理：

- 新增独立建库脚本。
- `final_test` 只能作为历史单模态结果参考，不能作为系统输入。

### R3：mixed level 容易错误实现成标签泄漏

可能性：中高。

处理：

- mixed 只用于评价 relevance，不参与排序。
- 检索排序只用模态 feature similarity。

### R4：三模态完整 pair 数量可能偏少

可能性：中。

处理：

- S1 先用完整 pair 得到干净主结果。
- S4 再评估缺失模态。
- coverage 必须作为实验产物报告。

### R5：query 语义特征的来源不一致

可能性：中。

处理：

- 系统第一版要求 query semantic 已经是 embedding。
- 不在 query 阶段调用 VLM。
- 如果未来要支持 raw video 到 semantic，需要单独设计在线 VLM 标注链路。

## 13. 最小可行实现顺序

### Phase 1：建库闭环

1. 新增 `build_retrieval_library.py`。
2. 补视频和轨迹单侧 encode API。
3. 导入已有 semantic feature。
4. 输出 `features/*.npy`、`feature_records.jsonl`、`coverage.json`。

验收：

- 能构建 `artifacts/retrieval/rh20t_cfg2_v1`。
- robot gallery 特征完整。
- human eval 特征可选保存。

### Phase 2：单条 query 检索

1. 新增 `query_retrieval_system.py`。
2. 支持 video / trajectory / semantic 任意组合输入。
3. 支持缺失模态权重重归一。
4. 输出 Top-K 和每个模态贡献分数。

验收：

- 只给 video 能检索。
- 只给 semantic 能检索。
- 给三模态能融合检索。

### Phase 3：系统评估

1. 新增 `evaluate_retrieval_system.py`。
2. 支持 `--require-modalities`。
3. 支持 `level=scene/task/mixed`。
4. 输出 `metrics.json`、`cases.json`、`per_query_results.jsonl`。

验收：

- 能在三模态完整 pair 上跑 S1。
- 能跑 S2/S3 消融。

### Phase 4：图表与 cfg3

1. 新增 final chart 脚本。
2. 构建 cfg3 检索库。
3. 用 cfg2 权重评估 cfg3。

验收：

- 生成主结果表和图。
- cfg3 不重新调权。

## 14. 最终结果呈现

建议最终报告只保留四类图：

1. `system_metrics_comparison.png`：Video / Trajectory / Semantic / V+S / V+T+S。
2. `missing_modality_ablation.png`：缺失模态鲁棒性。
3. `level_metrics_comparison.png`：scene/task/mixed 对照。
4. `modality_coverage.png`：检索库覆盖情况。

主表：

| Method | Scene R@1 | Scene R@5 | MRR | NDCG@10 | MixedNDCG@10 |
|---|---:|---:|---:|---:|---:|
| Video only |  |  |  |  |  |
| Trajectory only |  |  |  |  |  |
| Semantic only |  |  |  |  |  |
| Video + Semantic |  |  |  |  |  |
| Video + Trajectory + Semantic |  |  |  |  |  |

定性案例从 `cases.json` 选择：

- 三模态融合修正单模态错误。
- 只命中 task 但没命中 scene。
- 缺失某模态仍然检索成功。
- 融合失败案例，分析哪个模态误导。

## 15. 第一版推荐范围

为了尽快落地，不建议第一版做复杂联合训练。最小闭环应为：

- cfg2 建库。
- robot gallery + human eval query。
- scene-level 特征聚合。
- query 支持 video / trajectory / semantic 任意缺失。
- 先在三模态完整 pair 上做主实验。
- 用 `zscore + weighted_sum + missing renormalize`。
- `mixed` 只作为评价 relevance，不参与排序。

这个范围能最直接验证“检索系统是否成立”，也为后续真实在线 query、cfg3 迁移和缺失模态鲁棒性实验留下兼容空间。

## 16. 当前代码入口

### 16.1 构建 cfg2 检索库

```bash
scripts/build_retrieval_library.sh
```

等价显式命令：

```bash
python runs/retrieval/build_retrieval_library.py \
  --config configs/retrieval/library_rh20t_cfg2.json
```

常用覆盖变量：

```bash
CONFIG=configs/retrieval/library_rh20t_cfg2.json scripts/build_retrieval_library.sh
```

### 16.2 评估检索系统

```bash
scripts/evaluate_retrieval_system.sh
```

等价显式命令：

```bash
python runs/retrieval/evaluate_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/system_eval_full_modalities.json \
  --level mixed \
  --require-modalities video,trajectory,semantic_text \
  --output-dir artifacts/retrieval/rh20t_cfg2_v1/eval/full_modalities_mixed \
  --top-k 10
```

单模态输入能力评估示例：

```bash
python runs/retrieval/evaluate_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/system_default.json \
  --enabled-modalities video \
  --require-modalities video \
  --level scene \
  --output-dir artifacts/retrieval/rh20t_cfg2_v1/eval/video_only_scene
```

### 16.3 单条 query 检索

使用检索库内 human eval query：

```bash
python runs/retrieval/query_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/system_default.json \
  --query-id "<query_id>" \
  --top-k 10
```

使用外部已编码语义特征：

```bash
python runs/retrieval/query_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/system_default.json \
  --semantic-feature query_semantic_feature.json \
  --top-k 10
```

使用外部 human video：

```bash
python runs/retrieval/query_retrieval_system.py \
  --library-dir artifacts/retrieval/rh20t_cfg2_v1 \
  --config configs/retrieval/system_default.json \
  --video-path /path/to/human.mp4 \
  --video-config configs/video/vjepa_rh20t_baseline.json \
  --video-checkpoint artifacts/runs/video/video_vjepa_rh20t_baseline_20260429_145608/best_model.pth \
  --top-k 10
```

### 16.4 导出图表

```bash
python runs/retrieval/export_retrieval_system_charts.py \
  --runs-json artifacts/retrieval/system_final_runs.json \
  --output-dir artifacts/retrieval/final_charts \
  --level scene
```

`system_final_runs.json` 示例：

```json
{
  "Video": "artifacts/retrieval/rh20t_cfg2_v1/eval/video_only_scene",
  "Semantic": "artifacts/retrieval/rh20t_cfg2_v1/eval/semantic_only_scene",
  "V+T+S": "artifacts/retrieval/rh20t_cfg2_v1/eval/full_modalities_mixed"
}
```

## 17. 已知限制

- 第一版直接矩阵检索，FAISS index 目录已预留但还未接入在线查询。
- `query_retrieval_system.py` 当前支持 raw video 编码、已编码 trajectory feature、已编码 semantic feature；raw trajectory 在线编码后续可补。
- `level=mixed` 已实现为评价指标，不参与排序。
- 检索库可保存 human/robot 两侧特征，但生产检索的 gallery 只读取 `gallery_robot.jsonl`。
