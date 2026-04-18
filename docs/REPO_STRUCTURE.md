# BISE 仓库重构说明

## 目标

本次重构的目标不是简单换目录名，而是把仓库从“研究脚本堆叠”收敛为“领域清晰、入口稳定、产物有归档规范”的工程结构。

核心原则：

- `src/bise/` 只放可复用库代码，不再把训练主流程直接塞进包根的零散脚本。
- `tools/` 只放命令入口，职责是加载配置、调用库模块、保存产物。
- `configs/` 保存可复现配置样例，不再在代码里硬编码实验参数。
- `artifacts/` 统一承接运行产物，避免源码和模型/曲线/索引混放。
- 目录优先按领域组织，再按技术层细分。

## 关键文档

- `docs/REPO_STRUCTURE.md`：仓库结构、目录职责和迁移原则。
- `docs/SEMANTIC_VLM_MODULE_PLAN.md`：语义层 VLM 模块完整开发方案，覆盖技术栈、数据组织、测试、指标和实验呈现。
- `docs/SEMANTIC_VLM_IMPLEMENTATION_GUIDE.md`：语义层 VLM 模块开发实现文档，面向具体代码实现、文件职责、数据流和分阶段落地。

## 新目录结构

```text
BISE/
├── artifacts/
│   ├── figures/
│   ├── models/
│   └── runs/
├── configs/
│   ├── retrieval/
│   ├── semantic/
│   ├── trajectory/
│   └── video/
├── docs/
├── experiments/
│   └── video/
├── src/
│   └── bise/
│       ├── common/
│       ├── data/
│       │   ├── rh20t/
│       │   └── whirl/
│       ├── modalities/
│       │   ├── semantic/
│       │   ├── trajectory/
│       │   └── video/
│       ├── retrieval/
│       └── service/
├── tests/
└── tools/
```

## 各目录职责

### `src/bise/common`

放跨模块复用能力：

- `config.py`：读取 JSON 配置与覆盖参数。
- `paths.py`：统一路径解析与目录创建。
- `run_artifacts.py`：统一保存实验参数、指标和曲线。
- `schemas.py`：定义统一的 embedding sample 结构。

### `src/bise/data`

按数据域拆分，而不是把所有数据逻辑塞进一个 `data/` 平铺层。

- `rh20t/`
  - `scanner.py`：目录扫描与 `SceneRecord` 定义。
  - `video_dataset.py`：RH20T 视频数据集。
  - `trajectory_dataset.py`：RH20T 轨迹数据集。
  - `collate.py`：轨迹批处理逻辑。
  - `subset_extractor.py` / `hand_pose_extractor.py` / `*_validator.py`：预处理与校验工具。
- `whirl/`
  - `video_pair_dataset.py`：WHIRL 配对视频数据集。
  - `csv_manifest.py`：CSV manifest 生成器。

### `src/bise/modalities`

按模态组织核心研究代码。

- `trajectory/`
  - `models/trajectory_encoder.py`
  - `losses.py`
  - `augmentations.py`
  - `trainer.py`
  - `evaluator.py`
- `video/`
  - `models/videomae_adapter.py`
  - `models/vjepa_adapter.py`
  - `losses.py`
  - `trainer.py`
  - `evaluator.py`
- `semantic/`
  - 预留 VLM 接入、prompt、缓存、schema。

### `src/bise/retrieval`

放检索系统共性能力，而不是散落在评估脚本里。

- `metrics.py`：Recall/MRR/NDCG
- `fusion.py`：加权融合
- `feature_store.py`：特征文件读写
- `faiss_index.py`：FAISS 索引封装
- `extractor.py`：统一 embedding sample 构造

### `src/bise/service`

在线服务层：

- `app.py`：Flask app 工厂
- `query_pipeline.py`：排序/融合流程
- `schemas.py`：请求响应结构

### `tools`

收敛所有入口：

- `train_trajectory.py`
- `train_video.py`
- `evaluate_retrieval.py`
- `build_index.py`
- `serve_api.py`

这样做的意义是：实验入口改动只影响 `tools/`，可复用逻辑仍沉淀在 `src/bise/`。

## 命名规范

- 文件名统一 `snake_case`
- 类名统一 `PascalCase`
- 模型/数据集文件名必须体现职责，例如 `trajectory_dataset.py`
- 入口脚本使用动词短语，例如 `train_trajectory.py`
- 禁止继续使用 `functions.py`、`test.py`、`videomae1.py` 这类弱语义命名

## 原结构存在的问题与对应处理

## 旧路径到新路径映射

| 旧路径 | 新路径 |
| :--- | :--- |
| `src/data/rh20t.py` | `src/bise/data/rh20t/scanner.py` + `trajectory_dataset.py` + `video_dataset.py` + `collate.py` |
| `src/data/whirlDataset.py` | `src/bise/data/whirl/video_pair_dataset.py` |
| `src/utils/csv_generator.py` | `src/bise/data/whirl/csv_manifest.py` |
| `src/models/trajectoryEncoder.py` | `src/bise/modalities/trajectory/models/trajectory_encoder.py` |
| `src/loss/functions.py` | `src/bise/modalities/trajectory/losses.py` |
| `src/utils/data_augment.py` | `src/bise/modalities/trajectory/augmentations.py` |
| `src/evaluation/trajectory_functions.py` | `src/bise/modalities/trajectory/evaluator.py` |
| `src/models/finetuner.py` | `src/bise/modalities/video/models/videomae_adapter.py` + `vjepa_adapter.py` |
| `src/loss/info_nce.py` | `src/bise/modalities/video/losses.py` |
| `src/evaluation/functions.py` | `src/bise/retrieval/metrics.py` |
| `src/pipelines/*.py` | `tools/*.py` + `src/bise/modalities/*/trainer.py` |
| `scripts/videomae*.py` | `experiments/video/*.py` |

### 1. 包根混乱

原仓库把 `src` 当成实际 import 包使用，并依赖 `sys.path.append('.')`。

处理：

- 统一迁移到 `src/bise/`
- `tools/` 负责本地开发态 bootstrap
- 包安装配置切到标准 `src` layout

### 2. 技术层平铺导致跨目录跳转严重

原来 `src/data`、`src/models`、`src/loss`、`src/evaluation`、`src/pipelines` 是横切平铺结构。

处理：

- 改成 `data/rh20t`、`modalities/trajectory`、`modalities/video`、`retrieval`、`service`
- 让“同一业务域代码”尽量相邻

### 3. 单文件职责过大

原 `src/data/rh20t.py` 同时承担扫描、视频数据集、轨迹数据集和 collate。

处理：

- 拆成 `scanner.py`、`video_dataset.py`、`trajectory_dataset.py`、`collate.py`

### 4. 训练脚本碎片化

原先 `trajectoryTrain.py`、`trajectory_augment.py`、`trajectory_augment_2stage.py`、`trajectoryTrain_cfg23.py` 都是独立脚本。

处理：

- 训练逻辑收敛到 `src/bise/modalities/trajectory/trainer.py`
- 入口统一到 `tools/train_trajectory.py`
- 通过 `configs/trajectory/*.json` 区分 baseline / augment / two_stage

### 5. 运行产物缺乏统一落点

原仓库里 `results/`、`model_weight/`、`figures/`、根目录零散文件并存。

处理：

- 新增 `artifacts/runs`、`artifacts/models`、`artifacts/figures`
- 新产物默认应优先落到 `artifacts/`

## 建议的后续迁移原则

本次已经完成结构收敛，后续新增功能建议遵守：

1. 新增模态代码先判断属于 `data`、`modalities`、`retrieval`、`service` 哪一层。
2. 新实验不要再复制一份训练脚本，先考虑是否只需要加一个 config。
3. 任何离线产物都不要直接写入源码目录。
4. 文档里的目录说明需要与实际仓库保持同步更新。
