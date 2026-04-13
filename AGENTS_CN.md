# 仓库指南

## 编码使用虚拟环境torch2（conda activate torch2）

## 项目结构与模块组织
- `src/bise/common`：配置、路径、日志、实验产物保存与统一 schema。
- `src/bise/data/rh20t`、`src/bise/data/whirl`：RH20T / WHIRL 数据集、预处理与校验工具。
- `src/bise/modalities/trajectory`、`src/bise/modalities/video`、`src/bise/modalities/semantic`：按模态组织模型、loss、trainer、evaluator。
- `src/bise/retrieval`：检索指标、融合、feature store 与 FAISS 索引封装。
- `src/bise/service`：Flask API 与在线查询排序流程。
- `tools/`：稳定命令入口；`configs/`：实验配置；`artifacts/`：模型、曲线、运行产物；`experiments/`：归档的探索性脚本。

## 构建、测试与开发命令
- `python -m pip install -e .` —— 可编辑模式安装 `bise` 及其 PyTorch 依赖，修改 `setup.py` 后需重装。
- `python tools/train_trajectory.py --config configs/trajectory/baseline.json` —— 启动轨迹 baseline 训练。
- `python tools/train_trajectory.py --config configs/trajectory/augment.json` —— 启动轨迹增强训练。
- `python tools/train_video.py --config configs/video/vjepa_whirl.json` —— 启动视频训练入口。
- `python tools/evaluate_retrieval.py --config configs/trajectory/baseline.json --checkpoint <路径>` —— 评估轨迹检索模型。
- `python tools/build_index.py --features <feature_store.json> --output artifacts/models/index.faiss` —— 构建 FAISS 索引。
- `python tools/serve_api.py --port 5000` —— 启动检索服务。

## 代码风格与命名
- 遵循 PEP 8：四空格缩进、100 字符行宽、函数/模块 `snake_case`，类使用 `PascalCase`。
- 显式处理张量设备与 dtype（每批次统一 `tensor.to(device)`）；公共函数尽量添加类型注解。
- 新实验优先新增 `configs/*.json`，不要复制一份新的训练脚本。
- 行内注释保持英文并聚焦动作；遇到复杂张量变换前先写一句总结。

## 测试规范
- 采用 `pytest` 在 `tests/` 下编写场景化用例，文件名与模块对应（如 `test_retrieval_metrics.py`）。
- 排名逻辑优先覆盖 `src/bise/retrieval/metrics.py` 与 `src/bise/data/rh20t/collate.py`。
- 提交前执行 `python -m pytest -q tests`；若环境缺少 `pytest`，先安装依赖再跑。

## 提交与 PR 准则
- 提交信息保持简短命令式（如 `use_6_keypoints`、`trajectory_augment`），本地合并碎片化 WIP。
- PR 需描述动机、数据集分片与指标（贴出 `R@1/5/10`、`mean_percentage_rank` 及 `artifacts/` 或 `results/` 路径）。
- 关联 Issue 或实验记录，变更学习率/调度时附曲线截图，并说明数据集 schema、权重格式等兼容性风险。

## 安全与配置提示
- 禁止提交 RH20T 原始子集或凭证；路径放入 `.env` 或私有 YAML，遵守 `.gitignore`。
- 将 `data_root`、`output_dir`、`checkpoint` 等参数放在 `configs/` 中管理，避免硬编码 `/home/ttt`。
- 在接入真实 VLM 或在线检索服务前，先用本地 stub 与缓存层验证，确认不会泄露真实密钥。
