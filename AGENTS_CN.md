# 仓库指南

## 项目结构与模块组织
- `src/data`、`src/utils`：共享的数据集适配器（如 `RH20TTraceDataset`）与张量工具。
- `src/models`、`src/loss`：跨模态 Transformer 编码器与对比损失；对应权重放在 `model_weight/`。
- `src/pipelines`：训练入口（`trajectoryTrain.py`、`trajectory_hypersearch.py`、增强脚本）；脚本默认读取 `dataset/` 下的数据根目录。
- `src/evaluation`：检索评估（`evaluate_gemini`）与排名统计，输出结果保存在 `results/`。
- 顶层 `scripts/` 用于快速实验（`videomae*.py`、`test.py`）；稳定逻辑请同步回 `src/`。

## 构建、测试与开发命令
- `python -m pip install -e .` —— 可编辑模式安装 `bise` 及其 PyTorch 依赖，修改 `setup.py` 后需重装。
- `python src/pipelines/trajectoryTrain.py --data-root <路径>` —— 启动默认训练循环；未来若加入 CLI 参数，可用于覆盖 `BATCH_SIZE`、`MODEL_PARAMS` 等。
- `python src/pipelines/trajectory_hypersearch.py --config configs/cfg23.yaml` —— 进行超参数搜索后再更新默认值。
- `python scripts/test.py` —— CUDA/环境快速自检，长训练前先运行以确认设备可用。

## 代码风格与命名
- 遵循 PEP 8：四空格缩进、100 字符行宽、函数/模块 `snake_case`，类使用 `PascalCase`。
- 显式处理张量设备与 dtype（每批次统一 `tensor.to(device)`）；公共函数尽量添加类型注解。
- 配置或实验文件沿用 `trajectoryTrain_cfg<number>.py` 命名。
- 行内注释保持英文并聚焦动作；遇到复杂张量变换前先写一句总结。

## 测试规范
- 采用 `pytest` 在 `tests/`（如不存在先创建）下编写场景化用例，文件名与模块对应（`test_trajectory_encoder.py`），覆盖率≥80%。
- 排名逻辑复用小型 `torch` 张量夹具，断言 `evaluate_gemini` 指标。
- 提交 PR 前执行 `pytest src tests` 与 `python scripts/test.py`，防止 CUDA 回归。

## 提交与 PR 准则
- 提交信息保持简短命令式（如 `use_6_keypoints`、`trajectory_augment`），本地合并碎片化 WIP。
- PR 需描述动机、数据集分片与指标（贴出 `R@1/5/10`、`mean_percentage_rank` 及 `results/` 路径）。
- 关联 Issue 或实验记录，变更学习率/调度时附 TensorBoard 截图，并说明数据集 schema、权重格式等兼容性风险。

## 安全与配置提示
- 禁止提交 RH20T 原始子集或凭证；路径放入 `.env` 或私有 YAML，遵守 `.gitignore`。
- 将 `DATASET_ROOT` 和模型保存路径参数化，避免硬编码 `/home/ttt`；脚本尽量使用相对路径。
- 在 `evaluate_gemini` 中加入云服务评估前，先用模拟响应验证，确认不会泄露真实密钥。
