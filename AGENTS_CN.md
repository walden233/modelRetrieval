# 仓库指南

## 编码使用虚拟环境torch2（conda activate torch2）

## 项目定位
跨域多模态检索：输入人类侧的演示视频 / 人手轨迹 / 语义特征（三者可任意缺失，至少一个），
从机器人侧候选库中检索最匹配的机器人执行样本。评价粒度分 `scene` / `task` / `mixed` 三档。

## 项目结构与模块组织
- `src/bise/common`：配置、路径、日志、run 产物保存与统一 schema。
- `src/bise/data/rh20t`、`src/bise/data/whirl`：RH20T / WHIRL 数据集扫描、子集抽取、手部关键点提取、预处理与校验。
- `src/bise/modalities/trajectory`、`src/bise/modalities/video`：按模态组织模型、loss、trainer、evaluator、factory。
- `src/bise/modalities/semantic`：VLM 标注流水线（不训练模型）。含 prompt 模板、响应解析、taxonomy 归一化、文本编码、同步与 Batch 两条执行链路。
- `src/bise/retrieval`：检索指标、融合、feature store 与 FAISS 索引封装。
- `src/bise/retrieval/system`：统一检索系统核心——检索库 schema、逐模态打分、分数校准、加权融合、系统级评估。
- `src/bise/service`：Flask API。**当前是早期原型，未接入检索库与编码器，仅有分数融合骨架；可用入口是 `runs/retrieval/query_retrieval_system.py`。**
- `runs/`：命令入口，按模态分目录（`trajectory/`、`video/`、`semantic/`、`batch/`、`retrieval/`），另有 `serve_api.py`、`restore_charts_from_json.py`。
- `scripts/`：shell 封装，把常用路径和变量前置成环境变量，复现实验优先用它们。
- `configs/`：实验配置；`artifacts/`：训练产物、语义标注与检索库；`docs/`：设计与实验计划。

产物目录约定：
```text
artifacts/runs/video|trajectory/<run>/       单模态训练产物（metrics.json、curves.json、split_manifest.json、final_test/）
artifacts/semantic/rh20t/<cfg>/              语义标注、embedding、评估、图表
artifacts/retrieval/rh20t_<cfg>_v1/          检索库（features/、manifests/、coverage.json）与 eval/
```

## 构建、测试与开发命令
先激活环境：`conda activate torch2`。

- `python -m pip install -e .` —— 可编辑模式安装 `bise` 及其依赖，修改 `setup.py` 后需重装。
  注意：`zai`（智谱 Batch SDK，`src/bise/modalities/semantic/batch_client.py` 使用）尚未在 `setup.py` 中声明。
- 训练（优先用 `scripts/*.sh`，也可直接跑 `runs/` 下的脚本）：
  - `scripts/train_trajectory.sh` —— 轨迹训练，默认 `configs/trajectory/T1_trajectory_baseline_scene.json`；用 `CONFIG=... scripts/train_trajectory.sh` 换配置，`DATA_ROOT=` / `OUTPUT_DIR=` 覆盖路径。
  - `scripts/train_video.sh` —— 视频训练，默认 V-JEPA baseline。
  - `scripts/run_all_video_experiments.sh`、`scripts/run_all_trajectory_experiments.sh` —— 一键跑实验组。
- 单模态评估：`scripts/evaluate_trajectory.sh`、`scripts/evaluate_video.sh`（默认评估 `RUN/final_test`，用 `RUN=<run 目录>` 指定）。
- 语义：`python runs/semantic/build_semantic_manifest.py --config configs/semantic/pipeline_v1.json ...` 建 manifest；
  `python runs/semantic/run_semantic_annotation.py --config configs/semantic/pipeline_v1.json` 跑标注（需 `export ZHIPU_API_KEY=...`）。
- 检索系统：
  - `scripts/build_retrieval_library.sh` —— 从数据 + 模型 checkpoint 建检索库。
  - `scripts/query_retrieval_system.sh --query-id "<query_id>"` —— 单条检索；也支持 `--video-path` 现场编码或传 embedding 文件。
  - `scripts/evaluate_retrieval_system.sh` —— 系统评估；`scripts/run_retrieval_system_ablation.sh` —— 七种模态组合消融。
- 图表：`scripts/export_{video,trajectory,semantic,retrieval_system}_final_charts.sh`；
  从已有 JSON 恢复 PNG 用 `python runs/restore_charts_from_json.py --root artifacts --overwrite`。
- `python runs/serve_api.py --port 5000` —— 启动检索服务（当前为原型接口）。

## 代码风格与命名
- 遵循 PEP 8：四空格缩进、100 字符行宽、函数/模块 `snake_case`，类使用 `PascalCase`。
- 显式处理张量设备与 dtype（每批次统一 `tensor.to(device)`）；公共函数尽量添加类型注解。
- 新实验优先新增 `configs/*.json`，不要复制一份新的训练脚本。
- 行内注释保持英文并聚焦动作；遇到复杂张量变换前先写一句总结。

## 测试规范
- 采用 `pytest` 在 `tests/` 下编写场景化用例，文件名与模块对应（如 `test_retrieval_metrics.py`、`test_semantic_parser.py`）。
- 排名逻辑优先覆盖 `src/bise/retrieval/metrics.py`、`src/bise/retrieval/system/` 与 `src/bise/data/rh20t/collate.py`。
- 提交前执行 `python -m pytest -q tests`；若环境缺少 `pytest`，先安装依赖再跑。

## 提交与 PR 准则
- 提交信息保持简短命令式（如 `use_6_keypoints`、`trajectory_augment`），本地合并碎片化 WIP。
- PR 需描述动机、数据集分片与指标（贴出 `R@1/5/10`、`MRR`、`NDCG@10` 及 `artifacts/` 路径）。
- 关联 Issue 或实验记录，变更学习率/调度时附曲线截图，并说明数据集 schema、权重格式等兼容性风险。

## 安全与配置提示
- 禁止提交 RH20T 原始子集或凭证；遵守 `.gitignore`。
- 将 `data_root`、`output_dir`、`checkpoint` 等参数放在 `configs/` 中管理，避免硬编码 `/home/ttt`。
  （`scripts/*.sh` 中的 `PYTHON_BIN` 默认指向该绝对路径，已可通过环境变量覆盖。）
- VLM API key 走环境变量 `ZHIPU_API_KEY`，不要写进配置文件。
- 在接入真实 VLM 或在线检索服务前，先用本地 stub 与缓存层验证，确认不会泄露真实密钥。
