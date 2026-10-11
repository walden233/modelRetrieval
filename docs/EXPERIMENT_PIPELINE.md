# 配置驱动的实验流水线

更新时间：2026-10-10。

## 当前两模态 retrain 入口

当前轨迹 T0–T5、N1–N3 与视频 E0–E7、RawJ/RawM 使用
唯一模板 `configs/retrain/s43/run.json`（seed 43），两模态配置分别位于其 `trajectory/`、`video/`。
完整矩阵、划分开关和产物规则见 [RETRAIN.md](RETRAIN.md)。

```bash
conda activate torch2
python runs/retrain.py --config configs/retrain/s43/run.json --batch s45
```

该入口依次准备清单、审计、准备全部 run 元信息、顺序训练、独立评估 cfg2/cfg3，再输出比较图。
cfg2 评估各 run 自己的测试划分；cfg3 每 task 固定两个 scene。
`--skip-existing` 只跳过已完成训练，评估仍重新计算；`--stage charts` 只重新汇总已有指标。
图表依赖记入批次根目录 `plots.json`，`runs/redraw.py --batch <batch>` 可重绘全部图片。
s44 视频已完成编号迁移，无需为编号调整重训，详见 [映射和复用入口](RETRAIN.md#s44-视频编号迁移)。

## 历史评估编排与论文数据包入口

下文的 `configs/experiments/`、旧 E1–E6 / T1–T5 和 cfg3 all-test 方案保留用于历史复现，
与当前 retrain 的配置、编号、划分和跳过规则分别解释。

仓库已有 `scripts/evaluate_trajectory.sh`、`scripts/evaluate_video.sh`、
`scripts/evaluate_retrieval_system.sh`、`scripts/run_retrieval_system_ablation.sh` 和
`scripts/search_fusion_weights.sh` 等入口，但过去需要手工串联，而且输出目录没有统一登记。
现在用 [configs/experiments/paper_baseline_cfg2_cfg3.json](../configs/experiments/paper_baseline_cfg2_cfg3.json)
描述一组实验，用 `scripts/run_experiment_pipeline.sh` 按步骤执行。

## 常用命令

先进入项目并使用 `torch2` 环境：

```bash
cd /home/ttt/BISE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch2
```

查看将要执行的命令，不实际运行：

```bash
scripts/run_experiment_pipeline.sh --dry-run
```

只跑全部单模态测试（轨迹 T1–T5、视频 E1–E6 及两个 RAW backbone，cfg2/cfg3
各一遍）：

```bash
scripts/run_experiment_pipeline.sh \
  --group trajectory_single,video_single \
  --skip-existing
```

一键执行配置中的所有步骤，并在已有结果时跳过：

```bash
scripts/run_experiment_pipeline.sh --skip-existing
```

只执行某些步骤，例如两个单模态评估：

```bash
scripts/run_experiment_pipeline.sh \
  --only trajectory_T2_cfg2_test,trajectory_T2_cfg3_all_test \
  --skip-existing
```

默认配置还包含视频 `E1`–`E6`、`RAW_VJEPA`、`RAW_VIDEOMAE` 和轨迹 `T1`–`T5`，
每个模型均有 cfg2 `test` 与 cfg3 `all_test` 两个单模态评估步骤。

也可以指定另一份配置：

```bash
CONFIG=configs/experiments/my_experiment.json \
scripts/run_experiment_pipeline.sh --skip-existing
```

示例配置中的三模态模型与现有 `configs/retrieval/library_rh20t_cfg2.json`、
`library_rh20t_cfg3.json` 保持一致：轨迹使用 T2 augment run，视频使用 V-JEPA
baseline。若论文改用 T1 或 VideoMAE，只需修改对应步骤的
`--config`、`--checkpoint` 和 `--split-manifest`。

每次运行都会生成：

```text
artifacts/experiments/<experiment_id>/
├── pipeline_manifest.json  # 每一步的命令、状态、日志、输出和配置哈希
└── logs/                   # 每一步的 stdout/stderr
```

配置中的 `outputs` 是幂等判断依据；只有声明的输出全部存在时，
`--skip-existing` 才会跳过该步骤。删除某一步输出或不使用 `--skip-existing` 即可重跑。

## 论文数据包

配置中的 `bundle` 会在全部步骤结束后，把评估目录、权重搜索结果、训练曲线、配置和
必要的图表收集到一个目录：

```text
artifacts/paper/<experiment_id>/
├── steps/                 # 各步骤导出的 metrics/cases/npy/png 等
├── source_runs/           # 选定训练 run 的 params、split、曲线和最佳指标
├── pipeline_manifest.json
└── bundle_manifest.json   # 每个文件的源路径、大小和 SHA256
```

如果只想重新导出而不重跑实验：

```bash
MANIFEST=artifacts/experiments/paper_baseline_cfg2_cfg3/pipeline_manifest.json \
scripts/export_paper_bundle.sh
```

导出工具不会复制 checkpoint 和原始数据，避免论文目录过大；如需纳入额外文件，
在配置 `bundle.files` 或 `bundle.patterns` 中显式添加。

## 目录约定

历史目录（例如 `artifacts/runs/trajectory/<run>/final_test` 和
`cfg3_all_test`）保持不动，保证中期结果路径和图表可复现。新的重复评估建议写到：

```text
artifacts/runs/<modality>/<run>/evaluations/<cfg>/<evaluation_name>/
```

例如 `evaluations/cfg2/test`、`evaluations/cfg3/all_test`。训练 run 是模型身份，
`evaluations` 是可重复的评估记录；同一 checkpoint 可以有多个评估配置而不会互相覆盖。
统一检索库和权重搜索继续放在 `artifacts/retrieval/`，论文成品数据再由 bundle 汇总。
