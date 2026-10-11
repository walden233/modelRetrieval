# RH20T retrain

更新时间：2026-10-10。本文为当前实验编号、流水线和产物规则的主说明。

配置位于 `configs/retrain/`，产物位于
`artifacts/runs/retrain/<batch>/{trajectory,video}/<group>/`。
当前轨迹设计和几何表示见 [轨迹模板](trajectory/S43.md)。默认模板 seed 42，s43 模板 seed 43；
两者实验矩阵相同。更新源配置不会自动改变已有 run 的实际训练设置；
以各 run 的 `params.json` 为准。s44 视频的编号迁移见下节。

## 当前实验矩阵

轨迹统一 65 epoch、batch 8、双残差适配头与共享检索投影。
单/双指 Transformer 数量；T3 为双 Transformer＋增强基线。
除 T5 使用 21 点（63 维）外，人手使用 11 点；N2/N3 再增加运动分支。

| 实验 | 相对设置 |
|---|---|
| T0 | 无模态内增强，单 Transformer |
| T1 | 无模态内增强，双 Transformer |
| T2 | 有模态内增强，单 Transformer |
| T3 | 有模态内增强，双 Transformer；基线 |
| T4 | T3＋task-held-out 划分 |
| T5 | T3＋21 点人手 |
| N1 | T3＋初始掌部平移、旋转归一化 |
| N2 | T3＋运动/手形分离，人类运动＋局部手形，机器人运动＋虚拟两指 |
| N3 | N1＋N2 |

轨迹初始学习率 2e-4，前 10 epoch 保持，随后余弦下降到 2e-5。
`best_model.pth` 仍按验证集 human→robot task MRR 选择。
每个 run 的 epoch 平均总训练 loss 首次严格低于 `loss_checkpoint_threshold=0.73` 时，
额外保存一次 `loss_model.pth`，不覆盖它；对应 epoch/loss 记录在日志及 `status.json:loss_checkpoint`。
未达到阈值则不生成该文件。轨迹、视频均不再保存 `last_model.pth`，已有历史文件不自动删除。

视频统一 batch 4、5 epoch；以下未注明项以 E3 为准。

| 实验 | 跨域损失 | 模态内增强 | 模型/划分 |
|---|---|---|---|
| E0 | 一对一 InfoNCE | 无 | 冻结 V-JEPA2＋双适配头 |
| E1 | task 多正样本 | 无 | 同 E0 |
| E2 | 一对一 InfoNCE | 有 | 同 E0 |
| E3 | task 多正样本 | 有 | 冻结 V-JEPA2＋双适配头；基线 |
| E4 | 同 E3 | 有 | 冻结 V-JEPA2＋机器人单适配头 |
| E5 | 同 E3 | 有 | 微调 V-JEPA2 最后两个 block＋双适配头 |
| E6 | 同 E3 | 有 | task-held-out 划分 |
| E7 | 同 E3 | 有 | 冻结 VideoMAE＋双适配头 |
| RawJ / RawM | 不训练 | 无 | 原始 V-JEPA2 / VideoMAE 特征评估 |

E4 仍训练共享最终投影。所有增强组沿用现有增强和损失；
[跨相机扩展与语义监督](video/CROSS_VIEW.md)是后续独立方案，本次不纳入。
旧 E6s/E6d 已由 E4/E5 替代，不再注册；未注册的 N0 配置仅供历史检索库兼容。

## s44 视频编号迁移

2026-10-10 已核对实际训练参数，将 s44 的八个完整视频 run 对应到当前编号：

| 当前编号 | 迁移前编号 | 设置 |
|---|---|---|
| E0 | E0 | 一对一，无模态内增强 |
| E1 | E4 | 多正样本，无模态内增强 |
| E2 | E5 | 一对一，有模态内增强 |
| E3 | E1 | 多正样本，有模态内增强；基线 |
| E4 | E6s | 机器人单适配头 |
| E5 | E6d | 微调最后两个 block＋双适配头 |
| E6 | E3 | task-held-out |
| E7 | E2 | VideoMAE＋双适配头 |

目录、配置快照、`params.json`、评估来源记录、注册表、日志文件名和绘图依赖已同步。
八组均为 5 epoch、batch 4；RawJ/RawM 保留。权重、划分、训练曲线数据、评估指标及
特征数组经文件指纹校验保持不变，cfg2/cfg3 对比图和汇总曲线已按新编号重绘。
本次只调整视频实验身份，s44 继续使用原冻结清单，不应用新 batch 的 task 1–96 限制。
s44 轨迹仍是此前八组 70 epoch 的训练设置；当前九组轨迹矩阵应在新 batch 上运行。

迁移映射和元数据备份路径记入
[s44/pipeline.json](../artifacts/runs/retrain/s44/pipeline.json) 的 `rename_video` 步骤。
备份位于现有 `artifacts/runs/retrain/archive/s44/` 下；历史日志正文保留原执行编号。

```bash
# 复用已完成的视频训练，不重新评估。
python runs/retrain.py --config configs/retrain/s43/run.json --batch s44 --modality video --stage train --skip-existing

# 只重评新编号 E3，再合并其它已有指标；默认同时评估 cfg2/cfg3。
python runs/compare.py --runs artifacts/runs/retrain/s44/video/runs.json --only E3

# 从已有评估指标重新汇总视频主图与曲线，不加载模型。
python runs/compare.py --runs artifacts/runs/retrain/s44/video/runs.json --charts-only
```

## batch 划分准备

在批次入口 `configs/retrain/run.json` 或 `configs/retrain/s43/run.json` 中配置：

```json
"prepare": {
  "cfg2_test_first96": true
}
```

此开关属于 batch 清单准备，不属于单个实验配置。默认开启：新 batch 的 cfg2
固定测试集从 task_0001–task_0096 的共同有效 scene 中随机选择 120 个；
验证集仍从剩余全部 task 中选择 50 个，训练使用其余 scene。
关闭后，测试抽样不限制 task。不会把 task 97 之后的 scene 从整个数据集移除。

策略记入该 batch 的 `cfg2/metadata.json:test_task_max`（96 或 null），
轨迹、视频共用同一 scene 划分和顺序；视频每个 scene 展开两对固定相机。
T4/E6 仍按完整 task 隔离、接近 test 120 / val 50，自动读取冻结策略限制可选测试 task，
不改为共享 scene 测试集。三个集合保留 task 隔离，训练和验证 task 不受此范围限制。

新清单位于 `artifacts/manifests/retrain/<batch>/<cfg>/`，每个 cfg 保持四个文件：

```text
metadata.json
data_manifest.json
split_manifest_trajectory.json
split_manifest_video.json
```

两份 split 仅含 train/val/test 列表，与 run 内格式一致，scene 集合和顺序一致。
cfg3 不受 task 范围开关影响，仍为每个 task 两个评估 scene。
准备时只检查视频文件存在，不解码、不计算视频文件哈希。
本次不新增数据清单哈希字段。

已有 batch 优先复用冻结文件，即使准备开关改变，也不会改写已有清单，包括 s44。
要应用新开关和新矩阵，使用新的 batch，例如：

```bash
conda activate torch2
python runs/retrain.py --config configs/retrain/s43/run.json --batch s45
```

只准备共享清单：

```bash
python runs/prepare_splits.py --config configs/retrain/s43/run.json --batch s45
```

cfg2 评估继续使用每个 run 自己的 `split_manifest.json:test`；cfg3 使用 batch 的对应模态清单。
两个模态均评估 cfg2/cfg3，任务级划分 T4/E6 仍纳入主图并标记，不能将其分数差异直接归因于结构。

## 完整入口

在仓库根目录、`torch2` 环境执行。正式新配置要求 CUDA；不会静默改用 CPU。
输入来自 `dataset/RH20T_subset/RH20T_cfg2`、`RH20T_cfg3` 的人手、TCP、视频及
`gripper.npy` / `geometry.npy` / `geometry.json`，结构见
[数据存储说明](RH20T_SPLIT_AND_MANIFESTS.md#当前子集存储与夹爪来源)。
准备阶段直接从 dataset 构建有效库存，核验既有 NPY 指纹，并冻结当前视频配对；
无需预建 batch 清单、旧 `source/` 或额外的几何索引。

```bash
conda activate torch2

# 只显示计划，不创建文件或加载模型。
python runs/retrain.py --batch s42b --dry-run

# 生成/校验清单 → 审计源文件和划分 → 全部训练 → 独立评估 → 比较图。
python runs/retrain.py --batch s42b

# 已完成的训练可以复用；评估始终重新计算并覆盖输出。
python runs/retrain.py --batch s42b --skip-existing

# 只跑某一模态。
python runs/retrain.py --batch s42b --modality trajectory

# 固定划分种子来自 run.json；--seed 只覆盖训练和动态验证划分种子。
python runs/retrain.py --seed 43 --batch s43
```

也可用 `bash scripts/retrain.sh ...`，通过 `PYTHON_BIN` 指定 Python。
配置修改后可直接在原 batch 重跑，不检查批次配置或注册表是否变化。
批次 `configs/` 每次更新到源配置；已有 run 的 `params.json` 保留真实训练设置，
只有实际重训时才更新，独立评估继续读取该文件。

`--skip-existing` 是流水线复用，不是恢复中断训练的 optimizer 状态。
该参数仅跳过已完成的训练，评估阶段始终重新计算所选 run 的 cfg2/cfg3 结果。
`--only` 只重评所选组，随后合并其它已有指标；仅未选择且缺少评估的组会跳过并提示。
所选 run 没有评估目录时会生成新结果；子进程完成后必须存在完整指标、样本信息和 NPY 产物，
否则报错并指出对应日志，不会将所选 run 静默跳过。
`run_info.json` 是新评估生成的来源记录，不再作为重评、汇总或绘图的前置条件。
训练阶段的判断规则：

- 不加 `--skip-existing`：从头训练所有所选组，已有 run 先归档。
- 加 `--skip-existing`：同时存在 `best_model.pth`、`params.json`，且
  `status.json:state=complete` 时跳过训练；其它情况从头重训，不比较配置差异。
- 重训时沿用原 run 的 `split_manifest.json`，保持 train/val/test 及顺序。
  只有没有该清单时，才按当前配置生成划分；`last_model.pth` 不参与判断。
- 旧 run、过期比较图及对应日志归档，原评估缓存随旧 run 一起移走。

`train` 先为所有所选 run 准备配置、划分、库存和 `status.json:state=prepared`，
全部准备成功后才依次启动训练。RawJ/RawM 只准备元信息，不参与训练。

例如，在 `s42` 使用新配置重训全部轨迹并评估绘图：

```bash
python runs/retrain.py --batch s42 --modality trajectory

# 仅补训缺少最佳模型或未完成的 run，并复用其它已完成训练。
python runs/retrain.py --batch s42 --skip-existing
```

在 `--skip-existing` 模式下，修改配置本身不会触发重训。可删除目标 run 的
`best_model.pth` 后再跑，或用不带 `--skip-existing` 的 `--only` 命令指定重训组。

## 替换 task 划分实验

下面命令仅替换 `s42b/trajectory/T4` 和 `s42b/video/E6`，正式预算是 65 / 5 epoch。
当前有效 scene 总数为轨迹 428、视频 434；保留 test 120 / val 50 后，train 分别为
258 / 264。重训沿用已有划分；没有清单时按 task 隔离生成目标划分。
评估读取每个新 run 的清单。需要重新划分时，先删除所选 run 的
`split_manifest.json`，然后执行替换命令。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python runs/retrain.py --batch s42b --only T4 E6 --replace

# 先查看计划；不移动文件或加载模型。
python runs/retrain.py --batch s42b --only T4 E6 --replace --dry-run

# 分开执行：只替换并训练这两组，再独立评估和汇总绘图。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python runs/retrain.py --batch s42b --stage train --only T4 E6 --replace
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python runs/retrain.py --batch s42b --stage eval --only T4 E6
```

默认完整入口依次校验协议、审计、替换训练、评估 cfg2/cfg3，并与其它 run 已保存的评估指标
合并更新主图。旧 run、对应日志及过期的汇总图/数据移到
`artifacts/runs/retrain/archive/<batch>/<时间>/<modality>/`，其它组的 checkpoint、评估和
数据划分保留。绘图清单同步移除过期条目，完成训练/评估后记录新依赖。
早期 T4 / 视频旧 E3（当前任务划分编号 E6）曾使用 80%/10%/10% 划分，历史结果保留在归档中。
`--replace` 是兼容保留的显式替换入口，必须搭配 `--only` 和 `--stage all/train`，
每次执行都会从头训练所选组，也沿用已有划分；
`--skip-existing` 与替换互斥。

## 各阶段入口

```bash
# 只准备共享清单和数据库存快照。
python runs/prepare_splits.py --batch s42b

# 只审计；--files 校验轨迹文件哈希及视频文件存在。
python runs/audit_splits.py --files --output artifacts/runs/retrain/audit.json

# 只训练；也可选择部分组。--traj-epochs 可覆盖轨迹预算。
python runs/retrain.py --batch s42b --stage train
python runs/retrain.py --batch s42b --stage train --modality trajectory --only T2 N1

# 只评估和绘图；两个模态都默认 cfg2+cfg3。
python runs/retrain.py --batch s42b --stage eval

# 使用已训练的 checkpoint，完整重新评估所选组并覆盖已有输出。
HF_HUB_OFFLINE=1 python runs/retrain.py --batch s42b --stage eval --skip-existing

# 读取已有指标，重新汇总比较数据并导出主图，不重新评估。
python runs/retrain.py --batch s42b --stage charts

# 先记录依赖，再从已有数据重绘此批次全部图片，无需模型或原始数据集。
python runs/redraw.py --batch s42b

# 只补齐/更新图表依赖清单，不重绘。
python runs/redraw.py --batch s42b --index-only

# N1–N3 的独立人手变换稳健性评估，使用相同冻结查询/候选。
python runs/retrain.py --batch s42b --stage robust --modality trajectory
```

完整流水线默认评估全部 trajectory 与 video 组，包含原始 backbone；
人手稳健性作为独立阶段执行。WHIRL/debug 配置不属于这批 RH20T 重跑。

## 单 run 与增量评估

```bash
# 单 run cfg2：使用本 run 保存的模型配置与实际训练划分。
python runs/trajectory/evaluate_retrieval.py \
  --config artifacts/runs/retrain/s42b/trajectory/T2/params.json \
  --checkpoint artifacts/runs/retrain/s42b/trajectory/T2/best_model.pth \
  --split-manifest artifacts/runs/retrain/s42b/trajectory/T2/split_manifest.json \
  --output-dir artifacts/runs/retrain/s42b/trajectory/T2/evaluations/cfg2

# 单 run cfg3：必须同时换 data-root 和 manifest，不用 --all-as-test。
python runs/trajectory/evaluate_retrieval.py \
  --config artifacts/runs/retrain/s42b/trajectory/T2/params.json \
  --checkpoint artifacts/runs/retrain/s42b/trajectory/T2/best_model.pth \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --split-manifest artifacts/manifests/retrain/s42b/cfg3/split_manifest_trajectory.json \
  --output-dir artifacts/runs/retrain/s42b/trajectory/T2/evaluations/cfg3

# 只重算 T2/N1；其它组读取已有有效结果，重新输出 cfg2/cfg3 两张比较图。
python runs/compare.py \
  --runs artifacts/runs/retrain/s42b/trajectory/runs.json \
  --only T2 N1 --cfg cfg2 cfg3

# 视频相同，默认同时更新 cfg2/cfg3。
python runs/compare.py --runs artifacts/runs/retrain/s42b/video/runs.json --only E4

# 只导出比较图和原始数据。
python runs/compare.py --runs artifacts/runs/retrain/s42b/trajectory/runs.json --charts-only
```

所有组的 cfg2 单 run 命令都用自身 `split_manifest.json`；省略该参数时会自动读取。
视频单 run 使用 `runs/video/evaluate_video.py`，其参数与上述相同。
视频 cfg3 单 run 同样同时传入 cfg3 `--data-root` 与 `--split-manifest`。
单组训练仍可直接调用现有 `train_trajectory.py` / `train_video.py`，
使用 `configs/retrain/<modality>/<group>.json` 和 `--run-dir` 指定全新目录。
两个训练入口也支持 `--split-manifest <已有清单>`，可直接复用指定划分。

增量评估只重新计算所选组，并直接覆盖其已有输出；未选组读取已保存的
`metrics.json`、`metadata.json`、训练配置和划分。未选组缺少指标或样本信息时明确提示并跳过。
汇总不以 `run_info.json` 或旧评估指纹作为前置条件，也不会自动重评未选组。
新评估仍记录 checkpoint、配置、协议、查询/候选顺序和产物指纹供追溯。
cfg2 使用各 run 自身 test，cfg3 使用该 batch 的冻结模态清单；
若改变了指标算法或评估协议，应显式重评所有需要比较的组后再汇总。

## 产物

```text
artifacts/runs/retrain/s42b/
  pipeline.json
  audit.json
  plots.json                  # 图片路径 → 数据文件依赖和绘图参数
  trajectory/                 # video/ 采用相同结构
    configs/T0.json ...        # 最近一次流水线生成的配置快照
    runs.json
    summary.json
    logs/
    T0/ ... T5/ N1/ N2/ N3/
      params.json
      data_manifest.json
      split_manifest.json
      split_info.json
      protocols.json
      checkpoint_info.json
      status.json
      best_model.pth
      loss_model.pth            # 轨迹首次 train loss < 0.73 时保存
      trainable.json            # 仅视频：实际可训练参数和 block
      curves.json / curves.png
      evaluations/cfg2/ ...
      evaluations/cfg3/ ...
      human_robustness/cfg2/ ...  # 独立阶段生成
    charts/
      cfg2.png                  # 各 run 自身 test，包含 T4/E6
      cfg3.png                  # 共享 cfg3 test，两个模态都包含全部组
      cfg2.json / cfg2.csv       # 含所有组、方向、层级、指标与来源
      cfg3.json / cfg3.csv
      curves.png / curves.json
```

每个评估目录保存 `metrics.json`、`metadata.json`、`cases.json`、
`similarity_matrix.npy`、两侧 embedding、热图、数据库存与 `run_info.json`。
两种模态的 task / scene、两个检索方向都报告 `MPR-all`，只保存到评估指标、
比较 JSON/CSV 和汇总数据；柱状图仍只显示 `R@1/5/10`、`MRR`、`NDCG@10`。
`MPR-all` 对每个有正样本的查询计算所有正样本的平均排名除以候选数，
再对这些查询等权平均并乘以 100。排名从 1 开始，单位是百分比，越低越好；
例如 10 表示正样本平均位于候选库前 10%。无正样本的查询不参与平均。
已有 `Mean Percentage Rank` 仍只统计首个正样本，取值为 0–1，含义不变。
旧评估需要重新运行评估阶段才能加入新指标，无需重训：

```bash
python runs/retrain.py --batch s42 --stage eval --skip-existing
```

原始 backbone 权重按模型名集中保存，不随 batch 重复复制：

```text
artifacts/models/video/
  facebook--vjepa2-vitl-fpc64-256.pth
  MCG-NJU--videomae-base.pth
```

原始骨干评估优先读取 run 内历史 `backbone.pth`，缺失时使用上述公共权重。
公共权重尚不存在时，首次评估将已加载的预训练骨干保存到公共位置，后续批次直接复用。
s44 的两份权重已迁移至此，原 run 内副本移除；文件内容保持一致。
各 run 仍单独保存参数、划分、指标和评估来源，checkpoint 指纹对应实际使用的权重文件。
构建骨干结构和 processor 仍使用 Hugging Face 模型缓存，公共 `.pth` 不替代该缓存。

cfg2 主图中的 T4/E6 使用不同测试集合，图中标记并保留样本规模。
重新导出主图会清除旧的 `cfg2_task.png`，不再生成单独的 task 图。
训练曲线的 validation 指标使用各自 run 的验证集，T4/E6 的验证协议也不同。

训练曲线、评估热图和比较图在绘制前更新批次根目录的 `plots.json`。
清单以图片相对路径为键，记录绘图类型 `kind` 和直接读取的数据文件 `inputs`。
汇总图另有 `upstream` 原始来源，有绘图参数时才保存 `options`；空字段省略。
数据依赖均为相对批次根目录的路径列表，不记录 SHA-256。
比较图追溯到各 run 的 `metrics.json`、`metadata.json`、`params.json` 和划分，汇总曲线追溯到各 run 的
`curves.json`。例如：

```json
{
  "trajectory/N1/curves.png": {
    "kind": "run_curves",
    "inputs": ["trajectory/N1/curves.json"]
  }
}
```

`runs/redraw.py` 先扫描已有 JSON/NPY、补齐整个批次的清单，再依次覆盖所有 PNG。
覆盖两种模态的单 run 训练曲线、cfg2/cfg3 两类热图、主比较柱状图和汇总曲线；
PNG 已删除时也能恢复。旧批次首次执行会自动补齐清单。
该入口只重绘已保存的数据快照，不训练、不加载 checkpoint、不重新计算评估指标，
无需 CUDA、模型缓存或 RH20T 原始文件。
已有汇总 JSON 时优先从中恢复；汇总目录或 JSON 缺失时，从原 run 的评估与曲线重建。
如果更新了评估指标，先用 `--stage charts` 重新汇总，再用该入口重绘全部图。
缺少依赖文件或发现无法重绘的 PNG 时，在覆盖图片前报错。
自定义批次路径用 `python runs/redraw.py --root <批次目录>`。

## 历史检索建库模板

`configs/retrain/library_cfg2.json` 与 `library_cfg3.json` 仍绑定 s42 的 E4/N0 历史 checkpoint，
尚未随本次 s44 视频编号迁移更新。当前源 E4 已是机器人单适配头，
不能据此推断旧 s42/E4 的训练结构。使用这些模板时，需要将模型配置指向所选 run 的
`params.json`，并同步 checkpoint、batch 测试清单与输出目录，再执行建库。

```bash
python runs/retrieval/build_retrieval_library.py --config configs/retrain/library_cfg2.json
```

## 历史工程验证

在 `torch2` 环境执行 `python -m pytest -q tests`，覆盖划分、缓存与各评估入口。

`artifacts/runs/retrain/smoke_cpu/` 仅用于真实数据上的流水线验证：
八组轨迹各训练两轮，小模型宽度 16、1 层、最多 8 帧；T3 为 1+1。
该目录的指标和图不代表正式实验结果，正式 `s42` 批次使用上面的完整配置。
已验证 cfg2/cfg3 独立评估、只更新 T2/N1 后合并其余六组、完成训练复用，
以及 N1/N2 在两个评估集合上的独立人手变换入口。

视频入口由自动化集成测试覆盖训练、240 对固定测试样本评估、
当时的 task-held-out（旧 E3，现 E6）按自身 test 合入主图、默认 cfg3 评估、
原始骨干权重冻结复用和比较数据输出；
测试使用本地小骨干与模拟解码，
共享清单准备和视频文件审计仅检查文件存在；真实视频解码与文件指纹由视频训练、评估入口处理。
