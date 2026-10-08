# RH20T retrain

新实验配置统一放在 `configs/retrain/`，产物放在
`artifacts/runs/retrain/<batch>/{trajectory,video}/<group>/`。旧配置和 run 保留。

本文实验矩阵对应原 retrain 模板；s43 轨迹模板及新 batch 入口见
[s43 轨迹实验](trajectory/S43.md#新数据存储结构的入口)。
数据集内的夹爪参数、反馈与几何缓存格式及来源集中记录在
[当前子集存储与夹爪来源](RH20T_SPLIT_AND_MANIFESTS.md#当前子集存储与夹爪来源)。

正式训练默认：轨迹 70 epoch（T3 为 10+60）、视频 5 epoch、seed 42。
checkpoint 仅依据 validation human→robot task MRR 选择。
轨迹初始学习率 2e-4，前 10 轮保持不变，第 11–70 轮按余弦曲线降至 2e-5。
T3 两个阶段共用同一调度，不在微调开始时重置；每轮实际学习率记录在
`curves.json:learning_rate` 和训练日志中。短跑预算不超过 10 轮时保持初始学习率。
视频训练只保存 `best_model.pth`；轨迹仍保存最佳与最后一轮的权重。

## 协议与命名

| 名称 | 含义 |
|---|---|
| `E1` / `E2` | V-JEPA dual-head / VideoMAE dual-head |
| `E3` / `E4` / `E5` | V-JEPA task 隔离 / 模态内损失 / InfoNCE |
| `E6s` / `E6d` | V-JEPA 冻结 + 机器人单适配头 / 最后两层微调 + 双适配头 |
| `RawJ` / `RawM` | 原始 V-JEPA / VideoMAE，仅评估，不训练投影头 |

新轨迹组统一使用 `human_hand_3d.npy`、真实时间与缺测标记、
机器人 `relative_rot6d`，batch size 8。视频训练和评估 batch size 统一为 4。
E1–E5 均冻结骨干并训练双适配头；E6s 冻结骨干，人类侧不加适配头；
E6d 共用一套骨干，解冻最后两个 Transformer block 并训练双适配头。
三种设置都训练共享投影层，实际可训练参数和 block 编号保存到 `trainable.json`。
`N0/N1/N2` 只改变人手坐标表示，所有组保持同模态的有效相机集合一致。

轨迹组的具体区别：

| 组 | 人手点数 / 输入维度 | 人手表示 | 训练与损失 | 正式预算 | cfg2 划分 |
|---|---|---|---|---|---|
| T1 | 6 / 18 | 基座 XYZ | 跨域对比 | 70 | 共享 scene |
| T2 | 6 / 18 | 基座 XYZ | 跨域 + 域内增强一致性 | 70 | 共享 scene |
| T3 | 6 / 18 | 基座 XYZ | 域内预训练，再跨域 + 域内联合微调 | 10 + 60 | 共享 scene |
| T4 | 6 / 18 | 基座 XYZ | 跨域对比 | 70 | task 隔离，目标 test 120 / val 50 |
| T5 | 21 / 63 | 基座 XYZ | 跨域对比 | 70 | 共享 scene |
| N0 | 21 / 63 | 基座 XYZ | 跨域 + 域内增强一致性 | 70 | 共享 scene |
| N1 | 21 / 63 | 首有效腕点相对坐标 | 跨域 + 域内增强一致性 | 70 | 共享 scene |
| N2 | 21 / 63 | 首有效掌部相对坐标系 | 跨域 + 域内增强一致性 | 70 | 共享 scene |

八组统一按 task 定义训练正样本，评估同时报告 task / scene 两层指标。
联合训练的域内权重为 0.3，域内 task 正样本权重为 0.25；增强噪声标准差
0.005 m、最大旋转 10°。统一模型宽度 128、5 层、投影维度 128、初始学习率 2e-4，
双侧最多 256 帧。cfg3 对八组使用同一评估集合。

共享协议按大 batch 存在 `artifacts/manifests/retrain/<batch>/<cfg>/`，
同一 batch 内共享，两个 cfg 都采用以下格式：

```text
metadata.json                  # schema_version=3，数据根目录、快照指纹、两份划分路径
data_manifest.json             # 原轨迹数据库存格式
split_manifest_trajectory.json # train/val/test：task_xxxx/scene_x
split_manifest_video.json      # train/val/test：rh20t::task_xxxx::scene_x::camera
```

两份划分只有 `train/val/test` 三个列表，格式与各 run 的 `split_manifest.json` 相同。
视频按轨迹的 scene 顺序逐一展开，每个 scene 两条固定相机样本；读取时校验两份文件
的集合、划分和顺序一致。cfg2 随机固定 120 test / 50 val，其余共同有效 scene 训练；
cfg3 每个本地 task 固定两个评估 scene，train/val 为空。
`--batch` 同时选择 run 与清单目录；准备阶段优先校验并复用已有批次文件。
已有 batch 不原地修改 seed 或数据快照，不同 batch 使用独立文件。
协议仍为 `retrain_v2`，旧 schema_version=2 的清单和旧 run 继续可读。

固定抽样 seed 为 42，与训练 seed 分开。cfg2 从 428 个共同有效 scene 抽样；
cfg3 的 64 个 task 共 128 个评估 scene。视频划分文件显式冻结两个 camera，
人机路径按 sample ID 解析。生成新批次时检查均匀 16 帧解码；视频文件指纹继续记录
在 run 的数据快照中，训练期间文件变化或读取失败直接报错。

普通实验配置默认 `split.use_fixed_val=true`。设为 `false` 时，从排除 test 的可用
scene 中按训练 seed 随机选择 `split.val_count` 个验证 scene（默认 50）；原固定 val
不再单独保留。实际 train/val/test 保存到 run 的 `split_manifest.json`。

T4/E3 保留 task 隔离，seed 42，不使用共享 cfg2 test/val。
通过 `split.scene_counts={"test":120,"val":50}` 按 scene 数量选择整组 task，
剩余 task 全部训练。联合寻找最接近目标的划分，可达到目标时使用精确数量，
不能达到时选择总偏差最小的数量，优先保持测试规模；实际数量写入 `split_info.json`。
三个集合的 task 必须互斥，所有同 task 的有效 scene 随组进入对应集合。
当前两种模态都能达到 120 test / 50 val，各为 30 / 13 个 task，训练使用剩余 66 个 task。

当前数据的 scene 数量：

| 组别 | train | val | test |
|---|---:|---:|---:|
| 普通轨迹组 | 258 | 50 | 120 |
| T4（新目标） | 258 | 50 | 120 |
| 普通视频组（新 batch 共享清单） | 258 | 50 | 120 |
| E3（新目标） | 264 | 50 | 120 |

视频每 scene 两对，普通 cfg2 test 为 240 查询 / 240 候选。
轨迹使用所有冻结有效相机，其查询数量可能与视频不同；共享 scene 组的集合一致。
cfg3 task 编号是该子集内部标签，不表示已建立 cfg2↔cfg3 任务编号映射。

cfg2 独立评估统一读取每个 run 的 `split_manifest.json:test`，保留实际训练划分。
RawJ/RawM 没有训练阶段，首次评估会按它们的配置生成并保存同格式的 cfg2 划分。
cfg3 使用该 batch 的 `cfg3/split_manifest_<modality>.json:test`；
不能将 cfg2 的训练划分用于 cfg3 数据。已有 run 的参数和划分仍按原记录复用。
两个模态默认都评估 cfg2、cfg3，每个数据域各输出一张包含全部组的主比较图。
T4/E3 也放进 cfg2 主图，以星号标注 task 划分；图中显示各组测试 scene 数，
JSON/CSV 同时记录各自的清单路径、测试 scene/task 数、查询和候选数。

## 完整入口

在仓库根目录、`torch2` 环境执行。正式新配置要求 CUDA；不会静默改用 CPU。
输入前提是现有 `human_hand_3d.npy` / `tcp_base.npy` 以及
`artifacts/manifests/trajectory_hand_3d_v1/{cfg2,cfg3}_valid.json` 已准备好；
准备阶段会核验 NPY 指纹，并冻结当前视频配对。

```bash
conda activate torch2

# 只显示计划，不创建文件或加载模型。
python runs/retrain.py --batch s42b --dry-run

# 生成/校验清单 → 审计源文件和划分 → 全部训练 → 独立评估 → 比较图。
python runs/retrain.py --batch s42b

# 已完成的训练和有效评估可以复用；缺少最佳模型或未完成的 run 会重训。
python runs/retrain.py --batch s42b --skip-existing

# 只跑某一模态。
python runs/retrain.py --batch s42b --modality trajectory

# 更换训练 seed；共享 test 与默认固定 val 不变。
python runs/retrain.py --seed 43 --batch s43
```

也可用 `bash scripts/retrain.sh ...`，通过 `PYTHON_BIN` 指定 Python。
配置修改后可直接在原 batch 重跑，不检查批次配置或注册表是否变化。
批次 `configs/` 每次更新到源配置；已有 run 的 `params.json` 保留真实训练设置，
只有实际重训时才更新，独立评估继续读取该文件。

`--skip-existing` 是流水线复用，不是恢复中断训练的 optimizer 状态。
训练阶段的判断规则：

- 不加 `--skip-existing`：从头训练所有所选组，已有 run 先归档。
- 加 `--skip-existing`：同时存在 `best_model.pth`、`params.json`，且
  `status.json:state=complete` 时跳过训练；其它情况从头重训，不比较配置差异。
- 重训时沿用原 run 的 `split_manifest.json`，保持 train/val/test 及顺序。
  只有没有该清单时，才按当前配置生成划分；`last_model.pth` 不参与判断。
- 旧 run、过期比较图及对应日志归档，原评估缓存随旧 run 一起移走。

例如，在 `s42` 使用新配置重训全部轨迹并评估绘图：

```bash
python runs/retrain.py --batch s42 --modality trajectory

# 仅补训缺少最佳模型或未完成的 run，并复用其它已完成训练。
python runs/retrain.py --batch s42 --skip-existing
```

在 `--skip-existing` 模式下，修改配置本身不会触发重训。可删除目标 run 的
`best_model.pth` 后再跑，或用不带 `--skip-existing` 的 `--only` 命令指定重训组。

## 替换 task 划分实验

下面命令仅替换 `s42b/trajectory/T4` 和 `s42b/video/E3`，正式预算是 70 / 5 epoch。
当前有效 scene 总数为轨迹 428、视频 434；保留 test 120 / val 50 后，train 分别为
258 / 264。重训沿用已有划分；没有清单时按 task 隔离生成目标划分。
评估读取每个新 run 的清单。需要重新划分时，先删除所选 run 的
`split_manifest.json`，然后执行替换命令。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python runs/retrain.py --batch s42b --only T4 E3 --replace

# 先查看计划；不移动文件或加载模型。
python runs/retrain.py --batch s42b --only T4 E3 --replace --dry-run

# 分开执行：只替换并训练这两组，再独立评估和汇总绘图。
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python runs/retrain.py --batch s42b --stage train --only T4 E3 --replace
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python runs/retrain.py --batch s42b --stage eval --only T4 E3
```

默认完整入口依次校验协议、审计、替换训练、评估 cfg2/cfg3，并与其它 run 的有效评估
合并更新主图。旧 run、对应日志及过期的汇总图/数据移到
`artifacts/runs/retrain/archive/<batch>/<时间>/<modality>/`，其它组的 checkpoint、评估和
数据划分保留。绘图清单同步移除过期条目，完成训练/评估后记录新依赖。
旧 T4/E3 的 80%/10%/10% 划分仅保存在归档结果中。
`--replace` 是兼容保留的显式替换入口，必须搭配 `--only` 和 `--stage all/train`，
每次执行都会从头训练所选组，也沿用已有划分；
`--skip-existing` 与替换互斥。

## 各阶段入口

```bash
# 只准备共享清单和数据库存快照。
python runs/prepare_splits.py --batch s42b

# 只审计；--files 校验所有冻结源文件。
python runs/audit_splits.py --files --output artifacts/runs/retrain/audit.json

# 只训练；也可选择部分组。两阶段 --traj-epochs 指总预算。
python runs/retrain.py --batch s42b --stage train
python runs/retrain.py --batch s42b --stage train --modality trajectory --only T2 N1

# 只评估和绘图；两个模态都默认 cfg2+cfg3。
python runs/retrain.py --batch s42b --stage eval

# 使用已训练的 checkpoint，复用有效评估；清单变化的结果会重新评估。
HF_HUB_OFFLINE=1 python runs/retrain.py --batch s42b --stage eval --skip-existing

# 校验评估结果，重新汇总比较数据并导出主图。
python runs/retrain.py --batch s42b --stage charts

# 先记录依赖，再从已有数据重绘此批次全部图片，无需模型或原始数据集。
python runs/redraw.py --batch s42b

# 只补齐/更新图表依赖清单，不重绘。
python runs/redraw.py --batch s42b --index-only

# N0–N2 的独立人手变换稳健性评估，使用相同冻结查询/候选。
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
  --split-manifest artifacts/manifests/rh20t_shared_v1/cfg3_split_manifest.json \
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

增量更新前需要先建立所有组在新协议下的评估结果。未选组若缺失或过期，合并报错，
不会偷偷重算未选组，也不会混入旧数据。缓存校验包括 checkpoint、协议、数据库存、
评估代码、采样配置、查询/候选集合及顺序、矩阵与 embedding 一致性和产物文件指纹。
cfg2 test 与该 run 的 train/val 重叠时拒绝评估。
cfg2 允许不同 run 使用不同 test 集合；cfg3 仍校验各组查询/候选集合完全一致。
已有按共享 cfg2 清单记录的缓存需要重新评估一次，训练 checkpoint 可以继续复用。

## 产物

```text
artifacts/runs/retrain/s42b/
  pipeline.json
  audit.json
  plots.json                  # 图片路径 → 数据文件依赖和绘图参数
  trajectory/                 # video/ 采用相同结构
    configs/T1.json ...        # 该批次实际训练配置
    runs.json
    summary.json
    logs/
    T1/ ... T5/ N0/ N1/ N2/
      params.json
      data_manifest.json
      split_manifest.json
      split_info.json
      protocols.json
      checkpoint_info.json
      status.json
      best_model.pth
      last_model.pth            # 仅轨迹
      trainable.json            # 仅视频：实际可训练参数和 block
      curves.json / curves.png
      evaluations/cfg2/ ...
      evaluations/cfg3/ ...
      human_robustness/cfg2/ ...  # 独立阶段生成
    charts/
      cfg2.png                  # 各 run 自身 test，包含 T4/E3
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

原始 backbone 在 `RawJ/RawM/backbone.pth` 冻结权重，后续评估复用该版本。
cfg2 主图中的 T4/E3 使用不同测试集合，图中标记并保留样本规模。
重新导出主图会清除旧的 `cfg2_task.png`，不再生成单独的 task 图。
训练曲线的 validation 指标使用各自 run 的验证集，T4/E3 的验证协议也不同。

训练曲线、评估热图和比较图在绘制前更新批次根目录的 `plots.json`。
清单以图片相对路径为键，记录绘图类型 `kind` 和直接读取的数据文件 `inputs`。
汇总图另有 `upstream` 原始来源，有绘图参数时才保存 `options`；空字段省略。
数据依赖均为相对批次根目录的路径列表，不记录 SHA-256。
比较图追溯到各 run 的 `metrics.json` / `run_info.json`，汇总曲线追溯到各 run 的
`curves.json`。例如：

```json
{
  "trajectory/N0/curves.png": {
    "kind": "run_curves",
    "inputs": ["trajectory/N0/curves.json"]
  }
}
```

`runs/redraw.py` 先扫描已有 JSON/NPY、补齐整个批次的清单，再依次覆盖所有 PNG。
覆盖两种模态的单 run 训练曲线、cfg2/cfg3 两类热图、主比较柱状图和汇总曲线；
PNG 已删除时也能恢复。旧批次首次执行会自动补齐清单。
该入口只重绘已保存的数据快照，不训练、不加载 checkpoint、不重新计算评估指标，
无需 CUDA、模型缓存或 RH20T 原始文件。
如果更新了评估指标，先用 `--stage charts` 校验并重新汇总，再用该入口重绘全部图。
缺少依赖文件或发现无法重绘的 PNG 时，在覆盖图片前报错。
自定义批次路径用 `python runs/redraw.py --root <批次目录>`。

## 新检索建库配置

`configs/retrain/library_cfg2.json` 与 `library_cfg3.json` 使用新共享 test、
新 E4/N0 配置和 `s42` checkpoint，产物进入 `artifacts/retrieval/retrain/s42/`。
新建库预检校验共享协议与训练绑定、模型/数据配置和 test 泄漏。
旧 `configs/retrieval/` 文件及用户现有修改保留，避免把旧 checkpoint 当作新实验。

```bash
python runs/retrieval/build_retrieval_library.py --config configs/retrain/library_cfg2.json
```

## 工程验证

在 `torch2` 环境执行 `python -m pytest -q tests`，覆盖划分、缓存与各评估入口。

`artifacts/runs/retrain/smoke_cpu/` 仅用于真实数据上的流水线验证：
八组轨迹各训练两轮，小模型宽度 16、1 层、最多 8 帧；T3 为 1+1。
该目录的指标和图不代表正式实验结果，正式 `s42` 批次使用上面的完整配置。
已验证 cfg2/cfg3 独立评估、只更新 T2/N1 后合并其余六组、完成训练复用，
以及 N1/N2 在两个评估集合上的独立人手变换入口。

视频入口由自动化集成测试覆盖训练、240 对固定测试样本评估、
E3 按自身 test 合入主图、默认 cfg3 评估、原始骨干权重冻结复用和比较数据输出；
测试使用本地小骨干与模拟解码，
真实视频的配对、解码和文件指纹则由共享清单准备与全文件审计校验。
