# 轨迹第一批实施计划与验收

## 范围

保留五层 Transformer，新增四组累积消融：F0 有效样本/原始 7D；F1 位置+6D 旋转；F2 相对位姿；F3 时间与缺测。四组统一修复增强的四元数顺序并使用相同有效 manifest、split、主干和训练预算。基线不再保留已知错误增强。

本批不重建人手全局位移，不猜测工具标定。可提供有来源的 TCP-to-reference 刚体变换；未配置时使用已有 TCP 坐标，不能称为完成跨工具接触中心标定。

## 执行计划

1. 有效样本 manifest：全量检查、记录排除原因和文件哈希；划分前过滤；加载禁止替换身份。验收无跨 split 替换、文件变化可检测。
2. 几何表示：明确 wxyz/xyzw，增强同步位置/姿态；rot6d、相对位姿和可选工具变换。验收 q/-q 等价、Z90 旋转、SE(3) 不变性。
3. 时间缺测：保存帧索引/机器人秒级时间、人侧明确使用帧进度；一次采样；区分观测/缺测/padding，时间特征进入编码器。验收长缺口不压缩、mask 正确。
4. 贯通入口：训练、评估和建库共用数据/模型配置；保存配置、有效 manifest、split 与评估 provenance；增加四组配置和一键闭环。
5. 验证：相关回归测试与全仓测试；四组真实 cfg2 短训、cfg2 test 与 cfg3 all-test 评估、扰动指标。短训用于工程闭环，不作为收敛性能结论。
6. 交付：记录实际结果、限制及正式训练/独立评估命令。

## 进度

- 数据/几何/时间处理、四组入口和真实短训评估闭环已完成。

## 已实现的入口与配置

| 配置 | robot 输入 | 时间/缺测 | 共同设置 |
|---|---|---|---|
| `F0_valid_raw7.json` | 原始 xyz + wxyz | 压缩后的有效人手帧，作为消融基线 | 有效 manifest、正确 wxyz 增强、一次采样 |
| `F1_rot6d.json` | xyz + rot6d，9 维 | 同 F0 | 五层 Transformer，128 维 |
| `F2_relative.json` | 初始 TCP 参考系下的位置 + rot6d，9 维 | 同 F0 | 同一 clean split |
| `F3_relative_time.json` | 同 F2 | 保留缺测观测、时间上下文 | 同一训练预算与评估库 |

四组都使用 6 个人手点（18 维），人手形状/运动双路属于后续批次。四组均将输入上限设为 human/robot 各 256 帧，并设 `tcp_sample_factor=1`，防止二次采样。F0/F1/F2 使用序号均匀选点；F3 使用实际时间/原帧索引选取最近的真实记录，不插值、不制造跨缺口运动。

F3 的三个上下文特征为：相对进度、`log1p(相邻原始时间差)`、观测有效标记。robot 时间为原始毫秒 timestamp 转换后的相对秒；human 时间为原始 frame index，未声称恢复真实秒。缺测帧使用可学习 missing token，同时仍携带时间上下文；padding 不参与注意力。若时间数据未传入 F3，模型报错，避免静默退化。

新增 `geometry.py` 统一 robot 特征转换；所有增强仍作用在原始 TCP 上，之后由模型内部转换为网络特征。训练、独立评估、扰动评估和检索建库均通过共享 factory 建立 dataset/model。

## 首次准备数据

以下命令在仓库根目录、`torch2` 环境执行：

```bash
conda activate torch2
python runs/trajectory/prepare_first_batch.py
```

生成：

```text
artifacts/manifests/trajectory_first_batch/
  cfg2_valid.json
  cfg2_split.json
  cfg3_valid.json
  cfg3_split.json
```

有效清单包含源文件 SHA-256、相机覆盖率及排除原因；后续每次加载校验哈希。准备命令拒绝覆盖已有冻结清单；已有本轮清单时直接进入训练。数据变化后应使用新的 `--output-dir`，并让实验指向新版本。

当前有效规模：cfg2 为 429 个 scene、1,151 个 camera pair，划分 343 train / 43 val / 43 test；cfg3 为 255 个 scene、745 个 camera pair，全部用作外域测试。task ID 是当前子集本地编号，不作为跨 cfg 任务语义映射。

## 一条命令完成正式训练与评估

```bash
python runs/trajectory/run_first_batch.py --epochs 60
```

该入口依次执行 F0–F3，每组完成：

1. 在 cfg2 train 训练；仅按 cfg2 validation 的 task MRR 选 checkpoint。
2. 用保存的 checkpoint 和配置评估 cfg2 test。
3. 保持模型配置，切换数据根与对应 manifest，评估 cfg3 all-test。
4. 在 cfg2 test 测试 q/-q、X 平移 0.1 m、绕 Z 轴旋转 90°。
5. 验证 matrix 与 embedding 点积一致、Hit@1/MRR 可复算、样本无重复；检查对应模型的几何不变性阈值，写入 `summary.json`。

输出默认位于 `artifacts/runs/trajectory/first_batch_<时间戳>/`。也可以指定新目录和部分组：

```bash
python runs/trajectory/run_first_batch.py \
  --epochs 60 \
  --groups F2_relative F3_relative_time \
  --run-root artifacts/runs/trajectory/first_batch_relative_60ep
```

自定义数据/manifest 位置时使用 `--dataset-root` 和 `--manifest-dir`。若 manifest 尚不存在，一键入口会自动调用准备脚本。每个步骤的完整 stdout/stderr 保存在实验根目录的独立日志中；失败会立即停止，已有产物保留。

## 单独训练与评估命令

训练 F3（新建目录，不覆盖旧 run）：

```bash
python runs/trajectory/train_trajectory.py \
  --config configs/trajectory/F3_relative_time.json \
  --epochs 60 \
  --run-dir artifacts/runs/trajectory/F3_relative_time_60ep
```

也支持既有 shell 入口：

```bash
CONFIG=configs/trajectory/F3_relative_time.json \
bash scripts/train_trajectory.sh --epochs 60
```

以下 `TRAJECTORY_RUN` 改成实际训练输出目录：

```bash
TRAJECTORY_RUN=artifacts/runs/trajectory/F3_relative_time_60ep

python runs/trajectory/evaluate_retrieval.py \
  --config "$TRAJECTORY_RUN/params.json" \
  --checkpoint "$TRAJECTORY_RUN/best_model.pth" \
  --split test \
  --split-manifest "$TRAJECTORY_RUN/split_manifest.json" \
  --output-dir "$TRAJECTORY_RUN/evaluations/cfg2_test"

python runs/trajectory/evaluate_retrieval.py \
  --config "$TRAJECTORY_RUN/params.json" \
  --checkpoint "$TRAJECTORY_RUN/best_model.pth" \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --all-as-test \
  --output-dir "$TRAJECTORY_RUN/evaluations/cfg3_all_test"

python runs/trajectory/evaluate_robustness.py \
  --config "$TRAJECTORY_RUN/params.json" \
  --checkpoint "$TRAJECTORY_RUN/best_model.pth" \
  --output-dir "$TRAJECTORY_RUN/robustness_cfg2"
```

评估必须使用该 run 的 `params.json`。同为 9 维输入的 F1/F2 也不能互换配置；入口会检查训练配置中的表示/时间设置是否一致。cfg3 评估通过 `data_manifests` 选择 cfg3 清单，不会误用 cfg2 的清单或 split。

## 工具参考点接口与本批限制

`reference_frames` 可按数据根目录名配置，每项必须包含 `source` 和合法的 4×4 `tcp_to_reference`。模型按 `T_reference = T_tcp @ C` 转换，再计算相对位姿。测试覆盖工具偏移随 TCP 旋转的情形。

四组默认配置没有填写未经验证的变换，仍以数据中已有 TCP 为参考。若未来补齐 cfg2/cfg3 的工具标定，应将两者有来源的变换一同放进配置；配置了映射却缺少目标 cfg 时明确报错，不自动使用其他机器人的偏移。原始 raw7 模式不支持此工具变换，需使用 rot6d 模式。

本批不解决缺失的原任务映射、全局手平移、真实人演示时间戳或 robot execution 去重；评估仍为 camera 候选协议。新的有效 split 与旧训练 split 不同，成绩不能直接作为旧 T2 的同协议提升。旧 checkpoint 的基础网络参数结构仍可加载，但清洗后的 dataset 不再复现旧的递归换样本行为；带无效 scene 的旧 split 会显式报错。

## 验收结果

实际短训闭环及最终检查结果如下。短训采用完整 cfg2 train/val/test 和 cfg3 all-test，每组 2 epochs，用于验证链路与几何性质，不用于判断模型是否收敛。


实际执行命令：

```bash
python runs/trajectory/run_first_batch.py   --epochs 2   --run-root artifacts/runs/trajectory/first_batch_closed_loop_20260922
```

四组全部完成 2 epochs、cfg2 test、cfg3 all-test 与扰动评估，无步骤跳过。cfg2 test 为 113 个相机样本、43 个 scene；cfg3 为 745 个相机样本、255 个 scene。

| 组别 | cfg2 task Hit@1 | cfg3 task Hit@1 | q/-q 最大 embedding 差 | X 平移 0.1m 最大差 | Z 旋转 90° 最大差 |
|---|---:|---:|---:|---:|---:|
| F0_valid_raw7 | 2.65% | 2.15% | 3.50e-01 | 4.37e-02 | 3.05e-01 |
| F1_rot6d | 1.77% | 0.94% | 0.00e+00 | 9.26e-03 | 3.18e-01 |
| F2_relative | 1.77% | 0.27% | 0.00e+00 | 7.45e-08 | 8.20e-08 |
| F3_relative_time | 0.88% | 2.01% | 0.00e+00 | 8.94e-08 | 1.19e-07 |

以上准确率仅为 2 epochs 的工程验收结果，不能据此判断表示方案优劣。几何结果则验证：rot6d 消除了四元数符号差异；relative_rot6d 消除了整体平移/旋转坐标变化的影响。这些结论只针对机器人侧，尚未使人侧具备完整旋转不变性或补回全局位移。

最终验证：

- `python -m pytest -q tests`：73 passed；只有 PyTorch nested tensor 的已有接口提示。
- 八组独立评估的 matrix/embedding 一致性、Hit@1/MRR 复算及 scene-camera 唯一性全部通过。
- F1/F2/F3 的 q/-q 检查通过；F2/F3 的平移与旋转不变性检查通过。
- 五个 CLI 的 `--help` 均成功；`git diff --check` 通过。
- 同域评估和建库优先读取训练目录保存的有效 manifest 快照，避免原清单后续变化静默改变输入范围。

产物：[闭环 summary.json](../../artifacts/runs/trajectory/first_batch_closed_loop_20260922/summary.json)。每组目录含 `params.json`、`data_manifest.json`、`split_manifest.json`、`best_model.pth`、`checkpoint_info.json`、训练曲线、两域评估矩阵/embedding/metadata/metrics 和扰动结果。评估 `run_info.json` 保存 checkpoint 哈希、配置及数据清单哈希。

另已通过 F3 在 cfg3 的实际轨迹建库检查：64 个任务各选 1 scene、1 camera，生成 64 条 human 与 64 条 robot 特征。全部 128 条特征与独立评估中对应样本一致，最大绝对差 8.94e-08。详见 [建库一致性结果](../../artifacts/runs/trajectory/first_batch_closed_loop_20260922/library_consistency.json)。
