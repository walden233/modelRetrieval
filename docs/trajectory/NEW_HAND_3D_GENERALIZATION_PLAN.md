# 新 3D 人手数据的轨迹训练与 cfg2→cfg3 泛化计划

## 目标与边界

目标是在 cfg2 训练、仅用 cfg2 验证集选模型，最后在 cfg2 留出集和完整 cfg3 上评估跨机器人泛化。人侧改读 `human_hand_3d.npy` 的深度反投影基座系米制坐标；机器人仍读 `tcp_base.npy`。这样人手腕部位移进入模型，但 **基座系相同不等于机器人关键点/工具接触点相同**，也不自动带来 SE(3) 不变性。

当前已落地数据加载、冻结清单、N0–N2 表示、固定候选库扰动评估和一键训练评估入口；N3/N4 仍是后续实验。旧 `human_pose.npy` 路径及 F0–F3 结果保留，不能把两版清单或 checkpoint 混用。这里的“基座系”来自每个相机的 `T_base_from_camera`，其来源和修正记录在场景数据中；不从 MediaPipe 局部 world landmarks 推断全局位置。

## 数据契约与第一阶段加载器

每个 scene 只允许 `human_hand_3d.npy` schema v2 与 `tcp_base.npy` 的相机 serial 交集。先验证帧索引、RGB 毫秒时间戳、机器人毫秒时间戳单调，基座变换有限且为刚体，TCP 7D 有效。选定一条相机局部 `hand_track_id`：以完整深度基座关键点、非几何可疑、非追踪歧义的有效帧数最大者为准；同分取小 ID。该规则只是可复现的候选手选择，不等于任务手真值。清单记录选中的 ID、有效帧数/覆盖率、双手/歧义/几何可疑计数、排除原因及两份 NPY 的 SHA-256。至少两个有效帧才纳入该相机；没有有效相机的 scene 在划分前剔除。读失败或哈希变化立即报错，不替换样本。

输出沿用现有 `collate_trajectories` 协议：`human_poses [T,21,3]`（或显式 6 点消融）、`human_observed [T]`、`human_frame_indices [T]`、`human_times [T]`（RGB 原始毫秒时间差换算成秒）、`tcp_bases [R,7]`、`robot_times [R]`（各自相对首帧的秒）。无手、局部追踪不明和深度/几何无效的 RGB 帧仍在序列中，坐标置零并用 missing token 标记；padding 另有 `human_mask`。按真实时间栅格下采样并保留首尾、至少一个有效观测，不跨缺口插值。人和机器人示范不是同步录制，两个时间轴分别归零，不能逐帧配对。模型输入维数随 21/6 点分别为 63/18；第一阶段默认 21 点，以保留掌部 MCP 几何。

读取的是基座点，且仅使用完整的所选关键点集合；不把无效深度点用局部 MediaPipe 点硬拼成全局点。记录 `hand_track_ambiguous` 和 `wrist_middle_span_suspect` 供质量分层。cfg3 中外参 fallback 与少数标定高度修正要单独分层报告，不能藏在平均指标里。

## 对照实验：每次只改变一个关键因素

| 阶段 | 人侧 / 机器人侧输入 | 作用与预期 |
|---|---|---|
| N0（本次可训练基线） | 21 个基座关键点原始 XYZ / F3 的 `relative_rot6d` TCP | 验证新数据、真实秒级时间、缺测和正确 wxyz 增强的闭环；两侧表示还不对称，不作为最终泛化方案。 |
| N1 | 手点减去第一段可靠手腕位置；机器人继续 F3 | 检验去绝对平移是否降低 cfg 差异；保留手相对腕的形状及腕轨迹。锚点须从有效帧选择，不能取缺测首帧。 |
| N2 | 首个可靠掌部姿态定义人侧局部参考系：`R₀ᵀ(pₜ−p₀)`；机器人用首 TCP 姿态的 `relative_rot6d` | 使两侧整体坐标系左乘 SE(3) 时各自特征严格不变。掌部方向来自 `palm_pose_base`，需做退化/左右手/噪声检查。全姿态归一可能去掉有用重力信息，保留 N1 和基座系分支作对照。 |
| N3 | N1/N2 + 两侧一致的刚体扰动及视角一致性损失 | 对每条序列整体旋转/平移所有时间步与点；机器人姿态同步左乘四元数。小角度 yaw 是训练默认，90° yaw/完整 SO(3) 只作诊断或后续消融；跨相机同 scene 手轨迹作为正例前先审核身份与标定。 |
| N4 | 工具参考点统一 | 只有拿到 cfg2/cfg3 各自有来源的 TCP→接触参考点变换后才启用 `reference_frames`。不同机器人 TCP 原点/朝向不一致可能是主因；不能用 cfg3 测试集拟合该变换。 |

F3 已有的清单冻结、无样本替换、wxyz、`relative_rot6d`、时间/缺测上下文、一次采样和验证集 checkpoint 选择全部继承。现有增强是绕**每帧腕点**旋转人手形状、绕**首 TCP**旋转机器人；它没有给人手整体轨迹带来旋转一致性，也没有增强平移。N3 必须用一个序列级刚体变换施加到人侧所有点和时间步，且原始 TCP 的位置/四元数要作同一物理变换。相对化特征与增强分别消融，避免归一化后增强失去训练信号。

## 数据划分、指标及防泄漏

先冻结 cfg2/cfg3 有效数据清单。cfg2 从 428 个合格 scene 中固定保留 43 个（约 1/10）作为 test；每次训练按训练 seed 将剩余 385 个 scene 动态划成 342 train / 43 val。相同 seed 下 N0/N1/N2 使用相同 train/val，便于公平对照；不同 seed 的 train/val 成员变化，但 43 个 test 始终不变。每个 run 会保存最终 `split_manifest.json`，后续评估读取该文件，不重新抽样。cfg3 全量仅作为外域评估，不参与早停、阈值选择或超参搜索。子集里的 `task_####` 是本地编号，**不能**据此跨 cfg 认定同一语义任务，因此 cfg3 的 `task` 检索指标只有在另行建立原任务语义映射后才可解释为跨 cfg 任务识别。此前 cfg3 all-test 的本地 task 指标仅能作域内检索诊断。

每组至少 3 个 seed，同一训练预算、相同数据清单和相同固定 test；每个 seed 内三组共用动态 train/val。报告 cfg2-val/test 与 cfg3 的 scene/camera 候选数量、Hit@1、MRR、NDCG、按任务宏平均、质量分层及 95% bootstrap 区间。另做固定样本的 X/Y/Z 平移、yaw 90°、q/−q、跨相机特征余弦一致性；N2 应检验坐标左乘后人/机器人 embedding 数值不变，N0/N1 只看鲁棒性变化而不宣称严格不变。检索候选中的多个相机不能当独立任务；需另报每 scene 取一相机的结果。训练完成后只用该 seed 的 cfg2-val 定 checkpoint，最后一次性看固定 cfg2-test 和 cfg3 测试结果。

## 执行顺序与验收门槛

1. 数据清单 + 加载器：真实 cfg2/cfg3 数据能读入；抽样检查米制尺度、秒级时间、有效帧坐标、缺测零值与手 ID 稳定；清单版本、哈希与训练配置校验可拒绝混用。**已实现并验证。**
2. 工程短训：N0/N1/N2 各跑 1 epoch 的小规模真实数据闭环，分别完成 cfg2/cfg3 独立检索与固定候选库扰动评估；此步只证明入口可运行，不解读泛化性能。**已完成四个 cfg2 scene、两个 cfg3 scene 的烟测；全量短训尚未执行。**
3. 正式实验：N0/N1/N2 在全量冻结清单上使用同一预算、多 seed 训练；先用 cfg2 验证集选模型，再报告 cfg2 test 与 cfg3。随后才实施 N3/N4，并沿用相同协议。
4. 错误归因：按手轨迹质量、相机、任务、机器人工具/关键点差异分层；若 N2 数值不变而 cfg3 仍差，优先检查工具参考点、任务标签语义、任务手选择、相机/机器人数据分布，而不是加 Transformer 层。

## N0–N2 已实现的入口

`RH20THand3DTrajectoryDataset`、schema v2 新清单和 N0/N1/N2 配置已接入现有 factory/collate/训练/评估入口。三组的唯一数据表示差异由 `data_params.human_representation` 指定；同用 21 点、63 维人侧输入、F3 机器人 `relative_rot6d`、五层模型和相同清单。N1 用首个合格帧的腕点作平移原点；N2 用该帧 `palm_pose_base` 的掌心位置和旋转矩阵作参考。相机外参已用于生成保存的基座点，三组均从该基座点开始。真实数据清单结果如下；“全部 cfg3”指所有**合格**的 cfg3 scene，而不是未经质量筛选的 256 个原始 scene。

| 数据 | 原始 scene | 合格 scene | 合格相机序列 | 划分 / 排除 |
|---|---:|---:|---:|---|
| cfg2 | 434 | 428 | 1,108 | 固定 43 test；剩余 385 个按 seed 动态分成 342 train / 43 val；194 条相机序列少于两帧合格，6 个 scene 无合格相机 |
| cfg3 | 256 | 252 | 698 | 252 个 scene 全部为外域测试；69 条相机序列少于两帧合格，1 条 robot 序列为空，4 个 scene 无合格相机 |

已生成轨迹专用有效库存 `artifacts/manifests/trajectory_hand_3d_v1/{cfg2,cfg3}_valid.json`，其中每个 scene 都冻结两份 NPY 的哈希。跨模态固定测试集单独存放在 `artifacts/manifests/rh20t_shared_v1/cfg2_fixed_test_scenes.json`；旧 `cfg2_split.json` 仅保留用于历史实验复现。详细目录语义见 `docs/RH20T_SPLIT_AND_MANIFESTS.md`。cfg2/cfg3 总数与旧 F3 清单不同，因此不能直接比较旧 F3 的指标。

在仓库根目录、项目 `torch2` 环境运行。当前工作区已生成 v1 清单，准备命令只用于新机器；它拒绝覆盖已有目录：

```bash
python runs/trajectory/prepare_hand_3d.py \
  --output-dir artifacts/manifests/trajectory_hand_3d_v1
```

**一条命令完成 N0–N2** 的训练、cfg2 留出集与 cfg3 合格全量检索，以及两域的固定机器人候选库扰动评估：

```bash
python runs/trajectory/run_hand_3d_n0_n2.py \
  --epochs 60 --seed 42 \
  --run-root artifacts/runs/trajectory/hand_3d_n0_n2_seed42_60ep
```

入口实时显示训练/评估进度并同步保存各步骤日志，同时保存每组的配置、该次动态 train/val 与固定 test 的完整 `split_manifest.json`、checkpoint、两域检索矩阵/embedding/指标、机器人固定候选库 embedding、人手变换后的 embedding、逐变换排序稳定性和总 `summary.json`。N1 的平移、N2 的平移/旋转 embedding 最大差若超过 `1e-4`，入口会报错。可用 `--groups N2_hand_palm_relative` 单独运行一组，或用 `--epochs 1` 做工程短训；短训指标不能用于性能结论。正式多 seed 实验可分别传 `--seed 42`、`--seed 43`、`--seed 44` 并使用不同 `--run-root`：固定 test 不变，剩余数据的 train/val 随 seed 改变。训练中断后不能从本入口精确续训；重跑从第 1 个 epoch 开始，`--run-root` 必须指向不存在的目录。

单组手动入口（将 `GROUP` 设为 `N0_hand_3d_base`、`N1_hand_wrist_relative` 或 `N2_hand_palm_relative`）：

```bash
GROUP=N2_hand_palm_relative
RUN=artifacts/runs/trajectory/${GROUP}_60ep
python runs/trajectory/train_trajectory.py \
  --config "configs/trajectory/${GROUP}.json" \
  --epochs 60 --run-dir "$RUN"
python runs/trajectory/evaluate_retrieval.py \
  --config "$RUN/params.json" --checkpoint "$RUN/best_model.pth" \
  --split test --output-dir "$RUN/evaluations/cfg2_test"
python runs/trajectory/evaluate_human_robustness.py \
  --config "$RUN/params.json" --checkpoint "$RUN/best_model.pth" \
  --split test --output-dir "$RUN/human_robustness/cfg2_test"
python runs/trajectory/evaluate_retrieval.py \
  --config "$RUN/params.json" --checkpoint "$RUN/best_model.pth" \
  --data-root dataset/RH20T_subset/RH20T_cfg3 --all-as-test \
  --output-dir "$RUN/evaluations/cfg3_all_valid"
python runs/trajectory/evaluate_human_robustness.py \
  --config "$RUN/params.json" --checkpoint "$RUN/best_model.pth" \
  --data-root dataset/RH20T_subset/RH20T_cfg3 --all-as-test \
  --output-dir "$RUN/human_robustness/cfg3_all_valid"
```

扰动评估先用原始人手查询与机器人候选算一次基线，然后只重算变换后的人手查询；每组均复用完全相同的机器人 embedding。变换在 N1/N2 表示转换**之前**施加到原基座点，同时变换 N2 的首帧掌部姿态。因此它检验的是整段人手输入的坐标变化，而不是事后旋转已归一的网络特征。默认分别测 X 平移 0.1 m、yaw 90°、二者组合；输出 `max_abs_embedding_delta`、embedding 余弦、Top‑1 同候选率、Top‑5 重合率和变换前后的 scene/task 检索指标。

`N0_hand_3d_base.json`、`N1_hand_wrist_relative.json` 和 `N2_hand_palm_relative.json` 默认指向已生成的 `trajectory_hand_3d_v1`；若新数据需重新生成清单，改用全新的目录名，并同步修改配置里的两个 `data_manifests` 和 `split.manifest_path`。一键入口也可通过 `--manifest-dir` 指向新清单。cfg3 评估通过同一配置的 `data_manifests.RH20T_cfg3` 选择自己的新清单；cfg2 的训练清单快照只在根目录匹配时复用。`human_representation` 随 checkpoint 配置保存，独立评估拒绝混用不同表示。

## 真实数据烟测结果及限制

使用四个真实 cfg2 scene、两个真实 cfg3 scene 和 1 epoch 跑通 N0/N1/N2 的训练、两域检索和固定候选库评估。该临时小清单只有 3 条 cfg2 test、5 条 cfg3 查询，任务类型单一，**任何检索命中率都无泛化解释价值**。几何检查的 embedding 最大绝对差如下：

| 组别 | 域 | 平移 0.1 m | yaw 90° | 两者组合 |
|---|---|---:|---:|---:|
| N0 | cfg2 / cfg3 | 0.00595 / 0.01060 | 0.19216 / 0.07630 | 0.18171 / 0.06870 |
| N1 | cfg2 / cfg3 | 0 / 0 | 0.04800 / 0.02436 | 0.04800 / 0.02436 |
| N2 | cfg2 / cfg3 | 0 / 0 | 0 / 0 | 0 / 0 |

这验证了当前实现预期的坐标性质，不代表实际重提取视频后的深度、外参或掌部姿态噪声也为零。三组在该极小候选库中的 Top‑1 都没有变化；小候选库的排序不敏感，因此正式实验必须结合 embedding 差、Top‑k 重合及更大的全量候选库报告。完整测试待正式训练后执行。

全仓测试 92 项通过；N2 对完整 cfg2 的 428 个 scene / 1,108 条相机序列、cfg3 的 252 个 scene / 698 条相机序列全部可加载，采样后的有效人手帧数分别为 71,181 和 20,806，所有输入有限且缺测帧坐标为零。CLI 与 `git diff --check` 通过。N3/N4 仍未实现。
