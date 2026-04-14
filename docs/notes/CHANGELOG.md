# 改动记录

## 2026-04-13 轨迹公平 Ablation 与 Intra Loss 调整

- 新增公平 ablation 配置：
  - `configs/trajectory/baseline_fair.json`
  - `configs/trajectory/augment_fair.json`
  - `configs/trajectory/two_stage_fair.json`
- 公平 ablation 统一了关键对比条件：
  - 全部使用 `RH20T_cfg2`
  - 全部使用 6 个手部关键点，`human_input_dim=18`
  - 全部使用 `tcp_sample_factor=4`
  - 全部训练时按 `task` 定义跨模态正样本，评估时按 `scene` 计算检索指标
  - `two_stage_fair` 将总训练预算控制为 `10` 个 pretrain epoch + `50` 个 finetune epoch，便于与 `60` epoch 的 baseline / augment 对比
- 将轨迹模态内对比从“单正样本 SimCLR”改为“多正样本 SupCon 风格”：
  - 同一 `scene` 的不同 camera 视图在 human 分支内视为强正样本
  - 同一 `scene` 的 robot 轨迹副本在 robot 分支内视为强正样本
  - 同一 `task` 但不同 `scene` 的样本支持以较小权重作为弱正样本，默认由 `intra_task_positive_weight` 控制
- 将增强策略从“任意 3D 随机旋转”改为更保守的设置：
  - 小幅高斯噪声 `augmentation_noise_std`
  - 仅小角度 z 轴旋转 `augmentation_max_rotation_degrees`
  - human 轨迹仍绕 root keypoint 旋转
  - robot TCP 轨迹改为围绕首帧位置做小角度 z 轴旋转，并同步旋转末端姿态四元数
- 训练入口 `tools/train_trajectory.py` 已接入：
  - `intra_task_positive_weight`
  - `augmentation_noise_std`
  - `augmentation_max_rotation_degrees`
- 新增测试：
  - `tests/test_trajectory_losses.py`
  - 覆盖多正样本 `intra` loss 与新增强的基本行为

### 运行建议

- 公平 baseline：
  - `python tools/train_trajectory.py --config configs/trajectory/baseline_fair.json`
- 公平 augment：
  - `python tools/train_trajectory.py --config configs/trajectory/augment_fair.json`
- 公平 two-stage：
  - `python tools/train_trajectory.py --config configs/trajectory/two_stage_fair.json`
