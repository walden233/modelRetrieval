# RH20T 划分与数据清单

## 当前子集存储与夹爪来源

cfg2、cfg3 的夹爪数据属于数据集层，供不同 batch 共用：

```text
dataset/RH20T_subset/
  RH20T_cfg2/                    # RH20T_cfg3 采用相同结构
    geometry.json               # 仅 profiles：本 cfg 的夹爪参数及依据，无 scene 索引
    task_*/scene_*/
      tcp_base.npy              # 原有 TCP 位姿与绝对毫秒时间戳
      human_hand_3d.npy         # 原有 21 点人手数据
      T_base_from_camera.json  # 原有坐标变换与来源记录
      cam_*_human.mp4           # 原有人类视频
      cam_*_robot.mp4           # 原有机器人视频
      gripper.npy               # 新增：对应原始 scene 的完整夹爪反馈文件
      geometry.npy              # 新增：由该 scene 的 TCP 与实测反馈重新生成
```

目录树仅列相关文件，其余原有深度视频、元数据等保留。
`gripper.npy` 原样复制自原始数据的 `transformed/gripper.npy`，不改变相机键或内容。
`geometry.npy` 为字典 `{camera_serial: float32[N, 8]}`；每行是
`[x, y, z, qw, qx, qy, qz, opening_normalized]`，帧数和顺序与该相机的 TCP 一致，
空 TCP 相机不生成条目。开合来自 `gripper_info[0]` 实测反馈，并按 TCP 绝对时间戳对齐，
归一化为 `反馈值 × width_scale_to_m / max_width_m`；不使用目标开合指令。

### 夹爪参数及依据

此次参数沿用 `artifacts/manifests/retrain/s43/cfg2/geometry.json` 的 `profiles`，
按实际匹配到的原始 cfg 分别写入两个数据集目录，只复制参数，不复制旧缓存或 scene 记录。
来源说明也保存在各自参数文件的 `width_source`、`reference_source` 字段中。

| cfg | 夹爪 | `max_width_m` | `width_scale_to_m` | 已记录的来源依据 |
|---|---|---:|---:|---|
| cfg2 | Dahuan AG-95 | 0.095 m | 0.001（毫米→米） | [RH20T 官方夹爪预处理](https://github.com/rh20t/rh20t_api/blob/main/rh20t_api/configurations.py) 中的 95 mm；既有反馈检查覆盖 0–95 mm |
| cfg3 | WSG-50 | 0.110 m | 0.001（毫米→米） | [Weiss Robotics WSG 产品资料](https://weiss-robotics.com/servo-electric/wsg-series/product/wsg-serie/) 中 WSG 50-110 的 110 mm 全行程；既有反馈检查达到 110 mm |

两者均采用 `reference=aligned_tcp_proxy`、单位矩阵 `tcp_to_grasp`，保留上游已对齐
TCP 的原点与轴。参数中记录的实现来源为 [RH20T 官方 online.py](https://github.com/rh20t/rh20t_api/blob/main/rh20t_api/online.py)。
这些参数没有提供实测 TCP→指尖接触中心标定；虚拟两指与运动/手形表示见
[统一抓取几何](trajectory/S43.md#3-统一抓取几何)。

### 补齐与重新生成入口

在仓库根目录、`torch2` 环境下运行：

```bash
python runs/trajectory/prepare_dataset_geometry.py --cfg cfg2 cfg3 --check
python runs/trajectory/prepare_dataset_geometry.py --cfg cfg2 cfg3
```

此独立入口读取 `/mnt/d/RH20T_cfg2`、`/mnt/d/RH20T_cfg3`；D 盘只在补齐/核对来源时需要。
脚本建立原始 `transformed/tcp_base.npy` 的 SHA-256 索引，用子集 TCP 文件的相同哈希
唯一匹配原始 scene，不按子集 task/scene 编号猜测，也不以来源元数据代替哈希匹配。
零匹配或多个匹配均停止；匹配后复制原始夹爪反馈，再从子集 TCP 重新计算几何缓存。
已有输出相同则复用，不同则停止；文件使用独占创建，原有文件不覆盖，
不另存映射、几何索引或审计 JSON。

本次补齐 cfg2 的 434 个 scene、cfg3 的 256 个 scene，共新增 690 份 `gripper.npy`、
690 份 `geometry.npy` 和 2 份参数 JSON。缓存覆盖 5,147 条相机轨迹、2,945,912 帧。
复制内容与原始反馈哈希一致；全部 19,912 个原有子集文件的 SHA-256、大小、
修改时间和 ctime 均核验未变。数据集 scene 总数与训练有效池不同：
新 batch 的人手/视频共同有效池为 cfg2 428、cfg3 252 个 scene。

训练、评估及新 batch 准备直接读取 dataset 内的几何文件，不需要 D 盘或 s43 的旧几何缓存。
batch 仅保存数据清单、元数据和模态划分，数据清单不新增几何哈希字段。
新 batch 流水线及独立入口见 [s43 模板的新存储入口](trajectory/S43.md#新数据存储结构的入口)，
批次划分规则见 [RH20T retrain](RETRAIN.md#协议与命名)。

## 旧共享划分（历史实验复现）

以下记录原有 43 个固定测试 scene 的协议，不用于当前 retrain batch。

### 目录职责

```text
artifacts/manifests/
  rh20t_shared_v1/
    cfg2_fixed_test_scenes.json
  trajectory_hand_3d_v1/
    cfg2_valid.json
    cfg3_valid.json
    cfg2_split.json          # 旧实验的完整划分，保留用于复现
    cfg3_split.json
```

`rh20t_shared_v1/cfg2_fixed_test_scenes.json` 是跨模态协议文件。它只保存 43 个固定测试 scene ID，不保存 camera sample ID，也不保存 train/val。轨迹模态直接使用这些 scene；视频模态将每个 scene 展开为该 scene 的所有 camera 样本。因此两种模态的测试 scene 完全一致。

每次训练从各自模态实际可用、且不属于固定 test 的 scene 中，按训练 seed 动态抽取验证集。验证集占剩余数据的 `1/9`，其余用于训练。每个 run 目录中的 `split_manifest.json` 固化该次训练最终使用的 train/val/test；评估读取 run 内文件，不再次随机划分。

`trajectory_hand_3d_v1/cfg2_valid.json` 不是数据划分。它是新 3D 手轨迹加载器的有效数据库存，包含：

- 数据根目录与 schema/坐标语义；
- 每个可用 scene 和 camera；
- 选中的手 track、有效帧数、覆盖率和质量统计；
- `human_hand_3d.npy` 与 `tcp_base.npy` 的 SHA-256；
- 被排除的 scene/camera 及原因。

加载器先用该库存确定哪些输入有效，并在每次读取时校验哈希，防止源 NPY 变化后静默改变实验。之后才应用共享固定 test 和动态 train/val。视频模态不读取这个文件，因为视频有自己的样本扫描与有效性条件。

当前 scene 数量不同是预期行为：旧轨迹数据可用 429 个 scene，新 3D 手轨迹可用 428 个，视频可用 434 个；它们共享同一组 43 个固定测试 scene，但剩余训练池大小分别由各模态的数据有效性决定。

### 配置格式

```json
{
  "seed": 42,
  "split": {
    "fixed_test_manifest_path": "artifacts/manifests/rh20t_shared_v1/cfg2_fixed_test_scenes.json",
    "fixed_test_fraction": 0.1,
    "val_ratio_of_remaining": 0.1111111111111111,
    "unit": "scene"
  }
}
```

相同 seed 会复现相同的动态 train/val；更换 seed 会改变 train/val，但不会改变固定 test。task-held-out 实验不使用该配置，因为固定的 scene test 不能同时保证测试任务未出现在训练集。

通用训练入口可以直接覆盖 seed：

```bash
python runs/trajectory/train_trajectory.py \
  --config configs/trajectory/T2_trajectory_augment_scene.json --seed 42

python runs/video/train_video.py \
  --config configs/video/vjepa_rh20t_dual_encoder.json --seed 42
```
