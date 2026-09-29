# RH20T 划分与数据清单

## 目录职责

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

## 配置格式

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
