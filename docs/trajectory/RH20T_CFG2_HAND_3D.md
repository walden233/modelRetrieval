# RH20T cfg2 human hand 3D extraction

This uses the unique TCP SHA-256 scene mapping in `artifacts/audits/trajectory_20260923/rh20t_cfg2_hash_mapping.json` to find each raw robot scene and its `_human` partner in `/mnt/d/RH20T_cfg2`. The subset has 434 scenes and 1,302 human camera videos. The raw folder supplies paired depth videos, RGB/depth timestamps, and robot camera calibration.

The completed run copied and SHA-256 verified 2,604 depth videos, prepared metadata for all 434 scenes, and wrote all 434 `human_hand_3d.npy` files. Validation found 0 missing or malformed scene outputs: 224,272 RGB frames, 150,300 detected hands, 3,138,396 depth-backed/base-frame landmarks, and 146,396 palm poses. All 1,302 camera transforms came from the paired robot calibration; no fallback extrinsics or calibration TCP corrections were required. Detailed counts are in `artifacts/audits/trajectory_20260923/rh20t_cfg2_hand_3d_validation.json` and `rh20t_cfg2_hand_base.json`.

Use the project Python environment (`/home/ttt/miniconda3/envs/torch2/bin/python` here). To reproduce the complete dataset:

```bash
python scripts/copy_rh20t_depth_videos.py \
  --original-root /mnt/d/RH20T_cfg2 \
  --subset-root dataset/RH20T_subset/RH20T_cfg2 \
  --mapping artifacts/audits/trajectory_20260923/rh20t_cfg2_hash_mapping.json \
  --output artifacts/audits/trajectory_20260923/rh20t_cfg2_depth_copy_manifest.json
python scripts/prepare_rh20t_hand_metadata.py \
  --original-root /mnt/d/RH20T_cfg2 \
  --subset-root dataset/RH20T_subset/RH20T_cfg2 \
  --mapping artifacts/audits/trajectory_20260923/rh20t_cfg2_hash_mapping.json \
  --output artifacts/audits/trajectory_20260923/rh20t_cfg2_hand_metadata_manifest.json \
  --resolve-report artifacts/audits/trajectory_20260923/rh20t_cfg2_hand_intrinsics_resolution.json
python scripts/extract_human_hand_3d.py \
  --data-root dataset/RH20T_subset/RH20T_cfg2 --workers 3 \
  --report artifacts/audits/trajectory_20260923/rh20t_cfg2_hand_3d_extract.json
python scripts/add_hand_base_coordinates.py \
  --cfg 2 --raw-root /mnt/d/RH20T_cfg2 \
  --data-root dataset/RH20T_subset/RH20T_cfg2 \
  --mapping artifacts/audits/trajectory_20260923/rh20t_cfg2_hash_mapping.json \
  --report artifacts/audits/trajectory_20260923/rh20t_cfg2_hand_base.json
python scripts/validate_human_hand_3d.py \
  --data-root dataset/RH20T_subset/RH20T_cfg2 \
  --report artifacts/audits/trajectory_20260923/rh20t_cfg2_hand_3d_validation.json
```

`human_hand_3d.npy` stores each RGB frame's 2D hand landmarks, MediaPipe hand-local landmarks, depth-backed metric camera-frame and aligned robot-base landmarks, palm poses, validity masks, timestamp, and hand-track diagnostics. `T_base_from_camera.json` records the per-camera matrix and raw calibration provenance. No-hand frames remain in the sequence. `human_pose.npy` is unchanged.

The cfg2 transform uses the [official cfg2 configuration](https://github.com/rh20t/rh20t_api/blob/main/configs/configs.json) and [RH20T calibration chain](https://github.com/rh20t/rh20t_api/blob/main/rh20t_api/scene.py). Human metadata often names a different calibration than the paired robot scene. Following the checked same-camera-serial assumption, the script selects the mapped robot scene's calibration so its base frame corresponds to `tcp_base.npy`. The transform and measurement quality remain separate: a valid depth sample does not guarantee millimeter-level joint accuracy. Current trajectory training code still reads `human_pose.npy` until its loader is updated.

The MediaPipe output includes every detected hand, including hands at a nearby keyboard or outside the manipulation area. The two saved hand tracks are camera-local detection tracks, not verified task-hand identities. Use image locations, base-frame workspace bounds, and cross-view evidence before choosing a hand for training.
