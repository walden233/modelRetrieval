# RH20T human hand 3D extraction

The extractor writes `human_hand_3d.npy` into each scene and leaves the existing `human_pose.npy` unchanged. It preserves every RGB frame, including frames with no detected hand, so the sequence is not shortened.

Use the project environment for MediaPipe and video decoding (`/home/ttt/miniconda3/envs/torch2/bin/python` in this workspace). The commands below use `python` after that environment is activated.

For cfg3, first copy the human camera timestamps and per-camera intrinsic calibration from the uniquely hash-matched raw robot scene:

```bash
python scripts/prepare_rh20t_hand_metadata.py \
  --original-root 'D:\RH20T_cfg3' \
  --subset-root dataset/RH20T_subset/RH20T_cfg3
```

The preparation step also resolves any absent camera serials from another cfg3 calibration. It only does this when every available intrinsic matrix for that serial is exactly identical, and records the selected source in `hand_camera_metadata.json`. If timestamps and intrinsics were already copied, the local resolution step can be run alone:

```bash
python scripts/prepare_rh20t_hand_metadata.py --resolve-only \
  --subset-root dataset/RH20T_subset/RH20T_cfg3
```

Then run MediaPipe and back-project each 2D landmark through the paired depth video:

```bash
python scripts/extract_human_hand_3d.py \
  --data-root dataset/RH20T_subset/RH20T_cfg3
```

For outputs created before hand tracking was added, run the fast metadata pass once after extraction; it does not rerun MediaPipe:

```bash
python scripts/add_hand_tracks_to_3d.py \
  --data-root dataset/RH20T_subset/RH20T_cfg3
```

Validate the complete set of scene files:

```bash
python scripts/validate_human_hand_3d.py \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --report artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_3d_validation.json
```

Use `--scene task_0057/scene_1` for a smoke run, `--limit-scenes 1` for the first sorted scene, and `--overwrite` to regenerate existing outputs. The extractor report is written to `artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_3d_extract.json` by default.

Each camera record stores frame index and source timestamp in milliseconds, MediaPipe's local hand-centered landmarks, image-pixel landmarks, handedness, depth-backed camera-frame XYZ in meters, depth validity/quality values, and a palm pose estimated from wrist/metacarpal points. Per-frame detections remain in MediaPipe order; `hand_track_id` identifies a hand across frames, and `hand_track_ambiguous` flags uncertain assignments. These IDs use wrist motion and MediaPipe's raw Left/Right labels. The labels assume a mirrored camera image and should not be interpreted as verified anatomical sides. A 5×5 neighborhood median rejects invalid depth pixels and isolated outliers. RGB frames are matched to depth frames by source timestamps; missing or badly mismatched pairs remain invalid rather than borrowing a distant depth frame.

Each hand record also has `wrist_middle_span_ratio` and `wrist_middle_span_suspect`. The latter flags a depth-reconstructed wrist-to-middle-fingertip span less than half or more than twice the MediaPipe local estimate. This is a diagnostic for noisy RGB/depth landmark alignment, not a proof that the remaining depth joints are exact. `landmarks_camera_valid` means a usable local depth sample exists; training code can also inspect `depth_sample_count`, `depth_sample_mad_m`, and `hand_track_ambiguous`.

`hand_camera_intrinsics.npy` is sourced from the paired robot scene's calibration. Where that calibration omits a camera serial, `hand_camera_intrinsics_resolved.npy` adds the identical serial-specific intrinsic matrix from a different calibration. The extractor scales intrinsics from the RH20T 1280×720 calibration resolution to the stored 640×360 videos. Camera-frame XYZ is available without extrinsics.

For cfg3, the human metadata has `calib_quality=-1`. The following step uses the verified dataset-specific assumption that robot and human videos with the same fixed camera serial share camera extrinsics. It matches each subset scene to its raw robot scene through the existing exact TCP hash mapping, then applies the official RH20T cfg3 calibration chain and its inverse. The result uses the same aligned robot-base convention as `tcp_base.npy`. It updates existing hand files without rerunning MediaPipe:

```bash
python scripts/add_hand_base_coordinates.py \
  --raw-root /mnt/d/RH20T_cfg3 \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --mapping artifacts/audits/trajectory_20260923/rh20t_cfg3_hash_mapping.json
python scripts/validate_human_hand_3d.py \
  --data-root dataset/RH20T_subset/RH20T_cfg3 \
  --report artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_3d_validation.json
```

Each scene gains `T_base_from_camera.json`:

```json
{
  "convention": "T_base_from_camera",
  "cameras": {
    "038522062288": {
      "matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "source": "raw cfg3 extrinsics.npy + tcp.npy + official cfg3 config"
    }
  }
}
```

The matrix maps optical camera coordinates to aligned robot-base coordinates in meters. The scene file records the robot calibration ID, calibration quality, source file and whether a different calibration was needed for a missing camera serial. Fallbacks use the complete alternative calibration chain; raw marker-frame extrinsics from different calibrations must not be mixed. Two early cfg3 calibration files have TCP height 0.450036 m, while 26 of 28 numeric cfg3 calibrations have 0.200036 m. The three subset scenes using one of those early calibrations also begin near 0.200036 m in `tcp_base.npy`. The script corrects that 0.25 m calibration-height outlier to the calibration consensus and records the original and used heights in `T_base_from_camera.json`. The transformed points, validity mask, palm poses and matrix are stored in `human_hand_3d.npy`. Future `extract_human_hand_3d.py --overwrite` runs read this transform file and regenerate the base fields.

The base conversion preserves existing camera-frame measurement errors and the fixed-camera sharing assumption. `landmarks_base_valid` indicates that depth was available, not that the corresponding joint passed a cross-view accuracy threshold. A nearest-timestamp/nearest-wrist diagnostic across camera pairs has a median difference of about 0.17 m, but it can compare different visible hands and is not a calibrated point correspondence test. One overlapping camera pair (`038522062288`/`104422070011`) has a median of about 0.03 m. Training code currently reads `human_pose.npy`; it will not use the new base fields until the trajectory dataset loader is updated.
