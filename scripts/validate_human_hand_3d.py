#!/usr/bin/env python3
"""Validate the per-scene human_hand_3d.npy outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate_file(path: Path) -> list[str]:
    errors = []
    try:
        data = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        return [f"{path}: cannot load file: {exc}"]
    if data.get("schema_version") != 2 or not isinstance(data.get("cameras"), dict):
        return [f"{path}: unsupported or malformed schema"]
    expected_cameras = {
        video.name.removeprefix("cam_").removesuffix("_human.mp4")
        for video in path.parent.glob("cam_*_human.mp4")
    }
    if set(data["cameras"]) != expected_cameras:
        errors.append(f"{path}: output cameras do not match the scene's human videos")
    transform_path = path.parent / "T_base_from_camera.json"
    transform_data = json.loads(transform_path.read_text(encoding="utf-8-sig")) if transform_path.is_file() else None
    for camera_id, camera in data["cameras"].items():
        frames = np.asarray(camera["frame_index"])
        timestamps = np.asarray(camera["timestamp_ms"])
        landmarks = np.asarray(camera["landmarks_camera_m"])
        valid = np.asarray(camera["landmarks_camera_valid"], dtype=bool)
        if frames.ndim != 1 or not np.array_equal(frames, np.arange(len(frames))):
            errors.append(f"{path} [{camera_id}]: frame_index is not the full uncompressed frame sequence")
        if timestamps.shape != frames.shape or np.any(np.diff(timestamps) < 0):
            errors.append(f"{path} [{camera_id}]: invalid timestamps")
        timestamp_file = path.parent / f"cam_{camera_id}_human_timestamps.npy"
        if timestamp_file.is_file():
            source_timestamps = np.load(timestamp_file, allow_pickle=True).item()
            if not np.array_equal(timestamps, np.asarray(source_timestamps["color"], dtype=np.int64)):
                errors.append(f"{path} [{camera_id}]: timestamps differ from the raw human camera")
        if landmarks.shape != (len(frames), data["max_hands"], data["landmark_count"], 3):
            errors.append(f"{path} [{camera_id}]: camera landmark shape mismatch")
        if valid.shape != landmarks.shape[:-1]:
            errors.append(f"{path} [{camera_id}]: camera validity mask shape mismatch")
        if valid.any() and not np.isfinite(landmarks[valid]).all():
            errors.append(f"{path} [{camera_id}]: valid camera landmarks contain non-finite values")
        if not np.array_equal(np.isfinite(landmarks).all(axis=-1), valid):
            errors.append(f"{path} [{camera_id}]: camera landmark values and validity mask disagree")
        depth_time_valid = np.asarray(camera["depth_time_valid"], dtype=bool)
        depth_indices = np.asarray(camera["depth_frame_index"])
        if depth_time_valid.shape != frames.shape or depth_indices.shape != frames.shape:
            errors.append(f"{path} [{camera_id}]: invalid depth timestamp alignment shapes")
        elif np.any(depth_indices[~depth_time_valid] != -1):
            errors.append(f"{path} [{camera_id}]: invalid depth pairs have a frame index")
        elif timestamp_file.is_file() and depth_time_valid.any():
            source_depth_times = np.asarray(source_timestamps["depth"], dtype=np.int64)
            selected_indices = depth_indices[depth_time_valid]
            if np.any(selected_indices < 0) or np.any(selected_indices >= len(source_depth_times)):
                errors.append(f"{path} [{camera_id}]: depth frame index is out of range")
            elif not np.array_equal(
                np.asarray(camera["depth_timestamp_ms"])[depth_time_valid],
                source_depth_times[selected_indices],
            ):
                errors.append(f"{path} [{camera_id}]: depth timestamps differ from the raw camera")
        track_ids = np.asarray(camera["hand_track_id"])
        track_ambiguous = np.asarray(camera["hand_track_ambiguous"])
        if track_ids.shape != (len(frames), data["max_hands"]) or track_ambiguous.shape != track_ids.shape:
            errors.append(f"{path} [{camera_id}]: hand track shape mismatch")
        else:
            count = np.asarray(camera["hand_count"])
            if not np.array_equal((track_ids >= 0).sum(axis=1), count):
                errors.append(f"{path} [{camera_id}]: hand track IDs do not match detections")
            if np.any((track_ids < -1) | (track_ids >= data["max_hands"])):
                errors.append(f"{path} [{camera_id}]: invalid hand track ID")
            if np.any((track_ids[:, 0] >= 0) & (track_ids[:, 0] == track_ids[:, 1])):
                errors.append(f"{path} [{camera_id}]: two detections share one hand track ID")
        span_ratio = np.asarray(camera["wrist_middle_span_ratio"])
        span_suspect = np.asarray(camera["wrist_middle_span_suspect"], dtype=bool)
        if span_ratio.shape != track_ids.shape or span_suspect.shape != track_ids.shape:
            errors.append(f"{path} [{camera_id}]: wrist-middle span quality shape mismatch")
        elif np.any(span_suspect & ~np.isfinite(span_ratio)):
            errors.append(f"{path} [{camera_id}]: suspect span has no measured ratio")
        if camera["T_base_from_camera"] is None:
            if np.any(camera["landmarks_base_valid"]):
                errors.append(f"{path} [{camera_id}]: base-frame points marked valid without an extrinsic")
        else:
            transform = np.asarray(camera["T_base_from_camera"], dtype=np.float64)
            base_points = np.asarray(camera["landmarks_base_m"])
            base_valid = np.asarray(camera["landmarks_base_valid"], dtype=bool)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                errors.append(f"{path} [{camera_id}]: invalid base transform")
            elif not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5) or not np.allclose(
                transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3
            ) or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-3):
                errors.append(f"{path} [{camera_id}]: base transform is not rigid")
            elif transform_data is None or transform_data.get("convention") != "T_base_from_camera" or camera_id not in transform_data.get("cameras", {}):
                errors.append(f"{path} [{camera_id}]: missing source transform JSON")
            elif not np.allclose(transform, transform_data["cameras"][camera_id]["matrix"], atol=1e-6):
                errors.append(f"{path} [{camera_id}]: saved transform differs from source JSON")
            elif base_points.shape != landmarks.shape:
                errors.append(f"{path} [{camera_id}]: base landmark shape mismatch")
            elif not np.array_equal(base_valid, valid):
                errors.append(f"{path} [{camera_id}]: base validity differs from camera validity")
            elif not np.array_equal(np.isfinite(base_points).all(axis=-1), valid):
                errors.append(f"{path} [{camera_id}]: base landmark values and mask disagree")
            elif not np.allclose(base_points[valid], landmarks[valid] @ transform[:3, :3].T + transform[:3, 3], atol=2e-5):
                errors.append(f"{path} [{camera_id}]: base landmarks do not match the saved transform")
            camera_poses = np.asarray(camera["palm_pose_camera"])
            base_poses = np.asarray(camera["palm_pose_base"])
            pose_valid = np.isfinite(camera_poses).all(axis=(2, 3))
            if base_poses.shape != camera_poses.shape or not np.array_equal(
                np.isfinite(base_poses).all(axis=(2, 3)), pose_valid
            ):
                errors.append(f"{path} [{camera_id}]: base palm pose validity mismatch")
            elif transform.shape == (4, 4) and not np.allclose(
                base_poses[pose_valid], transform @ camera_poses[pose_valid], atol=2e-5
            ):
                errors.append(f"{path} [{camera_id}]: base palm poses do not match the saved transform")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/RH20T_subset/RH20T_cfg3")
    parser.add_argument("--report", help="Optional JSON report output.")
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    scene_dirs = sorted(path for path in data_root.glob("task_*/scene_*") if path.is_dir())
    files = sorted(data_root.glob("task_*/scene_*/human_hand_3d.npy"))
    if not files:
        parser.error(f"No human_hand_3d.npy files found under {data_root}")
    results = [{"file": str(path), "errors": validate_file(path)} for path in files]
    failed = [row for row in results if row["errors"]]
    missing = [str(scene_dir / "human_hand_3d.npy") for scene_dir in scene_dirs
               if not (scene_dir / "human_hand_3d.npy").is_file()]
    quality = {
        "camera_count": 0, "rgb_frames": 0, "detected_hands": 0,
        "depth_backprojected_landmarks": 0, "detected_landmark_slots": 0,
        "palm_poses": 0, "matched_depth_frames": 0,
        "geometry_checked_hands": 0, "geometry_suspect_hands": 0,
        "ambiguous_hand_tracks": 0,
        "base_transformed_cameras": 0, "base_landmarks": 0, "base_palm_poses": 0,
    }
    for row in results:
        if row["errors"]:
            continue
        data = np.load(row["file"], allow_pickle=True).item()
        for camera in data["cameras"].values():
            quality["camera_count"] += 1
            quality["rgb_frames"] += len(camera["frame_index"])
            detected = int(np.asarray(camera["hand_count"]).sum())
            quality["detected_hands"] += detected
            quality["detected_landmark_slots"] += detected * data["landmark_count"]
            quality["depth_backprojected_landmarks"] += int(np.asarray(camera["landmarks_camera_valid"]).sum())
            quality["palm_poses"] += int(np.isfinite(camera["palm_pose_camera"]).all(axis=(2, 3)).sum())
            quality["matched_depth_frames"] += int(np.asarray(camera["depth_time_valid"]).sum())
            quality["ambiguous_hand_tracks"] += int(np.asarray(camera["hand_track_ambiguous"]).sum())
            quality["geometry_checked_hands"] += int(np.isfinite(camera["wrist_middle_span_ratio"]).sum())
            quality["geometry_suspect_hands"] += int(np.asarray(camera["wrist_middle_span_suspect"]).sum())
            if camera["T_base_from_camera"] is not None:
                quality["base_transformed_cameras"] += 1
                quality["base_landmarks"] += int(np.asarray(camera["landmarks_base_valid"]).sum())
                quality["base_palm_poses"] += int(np.isfinite(camera["palm_pose_base"]).all(axis=(2, 3)).sum())
    checked = quality["geometry_checked_hands"]
    quality["geometry_suspect_fraction"] = quality["geometry_suspect_hands"] / checked if checked else None
    summary = {"data_root": str(data_root), "scene_count": len(scene_dirs), "file_count": len(files),
               "missing_count": len(missing), "failed_count": len(failed), "quality": quality}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({**summary, "missing": missing, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report: {report_path}")
    for row in failed[:20]:
        for error in row["errors"]:
            print(error)
    for missing_path in missing[:20]:
        print(f"Missing output: {missing_path}")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
