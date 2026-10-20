"""Extract MediaPipe hand landmarks and depth-backed camera-frame hand poses."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MAX_HANDS = 2
LANDMARK_COUNT = 21
CALIBRATION_WIDTH = 1280
CALIBRATION_HEIGHT = 720
DEPTH_SCALE_M_PER_UNIT = 0.001


def decode_rh20t_depth_frame(encoded_bgr: np.ndarray) -> np.ndarray:
    """Decode RH20T's vertically packed 16-bit depth image into raw units."""
    gray = cv2.cvtColor(encoded_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] % 2:
        raise ValueError(f"Packed RH20T depth height must be even, got {gray.shape}")
    half_height = gray.shape[0] // 2
    low = gray[:half_height].astype(np.uint16)
    high = gray[half_height:].astype(np.uint16)
    if low.shape != high.shape:
        raise ValueError("RH20T depth byte planes have different shapes")
    return high * np.uint16(256) + low


def camera_matrix_from_calibration(
    calibration: np.ndarray,
    image_width: int,
    image_height: int,
    calibrated_width: int = CALIBRATION_WIDTH,
    calibrated_height: int = CALIBRATION_HEIGHT,
) -> np.ndarray:
    """Scale an RH20T 3x4 calibration matrix to the decoded video resolution."""
    calibration = np.asarray(calibration, dtype=np.float64)
    if calibration.shape != (3, 4):
        raise ValueError(f"Expected RH20T 3x4 intrinsics, got {calibration.shape}")
    matrix = calibration[:, :3].copy()
    matrix[0, :] *= image_width / calibrated_width
    matrix[1, :] *= image_height / calibrated_height
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    return matrix


def project_pixel_to_camera(u: float, v: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project one pixel into the camera optical frame (x right, y down, z forward)."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    return np.asarray(((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m), dtype=np.float32)


def sample_depth_median(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int = 2,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
) -> tuple[float, int, float]:
    """Take a robust local depth median and return value, inlier count, and MAD."""
    center_x, center_y = int(round(u)), int(round(v))
    height, width = depth_m.shape
    x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return float("nan"), 0, float("nan")
    values = depth_m[y0:y1, x0:x1]
    values = values[np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)]
    if values.size < 3:
        return float("nan"), int(values.size), float("nan")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.02, 3.0 * 1.4826 * mad)
    inliers = values[np.abs(values - median) <= threshold]
    if inliers.size < 3:
        return float("nan"), int(inliers.size), mad
    return float(np.median(inliers)), int(inliers.size), mad


def nearest_timestamp_indices(
    color_timestamps_ms: np.ndarray, depth_timestamps_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each color frame to the nearest depth timestamp without compressing time."""
    color_timestamps_ms = np.asarray(color_timestamps_ms, dtype=np.int64).reshape(-1)
    depth_timestamps_ms = np.asarray(depth_timestamps_ms, dtype=np.int64).reshape(-1)
    if color_timestamps_ms.size == 0 or depth_timestamps_ms.size == 0:
        raise ValueError("Color and depth timestamps must be non-empty")
    if np.any(np.diff(color_timestamps_ms) < 0) or np.any(np.diff(depth_timestamps_ms) < 0):
        raise ValueError("RH20T camera timestamps must be sorted")

    right = np.searchsorted(depth_timestamps_ms, color_timestamps_ms, side="left")
    right = np.clip(right, 0, len(depth_timestamps_ms) - 1)
    left = np.clip(right - 1, 0, len(depth_timestamps_ms) - 1)
    choose_left = np.abs(color_timestamps_ms - depth_timestamps_ms[left]) <= np.abs(
        color_timestamps_ms - depth_timestamps_ms[right]
    )
    indices = np.where(choose_left, left, right).astype(np.int32)
    deltas = depth_timestamps_ms[indices] - color_timestamps_ms
    intervals = np.diff(depth_timestamps_ms)
    typical_interval = float(np.median(intervals)) if intervals.size else 0.0
    tolerance_ms = max(10.0, typical_interval * 0.51)
    valid = np.abs(deltas) <= tolerance_ms
    indices[~valid] = -1
    return indices, deltas.astype(np.int32), valid


def palm_pose_from_landmarks(landmarks_camera_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Construct a right-handed palm frame from wrist and metacarpal landmarks."""
    required = (0, 5, 9, 17)
    pose = np.full((4, 4), np.nan, dtype=np.float32)
    if not all(valid[index] for index in required):
        return pose
    wrist, index_mcp, middle_mcp, pinky_mcp = landmarks_camera_m[list(required)]
    x_axis = index_mcp - pinky_mcp
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-5:
        return pose
    x_axis /= x_norm
    y_axis = middle_mcp - wrist
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-5:
        return pose
    y_axis /= y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-5:
        return pose
    z_axis /= z_norm
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    palm_origin = np.mean(landmarks_camera_m[[0, 5, 9, 13, 17]][valid[[0, 5, 9, 13, 17]]], axis=0)
    pose[:3, :3] = rotation
    pose[:3, 3] = palm_origin
    pose[3, :] = (0.0, 0.0, 0.0, 1.0)
    return pose


def validate_rigid_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_base_from_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        raise ValueError("T_base_from_camera has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-3
    ):
        raise ValueError("T_base_from_camera rotation must be a proper rotation")
    return transform.astype(np.float32)


def track_hands_by_wrist(
    hand_count: np.ndarray,
    handedness_label: np.ndarray,
    landmarks_image_px: np.ndarray,
    image_size_wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign two persistent clip-level IDs using wrist continuity and hand labels."""
    frame_count = len(hand_count)
    track_ids = np.full((frame_count, MAX_HANDS), -1, dtype=np.int8)
    ambiguous = np.zeros((frame_count, MAX_HANDS), dtype=bool)
    last_wrist = np.full((MAX_HANDS, 2), np.nan, dtype=np.float32)
    last_frame = np.full(MAX_HANDS, -1, dtype=np.int32)
    image_diagonal = max(float(np.hypot(*image_size_wh)), 1.0)
    expected_labels = ("Left", "Right")  # Raw MediaPipe labels, not verified anatomical sides.

    for frame_index, count in enumerate(hand_count):
        count = int(count)
        if count == 0:
            continue
        if count > MAX_HANDS:
            raise ValueError(f"More than {MAX_HANDS} hands in frame {frame_index}")
        labels = handedness_label[frame_index, :count]
        wrists = landmarks_image_px[frame_index, :count, 0]
        costs = []
        for assignment in permutations(range(MAX_HANDS), count):
            cost = 0.0
            for detection, track_id in enumerate(assignment):
                label = labels[detection]
                if label in expected_labels and label != expected_labels[track_id]:
                    cost += 0.15
                elif label not in expected_labels:
                    cost += 0.075
                gap = frame_index - last_frame[track_id]
                if last_frame[track_id] >= 0 and gap <= 10 and np.isfinite(wrists[detection]).all():
                    cost += float(np.linalg.norm(wrists[detection] - last_wrist[track_id])) / image_diagonal
                    cost += min(0.02 * gap, 0.1)
                else:
                    cost += 0.2
            costs.append((cost, assignment))
        costs.sort(key=lambda item: (item[0], item[1]))
        best_cost, best_assignment = costs[0]
        uncertain_assignment = len(costs) > 1 and costs[1][0] - best_cost < 0.05
        duplicate_labels = count == 2 and labels[0] == labels[1]
        for detection, track_id in enumerate(best_assignment):
            track_ids[frame_index, detection] = track_id
            ambiguous[frame_index, detection] = (
                uncertain_assignment
                or duplicate_labels
                or labels[detection] != expected_labels[track_id]
            )
            if np.isfinite(wrists[detection]).all():
                last_wrist[track_id] = wrists[detection]
                last_frame[track_id] = frame_index
    return track_ids, ambiguous


def wrist_middle_span_quality(
    landmarks_camera_m: np.ndarray, landmarks_mediapipe_local_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flag large scale disagreements while keeping the measured 3D points unchanged."""
    measured = np.linalg.norm(
        landmarks_camera_m[:, :, 12] - landmarks_camera_m[:, :, 0], axis=-1
    )
    local = np.linalg.norm(
        landmarks_mediapipe_local_m[:, :, 12] - landmarks_mediapipe_local_m[:, :, 0], axis=-1
    )
    comparable = np.isfinite(measured) & np.isfinite(local) & (local > 0.02)
    ratio = np.full(measured.shape, np.nan, dtype=np.float32)
    np.divide(measured, local, out=ratio, where=comparable)
    suspect = comparable & ((ratio < 0.5) | (ratio > 2.0))
    return ratio, suspect


def _load_scene_timestamps(scene_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = scene_dir / f"cam_{camera_id}_human_timestamps.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source timestamps: {path}")
    data = np.load(path, allow_pickle=True).item()
    color = np.asarray(data["color"], dtype=np.int64).reshape(-1)
    depth = np.asarray(data["depth"], dtype=np.int64).reshape(-1)
    return color, depth, path


def _load_scene_intrinsics(scene_dir: Path, camera_id: str, image_width: int, image_height: int):
    resolved_path = scene_dir / "hand_camera_intrinsics_resolved.npy"
    path = resolved_path if resolved_path.is_file() else scene_dir / "hand_camera_intrinsics.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing copied camera intrinsics: {path}")
    data = np.load(path, allow_pickle=True).item()
    if camera_id not in data:
        raise KeyError(f"Camera {camera_id} is absent from {path}")
    matrix = camera_matrix_from_calibration(data[camera_id], image_width, image_height)
    return matrix, path


def _load_optional_extrinsics(scene_dir: Path, camera_id: str):
    path = scene_dir / "T_base_from_camera.json"
    if not path.is_file():
        return None, None, ""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("convention") != "T_base_from_camera":
        raise ValueError(f"{path} must declare convention='T_base_from_camera'")
    camera_data = data.get("cameras", {}).get(camera_id)
    if camera_data is None:
        return None, None, str(path)
    matrix = validate_rigid_transform(camera_data["matrix"])
    return matrix, camera_data.get("source", str(path)), str(path)


def extract_camera_hand_data(scene_dir: Path, camera_id: str) -> dict[str, Any]:
    import mediapipe as mp

    color_path = scene_dir / f"cam_{camera_id}_human.mp4"
    depth_path = scene_dir / f"cam_{camera_id}_human_depth.mp4"
    if not color_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"Missing color/depth video pair for camera {camera_id}")

    color_timestamps, depth_timestamps, timestamps_path = _load_scene_timestamps(scene_dir, camera_id)
    depth_indices, depth_deltas, depth_time_valid = nearest_timestamp_indices(
        color_timestamps, depth_timestamps
    )

    capture = cv2.VideoCapture(str(color_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open color video: {color_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError(f"Invalid color video dimensions: {color_path}")
    intrinsics, intrinsics_path = _load_scene_intrinsics(scene_dir, camera_id, frame_width, frame_height)

    frame_indices: list[int] = []
    image_points: list[np.ndarray] = []
    local_points: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    handedness: list[list[str]] = []
    handedness_score: list[np.ndarray] = []
    hand_count: list[int] = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as model:
        frame_index = 0
        try:
            while True:
                ok, image_bgr = capture.read()
                if not ok:
                    break
                if frame_index >= len(color_timestamps):
                    raise ValueError(f"More color frames than timestamps in {timestamps_path}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                result = model.process(image_rgb)

                frame_image = np.full((MAX_HANDS, LANDMARK_COUNT, 2), np.nan, dtype=np.float32)
                frame_local = np.full((MAX_HANDS, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
                frame_visibility = np.full((MAX_HANDS, LANDMARK_COUNT), np.nan, dtype=np.float32)
                labels = ["" for _ in range(MAX_HANDS)]
                scores = np.full(MAX_HANDS, np.nan, dtype=np.float32)
                image_hands = result.multi_hand_landmarks or []
                world_hands = result.multi_hand_world_landmarks or []
                classifications = result.multi_handedness or []
                indices = list(range(min(len(image_hands), MAX_HANDS)))
                indices.sort(
                    key=lambda i: (
                        classifications[i].classification[0].label
                        if i < len(classifications) and classifications[i].classification
                        else "~"
                    )
                )
                for hand_slot, result_index in enumerate(indices):
                    points = image_hands[result_index].landmark
                    frame_image[hand_slot, :, 0] = [point.x * frame_width for point in points]
                    frame_image[hand_slot, :, 1] = [point.y * frame_height for point in points]
                    frame_visibility[hand_slot] = [
                        float(getattr(point, "visibility", np.nan)) for point in points
                    ]
                    if result_index < len(world_hands):
                        frame_local[hand_slot] = np.asarray(
                            [[point.x, point.y, point.z] for point in world_hands[result_index].landmark],
                            dtype=np.float32,
                        )
                    if result_index < len(classifications) and classifications[result_index].classification:
                        classification = classifications[result_index].classification[0]
                        labels[hand_slot] = str(classification.label)
                        scores[hand_slot] = float(classification.score)
                frame_indices.append(frame_index)
                image_points.append(frame_image)
                local_points.append(frame_local)
                visibility.append(frame_visibility)
                handedness.append(labels)
                handedness_score.append(scores)
                hand_count.append(len(indices))
                frame_index += 1
        finally:
            capture.release()
    if frame_index != len(color_timestamps):
        raise ValueError(f"Color frame/timestamp count mismatch: {frame_index} != {len(color_timestamps)}")

    frame_indices_array = np.asarray(frame_indices, dtype=np.int32)
    image_points_array = np.asarray(image_points, dtype=np.float32)
    local_points_array = np.asarray(local_points, dtype=np.float32)
    frame_count = len(frame_indices_array)
    camera_points = np.full((frame_count, MAX_HANDS, LANDMARK_COUNT, 3), np.nan, dtype=np.float32)
    camera_valid = np.zeros((frame_count, MAX_HANDS, LANDMARK_COUNT), dtype=bool)
    depth_sample_count = np.zeros((frame_count, MAX_HANDS, LANDMARK_COUNT), dtype=np.uint8)
    depth_sample_mad = np.full((frame_count, MAX_HANDS, LANDMARK_COUNT), np.nan, dtype=np.float32)

    targets_by_depth_index: dict[int, list[int]] = {}
    for color_index, depth_index in enumerate(depth_indices):
        if depth_time_valid[color_index]:
            targets_by_depth_index.setdefault(int(depth_index), []).append(color_index)

    depth_capture = cv2.VideoCapture(str(depth_path))
    if not depth_capture.isOpened():
        raise OSError(f"Cannot open depth video: {depth_path}")
    depth_frame_index = 0
    while True:
        ok, encoded_depth = depth_capture.read()
        if not ok:
            break
        target_color_indices = targets_by_depth_index.get(depth_frame_index, [])
        if target_color_indices:
            raw_depth = decode_rh20t_depth_frame(encoded_depth)
            depth_m = raw_depth.astype(np.float32) * DEPTH_SCALE_M_PER_UNIT
            depth_height, depth_width = depth_m.shape
            for color_index in target_color_indices:
                scale_x = depth_width / frame_width
                scale_y = depth_height / frame_height
                for hand_slot in range(MAX_HANDS):
                    for landmark_index in range(LANDMARK_COUNT):
                        u_rgb, v_rgb = image_points_array[color_index, hand_slot, landmark_index]
                        if (
                            not np.isfinite(u_rgb)
                            or not np.isfinite(v_rgb)
                            or u_rgb < 0
                            or v_rgb < 0
                            or u_rgb >= frame_width
                            or v_rgb >= frame_height
                        ):
                            continue
                        u_depth, v_depth = u_rgb * scale_x, v_rgb * scale_y
                        z, count, mad = sample_depth_median(depth_m, u_depth, v_depth)
                        depth_sample_count[color_index, hand_slot, landmark_index] = min(count, 255)
                        depth_sample_mad[color_index, hand_slot, landmark_index] = mad
                        if not np.isfinite(z):
                            continue
                        camera_points[color_index, hand_slot, landmark_index] = project_pixel_to_camera(
                            u_rgb, v_rgb, z, intrinsics
                        )
                        camera_valid[color_index, hand_slot, landmark_index] = True
        depth_frame_index += 1
    depth_capture.release()
    if depth_frame_index != len(depth_timestamps):
        raise ValueError(f"Depth frame/timestamp count mismatch: {depth_frame_index} != {len(depth_timestamps)}")

    camera_poses = np.full((frame_count, MAX_HANDS, 4, 4), np.nan, dtype=np.float32)
    for frame_index in range(frame_count):
        for hand_slot in range(MAX_HANDS):
            camera_poses[frame_index, hand_slot] = palm_pose_from_landmarks(
                camera_points[frame_index, hand_slot], camera_valid[frame_index, hand_slot]
            )

    transform, transform_source, transform_path = _load_optional_extrinsics(scene_dir, camera_id)
    base_points = np.full_like(camera_points, np.nan)
    base_valid = np.zeros_like(camera_valid)
    base_poses = np.full_like(camera_poses, np.nan)
    if transform is not None:
        base_valid[:] = camera_valid
        base_points[camera_valid] = (
            camera_points[camera_valid] @ transform[:3, :3].T + transform[:3, 3]
        )
        pose_mask = np.isfinite(camera_poses).all(axis=(2, 3))
        base_poses[pose_mask] = transform @ camera_poses[pose_mask]

    handedness_array = np.asarray(handedness, dtype="U8")
    hand_count_array = np.asarray(hand_count, dtype=np.uint8)
    track_ids, track_ambiguous = track_hands_by_wrist(
        hand_count_array,
        handedness_array,
        image_points_array,
        np.asarray((frame_width, frame_height), dtype=np.int32),
    )
    span_ratio, span_suspect = wrist_middle_span_quality(camera_points, local_points_array)

    return {
        "frame_index": frame_indices_array,
        "timestamp_ms": color_timestamps.astype(np.int64),
        "depth_frame_index": depth_indices.astype(np.int32),
        "depth_timestamp_ms": np.where(depth_time_valid, depth_timestamps[np.maximum(depth_indices, 0)], -1).astype(np.int64),
        "depth_time_delta_ms": depth_deltas.astype(np.int32),
        "depth_time_valid": depth_time_valid,
        "hand_count": hand_count_array,
        "handedness_label": handedness_array,
        "handedness_score": np.asarray(handedness_score, dtype=np.float32),
        "hand_track_id": track_ids,
        "hand_track_ambiguous": track_ambiguous,
        "wrist_middle_span_ratio": span_ratio,
        "wrist_middle_span_suspect": span_suspect,
        "landmarks_image_px": image_points_array,
        "landmarks_visibility": np.asarray(visibility, dtype=np.float32),
        "landmarks_mediapipe_local_m": local_points_array,
        "landmarks_camera_m": camera_points,
        "landmarks_camera_valid": camera_valid,
        "depth_sample_count": depth_sample_count,
        "depth_sample_mad_m": depth_sample_mad,
        "palm_pose_camera": camera_poses,
        "landmarks_base_m": base_points,
        "landmarks_base_valid": base_valid,
        "palm_pose_base": base_poses,
        "intrinsics_camera_px": intrinsics.astype(np.float32),
        "T_base_from_camera": transform,
        "extrinsics_source": transform_source,
        "extrinsics_file": transform_path,
        "timestamp_file": str(timestamps_path),
        "intrinsics_file": str(intrinsics_path),
        "color_video": str(color_path),
        "depth_video": str(depth_path),
        "image_size_wh": np.asarray((frame_width, frame_height), dtype=np.int32),
        "depth_scale_m_per_unit": DEPTH_SCALE_M_PER_UNIT,
    }


def extract_scene(scene_dir: str | Path, overwrite: bool = False) -> dict:
    scene_dir = Path(scene_dir)
    output_path = scene_dir / "human_hand_3d.npy"
    if output_path.is_file() and not overwrite:
        return {"scene": str(scene_dir), "status": "exists", "output": str(output_path)}

    cameras = sorted(
        path.name.removeprefix("cam_").removesuffix("_human.mp4")
        for path in scene_dir.glob("cam_*_human.mp4")
    )
    if not cameras:
        raise FileNotFoundError(f"No cam_*_human.mp4 videos found in {scene_dir}")
    camera_data = {camera_id: extract_camera_hand_data(scene_dir, camera_id) for camera_id in cameras}
    calibration_metadata_path = scene_dir / "hand_camera_metadata.json"
    if calibration_metadata_path.is_file():
        calibration_metadata = json.loads(calibration_metadata_path.read_text(encoding="utf-8-sig"))
    else:
        calibration_metadata = {}
    result = {
        "schema_version": 2,
        "landmark_count": LANDMARK_COUNT,
        "max_hands": MAX_HANDS,
        "coordinate_frames": {
            "image": "pixel coordinates in the human RGB video",
            "mediapipe_local": "MediaPipe hand-centered metric landmarks; not global translation",
            "camera": "camera optical frame: x right, y down, z forward; meters",
            "base": "robot base frame, only populated when T_base_from_camera.json is provided",
            "transform_convention": "p_base = T_base_from_camera @ [p_camera, 1]",
            "hand_tracking": "hand_track_id 0/1 follows 2D wrist motion and raw MediaPipe Left/Right labels; ambiguous frames are flagged",
        },
        "camera_calibration_metadata": calibration_metadata,
        "cameras": camera_data,
    }
    temporary_path = output_path.with_suffix(".npy.tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, result, allow_pickle=True)
    temporary_path.replace(output_path)
    detected = {camera: int(np.count_nonzero(data["hand_count"])) for camera, data in camera_data.items()}
    valid_3d = {
        camera: int(np.count_nonzero(data["landmarks_camera_valid"]))
        for camera, data in camera_data.items()
    }
    return {
        "scene": str(scene_dir),
        "status": "written",
        "output": str(output_path),
        "camera_count": len(cameras),
        "frame_counts": {camera: int(len(data["frame_index"])) for camera, data in camera_data.items()},
        "detected_frames": detected,
        "valid_3d_landmarks": valid_3d,
    }
