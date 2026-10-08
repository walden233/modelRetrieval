"""Trajectory samples from depth-backed, robot-base-frame human hand landmarks."""

import json
from copy import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .scanner import SceneRecord
from .human_representation import HUMAN_REPRESENTATIONS, hand_xyz_features, validate_global_transform
from .trajectory_dataset import _sample_indices
from .trajectory_manifest import load_trajectory_file


MANIFEST_SCHEMA_VERSION = 2
HUMAN_SOURCE = "hand_3d_base"
KEYPOINT_INDICES = (0, 4, 8, 12, 16, 20)


def _validated_camera(camera: dict, robot_frames: list) -> tuple[np.ndarray, np.ndarray, dict]:
    frame_ids = np.asarray(camera["frame_index"])
    timestamps = np.asarray(camera["timestamp_ms"])
    points = np.asarray(camera["landmarks_base_m"])
    valid = np.asarray(camera["landmarks_base_valid"])
    tracks = np.asarray(camera["hand_track_id"])
    ambiguous = np.asarray(camera["hand_track_ambiguous"])
    suspect = np.asarray(camera["wrist_middle_span_suspect"])
    n = len(frame_ids)
    if n < 2 or points.shape != (n, 2, 21, 3) or valid.shape != (n, 2, 21):
        raise ValueError("invalid hand landmark dimensions")
    if any(x.shape != (n, 2) for x in (tracks, ambiguous, suspect)) or timestamps.shape != (n,):
        raise ValueError("invalid hand track or time dimensions")
    if (not np.isfinite(frame_ids).all() or not np.isfinite(timestamps).all()
            or (np.diff(frame_ids) <= 0).any() or (np.diff(timestamps) <= 0).any()):
        raise ValueError("human frame indices and millisecond timestamps must increase")
    transform = np.asarray(camera["T_base_from_camera"], dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-3)):
        raise ValueError("invalid camera-to-base rigid transform")
    if not robot_frames:
        raise ValueError("empty robot sequence")
    tcp = np.asarray([frame["tcp"] for frame in robot_frames])
    robot_timestamps = np.asarray([frame["timestamp"] for frame in robot_frames], dtype=np.float64)
    if (tcp.shape != (len(robot_frames), 7) or not np.isfinite(tcp).all()
            or (np.linalg.norm(tcp[:, 3:], axis=-1) < 1e-6).any()
            or not np.isfinite(robot_timestamps).all()
            or (np.diff(robot_timestamps) <= 0).any()):
        raise ValueError("invalid robot TCP or millisecond timestamps")

    complete = valid.all(axis=-1) & np.isfinite(points).all(axis=(-1, -2))
    clean = complete & ~ambiguous & ~suspect
    counts = [int(((tracks == track_id) & clean).sum()) for track_id in (0, 1)]
    track_id = int(np.argmax(counts))  # Stable tie-break: track 0.
    observed = (tracks == track_id) & clean
    frame_observed = observed.any(axis=1)
    if counts[track_id] < 2:
        raise ValueError("fewer than two clean base-frame hand observations")
    quality = {
        "selected_track_id": track_id,
        "human_frames": n,
        "observed_frames": counts[track_id],
        "coverage": counts[track_id] / n,
        "ambiguous_detections": int(ambiguous.sum()),
        "geometry_suspect_detections": int(suspect.sum()),
        "multi_hand_frames": int((np.asarray(camera["hand_count"]) > 1).sum()),
        "robot_frames": len(robot_frames),
        "extrinsics_source": str(camera.get("extrinsics_source", "unknown")),
    }
    return observed, frame_observed, quality


def build_hand_3d_trajectory_manifest(root_dir: str) -> dict:
    root = Path(root_dir)
    entries, exclusions = [], []
    for path in sorted(root.glob("task_*/scene_*")):
        if not path.is_dir():
            continue
        scene_id = f"{path.parent.name}/{path.name}"
        try:
            human, human_hash = load_trajectory_file(path / "human_hand_3d.npy")
            robot, robot_hash = load_trajectory_file(path / "tcp_base.npy")
            if human.get("schema_version") != 2 or human.get("landmark_count") != 21 or human.get("max_hands") != 2:
                raise ValueError("unsupported human_hand_3d.npy schema")
            if not isinstance(human.get("cameras"), dict) or not isinstance(robot, dict):
                raise ValueError("invalid scene data")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            exclusions.append({"scene_id": scene_id, "reason": str(exc)})
            continue
        cameras = {}
        for camera_id in sorted(human["cameras"].keys() & robot.keys()):
            try:
                _, _, cameras[camera_id] = _validated_camera(human["cameras"][camera_id], robot[camera_id])
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                exclusions.append({"scene_id": scene_id, "camera_id": camera_id, "reason": str(exc)})
        if not cameras:
            exclusions.append({"scene_id": scene_id, "reason": "no valid paired camera"})
            continue
        entries.append({"scene_id": scene_id, "task_id": path.parent.name,
                        "human_sha256": human_hash, "robot_sha256": robot_hash,
                        "cameras": cameras})
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "human_source": HUMAN_SOURCE,
            "data_root": str(root.resolve()), "human_time_unit": "seconds_from_rgb_timestamp_ms",
            "robot_time_unit": "seconds_from_timestamp_ms",
            "task_identity": "local_subset_id_not_cross_cfg_semantic_id",
            "hand_selection": "max_clean_21_point_frames_per_camera_track",
            "scenes": entries, "exclusions": exclusions}


def _sample_human_indices(times: np.ndarray, observed: np.ndarray, limit: int | None) -> np.ndarray:
    selected = _sample_indices(times, limit, by_time=True)
    if limit is None or len(times) <= limit:
        return selected
    # Protect endpoints and two genuine observations while keeping the frame cap.
    good = np.flatnonzero(observed)
    required = set((0, len(times) - 1, int(good[0]), int(good[-1])))
    if len(required) > limit:
        raise ValueError("Human frame limit cannot retain endpoints and observations")
    selected_set = set(map(int, selected)) | required
    if len(selected_set) > limit:
        optional = sorted(selected_set - required)
        # Remove the most redundant sampled times first.
        while len(selected_set) > limit:
            drop = min(optional, key=lambda i: min(abs(times[i] - times[j])
                                                    for j in selected_set if j != i))
            optional.remove(drop)
            selected_set.remove(drop)
    return np.asarray(sorted(selected_set), dtype=np.int64)


class RH20THand3DTrajectoryDataset(Dataset):
    def __init__(self, root_dir: str, use_6_keypoints: bool = False,
                 manifest_path: str | None = None, preserve_timing: bool = True,
                 max_human_frames: int | None = None, max_robot_frames: int | None = None,
                 human_representation: str = "base_xyz",
                 geometry_manifest_path: str | None = None):
        if not preserve_timing:
            raise ValueError("hand_3d_base requires preserve_timing=True")
        if max_human_frames is not None and max_human_frames < 4:
            raise ValueError("max_human_frames must be at least four for hand_3d_base")
        if max_robot_frames is not None and max_robot_frames < 2:
            raise ValueError("max_robot_frames must be at least two")
        if human_representation not in HUMAN_REPRESENTATIONS:
            raise ValueError(f"Unknown human representation: {human_representation}")
        self.root_dir = root_dir
        self.human_representation = human_representation
        self._global_human_transform = None
        self.use_6_keypoints = use_6_keypoints
        self.max_human_frames = max_human_frames
        self.max_robot_frames = max_robot_frames
        self.preserve_timing = preserve_timing
        self.manifest = (json.loads(Path(manifest_path).read_text()) if manifest_path
                         else build_hand_3d_trajectory_manifest(root_dir))
        if (self.manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
                or self.manifest.get("human_source") != HUMAN_SOURCE):
            raise ValueError("Unsupported hand_3d_base trajectory manifest")
        if Path(self.manifest["data_root"]).resolve() != Path(root_dir).resolve():
            raise ValueError("Trajectory manifest belongs to a different data root")
        self.entries = {entry["scene_id"]: entry for entry in self.manifest["scenes"]}
        self.geometry = (json.loads(Path(geometry_manifest_path).read_text())
                         if geometry_manifest_path else None)
        self.dataset_geometry = self.geometry is not None and "scenes" not in self.geometry
        if self.geometry is not None:
            if self.dataset_geometry:
                from .grasp_geometry import validate_profile
                if (use_6_keypoints or not self.geometry.get("profiles")
                        or Path(geometry_manifest_path).resolve()
                        != (Path(root_dir) / "geometry.json").resolve()):
                    raise ValueError("Grasp geometry needs dataset parameters and full 21-point hands")
                for profile in self.geometry["profiles"].values():
                    validate_profile(profile)
            elif (self.geometry.get("schema_version") != 1 or use_6_keypoints
                    or Path(self.geometry["data_root"]).resolve() != Path(root_dir).resolve()
                    or set(self.geometry["scenes"]) != set(self.entries)):
                raise ValueError("Grasp geometry must match the full 21-point trajectory inventory")
        if len(self.entries) != len(self.manifest["scenes"]):
            raise ValueError("Duplicate scene identity in trajectory manifest")
        task_names = sorted({entry["task_id"] for entry in self.manifest["scenes"]})
        task_indices = {name: i for i, name in enumerate(task_names)}
        self.scenes = []
        for entry in self.manifest["scenes"]:
            path = Path(root_dir) / entry["scene_id"]
            if not (path / "human_hand_3d.npy").is_file() or not (path / "tcp_base.npy").is_file():
                raise FileNotFoundError(f"Manifest scene files are missing: {path}")
            self.scenes.append(SceneRecord(scene_path=str(path),
                                           human_pose_path=str(path / "human_hand_3d.npy"),
                                           tcp_base_path=str(path / "tcp_base.npy"),
                                           task_idx=task_indices[entry["task_id"]]))
        if not self.scenes:
            raise ValueError(f"No valid hand_3d_base scenes found in {root_dir}")

    def __len__(self) -> int:
        return len(self.scenes)

    def with_human_transform(self, transform):
        """Return an evaluation view that changes only the human coordinate frame."""
        variant = copy(self)
        variant._global_human_transform = validate_global_transform(transform)
        return variant

    def __getitem__(self, idx: int) -> dict:
        scene = self.scenes[idx]
        scene_id = f"{Path(scene.scene_path).parent.name}/{Path(scene.scene_path).name}"
        entry = self.entries[scene_id]
        human, _ = load_trajectory_file(scene.human_pose_path, entry["human_sha256"])
        robot, _ = load_trajectory_file(scene.tcp_base_path, entry["robot_sha256"])
        grasp = None
        if self.geometry is not None:
            if self.dataset_geometry:
                grasp, _ = load_trajectory_file(Path(scene.scene_path) / "geometry.npy")
            else:
                prepared = self.geometry["scenes"][scene_id]
                if prepared["robot_sha256"] != entry["robot_sha256"]:
                    raise ValueError(f"Grasp geometry belongs to a different TCP sequence: {scene_id}")
                grasp, _ = load_trajectory_file(prepared["path"], prepared["sha256"])
        result = {key: [] for key in ("human_poses", "tcp_bases", "camera_ids", "human_times",
                                     "robot_times", "human_observed", "human_frame_indices")}
        for camera_id, camera_entry in sorted(entry["cameras"].items()):
            camera = human["cameras"][camera_id]
            observations, observed, quality = _validated_camera(camera, robot[camera_id])
            if (quality["selected_track_id"] != camera_entry["selected_track_id"]
                    or quality["observed_frames"] != camera_entry["observed_frames"]):
                raise ValueError(f"Hand selection changed since manifest creation: {scene_id}/{camera_id}")
            frame_ids = np.asarray(camera["frame_index"], dtype=np.int64)
            times = np.asarray(camera["timestamp_ms"], dtype=np.float64)
            times = (times - times[0]) / 1000.0
            points = np.asarray(camera["landmarks_base_m"], dtype=np.float32)
            poses = np.zeros((len(times), 21, 3), dtype=np.float32)
            slot = np.argmax(observations, axis=1)
            poses[observed] = points[np.flatnonzero(observed), slot[observed]]
            first_palm_pose = None
            if self.human_representation == "palm_relative":
                first = int(np.flatnonzero(observed)[0])
                first_palm_pose = np.asarray(camera["palm_pose_base"])[first, slot[first]]
            poses = hand_xyz_features(poses, observed, self.human_representation,
                                      first_palm_pose, self._global_human_transform)
            selected = _sample_human_indices(times, observed, self.max_human_frames)
            if self.use_6_keypoints:
                poses = poses[:, KEYPOINT_INDICES, :]
            robot_frames = robot[camera_id]
            tcp = np.asarray([frame["tcp"] for frame in robot_frames], dtype=np.float32)
            if grasp is not None:
                tcp = np.asarray(grasp[camera_id], dtype=np.float32)
                if (tcp.shape != (len(robot_frames), 8) or not np.isfinite(tcp).all()
                        or (tcp[:, 7] < 0).any() or (tcp[:, 7] > 1).any()):
                    raise ValueError(f"Invalid calibrated grasp sequence: {scene_id}/{camera_id}")
            robot_times = np.asarray([frame["timestamp"] for frame in robot_frames], dtype=np.float64)
            robot_times = (robot_times - robot_times[0]) / 1000.0
            robot_selected = _sample_indices(robot_times, self.max_robot_frames, by_time=True)
            result["human_poses"].append(torch.from_numpy(poses[selected]))
            result["human_times"].append(torch.as_tensor(times[selected], dtype=torch.float32))
            result["human_observed"].append(torch.as_tensor(observed[selected], dtype=torch.bool))
            result["human_frame_indices"].append(torch.as_tensor(frame_ids[selected], dtype=torch.long))
            result["tcp_bases"].append(torch.from_numpy(tcp[robot_selected]))
            result["robot_times"].append(torch.as_tensor(robot_times[robot_selected], dtype=torch.float32))
            result["camera_ids"].append(camera_id)
        result.update({"scene_idx": idx, "task_idx": scene.task_idx,
                       "scene_id": scene_id, "task_id": entry["task_id"],
                       "scene_path": scene.scene_path,
                       "sampled_once": self.max_robot_frames is not None})
        return result
