import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .scanner import SceneRecord, scan_valid_trajectory_scenes
from .trajectory_manifest import SCHEMA_VERSION, build_trajectory_manifest, load_trajectory_file


def _sample_indices(times, limit, by_time=False):
    if limit is None or len(times) <= limit:
        return np.arange(len(times))
    if limit < 2:
        raise ValueError("A trajectory frame limit must be at least two")
    if not by_time:
        return np.linspace(0, len(times) - 1, limit).round().astype(int)
    # Select actual observations near a time grid; never interpolate across gaps.
    grid = np.linspace(times[0], times[-1], limit)
    right = np.searchsorted(times, grid).clip(0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(abs(times[left] - grid) <= abs(times[right] - grid), left, right)
    return np.unique(nearest)


class RH20TTrajectoryDataset(Dataset):
    KEYPOINT_INDICES = [0, 4, 8, 12, 16, 20]

    def __init__(self, root_dir: str, use_6_keypoints: bool = False,
                 manifest_path: str | None = None, preserve_timing: bool = False,
                 max_human_frames: int | None = None, max_robot_frames: int | None = None):
        self.root_dir = root_dir
        self.use_6_keypoints = use_6_keypoints
        self.preserve_timing = preserve_timing
        self.max_human_frames = max_human_frames
        self.max_robot_frames = max_robot_frames
        for limit in (max_human_frames, max_robot_frames):
            if limit is not None and limit < 2:
                raise ValueError("Frame limits must be at least two")
        self.manifest = (json.loads(Path(manifest_path).read_text()) if manifest_path
                         else build_trajectory_manifest(root_dir))
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported trajectory manifest schema")
        if Path(self.manifest["data_root"]).resolve() != Path(root_dir).resolve():
            raise ValueError("Trajectory manifest belongs to a different data root")
        self.entries = {entry["scene_id"]: entry for entry in self.manifest["scenes"]}
        if len(self.entries) != len(self.manifest["scenes"]):
            raise ValueError("Duplicate scene identity in trajectory manifest")
        self.scenes = [scene for scene in scan_valid_trajectory_scenes(root_dir)
                       if self._scene_id(scene) in self.entries]
        if len(self.scenes) != len(self.entries):
            raise ValueError("Manifest scenes are missing from the dataset")
        if not self.scenes:
            raise ValueError(f"No valid scenes found in {root_dir}")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict:
        scene: SceneRecord = self.scenes[idx]
        entry = self.entries[self._scene_id(scene)]
        # Hash-check every read so a saved split cannot silently change its inputs.
        human, _ = load_trajectory_file(scene.human_pose_path, entry["human_sha256"])
        robot, _ = load_trajectory_file(scene.tcp_base_path, entry["robot_sha256"])
        result = {key: [] for key in ("human_poses", "tcp_bases", "camera_ids", "human_times",
                                     "robot_times", "human_observed", "human_frame_indices")}
        for camera in sorted(entry["cameras"]):
            frames = human[camera]
            observed = np.asarray([bool(frame.get("hands_landmarks")) for frame in frames])
            ids = np.asarray([frame["frame_index"] for frame in frames], dtype=np.float64)
            poses = np.stack([frame["hands_landmarks"][0] if ok else np.zeros((21, 3))
                              for frame, ok in zip(frames, observed)])
            if not self.preserve_timing:
                poses, ids, observed = poses[observed], ids[observed], observed[observed]
            h_indices = _sample_indices(ids, self.max_human_frames, self.preserve_timing)
            if not observed[h_indices].any():
                # Keep at least one genuine observation without inventing landmarks.
                h_indices = np.sort(np.unique(np.append(h_indices[:-1], np.flatnonzero(observed)[0])))
            poses, ids, observed = poses[h_indices], ids[h_indices], observed[h_indices]
            if self.use_6_keypoints:
                poses = poses[:, self.KEYPOINT_INDICES]
            tcps = np.stack([frame["tcp"] for frame in robot[camera]])
            times = np.asarray([frame["timestamp"] for frame in robot[camera]], dtype=np.float64)
            times = (times - times[0]) / 1000.0
            r_indices = _sample_indices(times, self.max_robot_frames, self.preserve_timing)
            result["human_poses"].append(torch.as_tensor(poses, dtype=torch.float32))
            result["tcp_bases"].append(torch.as_tensor(tcps[r_indices], dtype=torch.float32))
            result["human_times"].append(torch.as_tensor(ids - frames[0]["frame_index"], dtype=torch.float32))
            result["robot_times"].append(torch.as_tensor(times[r_indices], dtype=torch.float32))
            result["human_frame_indices"].append(torch.as_tensor(ids, dtype=torch.long))
            result["human_observed"].append(torch.as_tensor(observed, dtype=torch.bool))
            result["camera_ids"].append(camera)
        result.update({"scene_idx": idx, "task_idx": scene.task_idx if scene.task_idx is not None else -1,
                       "scene_id": self._scene_id(scene), "task_id": self._task_id(scene),
                       "scene_path": scene.scene_path, "sampled_once": self.max_robot_frames is not None})
        return result

    @staticmethod
    def _scene_id(scene: SceneRecord) -> str:
        path = Path(scene.scene_path)
        return f"{path.parent.name}/{path.name}"

    @staticmethod
    def _task_id(scene: SceneRecord) -> str:
        return Path(scene.scene_path).parent.name
