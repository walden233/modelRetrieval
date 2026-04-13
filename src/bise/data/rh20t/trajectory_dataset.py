from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .scanner import SceneRecord, scan_valid_trajectory_scenes


class RH20TTrajectoryDataset(Dataset):
    KEYPOINT_INDICES = [0, 4, 8, 12, 16, 20]

    def __init__(self, root_dir: str, use_6_keypoints: bool = False):
        self.root_dir = root_dir
        self.use_6_keypoints = use_6_keypoints
        self.scenes = scan_valid_trajectory_scenes(root_dir)
        if not self.scenes:
            raise ValueError(f"No valid scenes found in {root_dir}")

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> Dict[str, List[torch.Tensor]]:
        scene: SceneRecord = self.scenes[idx]
        try:
            human_pose_dict = np.load(scene.human_pose_path, allow_pickle=True).item()
            tcp_base_dict = np.load(scene.tcp_base_path, allow_pickle=True).item()
        except Exception as exc:
            print(f"Warning: failed to load trajectory data from {scene.scene_path}: {exc}")
            return self.__getitem__((idx + 1) % len(self))

        common_camera_ids = sorted(list(human_pose_dict.keys() & tcp_base_dict.keys()))
        if not common_camera_ids:
            return self.__getitem__((idx + 1) % len(self))

        pose_tensors = []
        tcp_tensors = []
        for camera_id in common_camera_ids:
            valid_landmarks = [
                frame["hands_landmarks"][0]
                for frame in human_pose_dict[camera_id]
                if frame.get("hands_landmarks")
            ]
            all_tcps = [record["tcp"] for record in tcp_base_dict[camera_id]]
            if not valid_landmarks or not all_tcps:
                continue

            pose_trajectory = np.stack(valid_landmarks, axis=0)
            if self.use_6_keypoints:
                pose_trajectory = pose_trajectory[:, self.KEYPOINT_INDICES, :]

            tcp_trajectory = np.stack(all_tcps, axis=0)
            pose_tensors.append(torch.from_numpy(pose_trajectory).float())
            tcp_tensors.append(torch.from_numpy(tcp_trajectory).float())

        if not pose_tensors:
            return self.__getitem__((idx + 1) % len(self))

        return {
            "human_poses": pose_tensors,
            "tcp_bases": tcp_tensors,
            "scene_idx": idx,
            "task_idx": scene.task_idx if scene.task_idx is not None else -1,
        }
