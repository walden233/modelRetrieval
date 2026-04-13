import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from decord import VideoReader, cpu


def sample_video_frames(video_path: str | Path, num_frames: int = 16):
    try:
        video_reader = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(video_reader)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        return list(video_reader.get_batch(indices).asnumpy())
    except Exception as exc:
        print(f"Error reading video file {video_path}: {exc}")
        return None


@dataclass
class SceneRecord:
    scene_path: str
    video_pairs: List[Tuple[str, str]] = field(default_factory=list)
    human_pose_path: str = ""
    tcp_base_path: str = ""
    task_idx: Optional[int] = None


def scan_task_scenes(root_dir: str) -> List[List[SceneRecord]]:
    tasks: List[List[SceneRecord]] = []
    task_folders = sorted([name for name in os.listdir(root_dir) if name.startswith("task_")])

    for task_folder in task_folders:
        task_path = os.path.join(root_dir, task_folder)
        if not os.path.isdir(task_path):
            continue

        task_scenes: List[SceneRecord] = []
        scene_folders = sorted([name for name in os.listdir(task_path) if name.startswith("scene_")])

        for scene_folder in scene_folders:
            scene_path = os.path.join(task_path, scene_folder)
            scene = SceneRecord(
                scene_path=scene_path,
                human_pose_path=os.path.join(scene_path, "human_pose.npy"),
                tcp_base_path=os.path.join(scene_path, "tcp_base.npy"),
            )
            human_videos: Dict[str, str] = {}
            robot_videos: Dict[str, str] = {}

            for filename in os.listdir(scene_path):
                if filename.endswith("_human.mp4"):
                    camera_id = filename.replace("_human.mp4", "")
                    human_videos[camera_id] = os.path.join(scene_path, filename)
                elif filename.endswith("_robot.mp4"):
                    camera_id = filename.replace("_robot.mp4", "")
                    robot_videos[camera_id] = os.path.join(scene_path, filename)

            for camera_id, human_path in human_videos.items():
                if camera_id in robot_videos:
                    scene.video_pairs.append((human_path, robot_videos[camera_id]))

            if scene.video_pairs:
                task_scenes.append(scene)

        if task_scenes:
            tasks.append(task_scenes)

    return tasks


def scan_valid_trajectory_scenes(root_dir: str) -> List[SceneRecord]:
    scenes: List[SceneRecord] = []
    task_folders = sorted([name for name in os.listdir(root_dir) if name.startswith("task_")])

    for task_idx, task_folder in enumerate(task_folders):
        task_path = os.path.join(root_dir, task_folder)
        if not os.path.isdir(task_path):
            continue

        scene_folders = sorted([name for name in os.listdir(task_path) if name.startswith("scene_")])
        for scene_folder in scene_folders:
            scene_path = os.path.join(task_path, scene_folder)
            human_pose_path = os.path.join(scene_path, "human_pose.npy")
            tcp_base_path = os.path.join(scene_path, "tcp_base.npy")

            if os.path.exists(human_pose_path) and os.path.exists(tcp_base_path):
                scenes.append(
                    SceneRecord(
                        scene_path=scene_path,
                        human_pose_path=human_pose_path,
                        tcp_base_path=tcp_base_path,
                        task_idx=task_idx,
                    )
                )

    return scenes
