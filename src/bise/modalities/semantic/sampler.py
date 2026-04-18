from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def extract_camera_id(video_path: str | Path, suffix: str) -> str:
    filename = Path(video_path).name
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return Path(video_path).stem


def select_scene_camera_pair(video_pairs: Iterable[Tuple[str, str]], strategy_config: Dict[str, Any]) -> Tuple[str, str]:
    pairs = list(video_pairs)
    if not pairs:
        raise ValueError("No video pairs provided for scene sampling.")

    preferred_robot_cam_id = str(strategy_config.get("preferred_robot_cam_id", "")).strip()
    preferred_human_cam_id = str(strategy_config.get("preferred_human_cam_id", "")).strip()

    if preferred_robot_cam_id and preferred_human_cam_id:
        for human_path, robot_path in pairs:
            human_id = extract_camera_id(human_path, "_human.mp4")
            robot_id = extract_camera_id(robot_path, "_robot.mp4")
            if human_id == preferred_human_cam_id and robot_id == preferred_robot_cam_id:
                return human_path, robot_path

    pairs.sort(key=lambda item: (extract_camera_id(item[0], "_human.mp4"), extract_camera_id(item[1], "_robot.mp4")))
    return pairs[0]
