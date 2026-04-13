import os

import cv2
import numpy as np


def verify_hand_pose_file(scene_path: str) -> None:
    npy_path = os.path.join(scene_path, "human_pose.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing human_pose.npy in {scene_path}")

    data = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise TypeError("human_pose.npy must contain a dict")

    for camera_serial, trajectory in data.items():
        video_path = os.path.join(scene_path, f"cam_{camera_serial}_human.mp4")
        if not os.path.exists(video_path):
            continue
        capture = cv2.VideoCapture(video_path)
        video_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if len(trajectory) != video_frame_count:
            raise ValueError(f"Trajectory length mismatch for {camera_serial}: {len(trajectory)} != {video_frame_count}")
        if trajectory:
            sample_record = trajectory[len(trajectory) // 2]
            if "frame_index" not in sample_record or "hands_landmarks" not in sample_record:
                raise ValueError(f"Malformed record for {camera_serial}")
