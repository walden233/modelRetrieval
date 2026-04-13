import os
import re
from typing import Any, Dict, List

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands


def find_scenes_to_process(root_dir: str) -> List:
    tasks = []
    camera_pattern = re.compile(r"cam_([a-zA-Z0-9]+)_human\.mp4")
    for dirpath, _, filenames in os.walk(root_dir):
        if any(name.endswith("_human.mp4") for name in filenames) and "tcp_base.npy" in filenames:
            for filename in filenames:
                match = camera_pattern.match(filename)
                if match:
                    tasks.append((dirpath, os.path.join(dirpath, filename), match.group(1)))
    return tasks


def structure_frame_data(frame_index: int, mediapipe_results: Any) -> Dict[str, Any]:
    frame_landmarks = []
    if mediapipe_results.multi_hand_world_landmarks:
        for hand_world_landmarks in mediapipe_results.multi_hand_world_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark], dtype=np.float32)
            frame_landmarks.append(landmarks)
    return {"frame_index": frame_index, "hands_landmarks": frame_landmarks}


def extract_hand_poses_from_video(video_path: str) -> List:
    trajectory = []
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return trajectory

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands_model:
        for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
            success, image = capture.read()
            if not success:
                break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            trajectory.append(structure_frame_data(frame_idx, hands_model.process(image_rgb)))

    capture.release()
    return trajectory


def group_tasks_by_scene(tasks: List) -> Dict:
    grouped_tasks = {}
    for scene_path, video_path, camera_serial in tasks:
        grouped_tasks.setdefault(scene_path, []).append((video_path, camera_serial))
    return grouped_tasks
