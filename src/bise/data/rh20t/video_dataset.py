import random
from typing import Dict

from torch.utils.data import Dataset

from .scanner import sample_video_frames, scan_task_scenes


class RH20TVideoDataset(Dataset):
    def __init__(self, root_dir: str, scene_num: int, cam_num: int, processor, num_frames: int = 16):
        self.root_dir = root_dir
        self.scene_num = scene_num
        self.cam_num = cam_num
        self.processor = processor
        self.num_frames = num_frames
        self.tasks = scan_task_scenes(root_dir)

        if self.processor is None:
            raise ValueError("processor is required for video preprocessing")

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict:
        task_scenes = self.tasks[idx]
        selected_scenes = task_scenes if self.scene_num >= len(task_scenes) else random.sample(task_scenes, self.scene_num)
        batch_human_frames = []
        batch_robot_frames = []

        for scene in selected_scenes:
            if not scene.video_pairs:
                continue
            selected_pairs = scene.video_pairs if self.cam_num >= len(scene.video_pairs) else random.sample(scene.video_pairs, self.cam_num)

            for human_path, robot_path in selected_pairs:
                human_frames = sample_video_frames(human_path, self.num_frames)
                robot_frames = sample_video_frames(robot_path, self.num_frames)
                if human_frames is not None and robot_frames is not None:
                    batch_human_frames.append(human_frames)
                    batch_robot_frames.append(robot_frames)

        if not batch_human_frames:
            return self.__getitem__((idx + 1) % len(self))

        processed_human = []
        processed_robot = []
        for human_frames, robot_frames in zip(batch_human_frames, batch_robot_frames):
            inputs = self.processor([human_frames, robot_frames], return_tensors="pt")
            pixel_values = inputs.get("pixel_values") or inputs.get("pixel_values_videos")
            if pixel_values is not None:
                processed_human.append(pixel_values[0])
                processed_robot.append(pixel_values[1])

        return {
            "human_pixel_values": processed_human,
            "robot_pixel_values": processed_robot,
            "task_idx": idx,
        }
