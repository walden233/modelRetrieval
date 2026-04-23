from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from bise.common.schemas import VideoPairSample
from bise.modalities.video.batch import extract_pixel_values
from bise.modalities.video.frame_sampling import sample_video_frames
from bise.modalities.video.transforms import apply_video_transforms


class VideoPairDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        processor,
        num_frames: int = 16,
        sampling_strategy: str = "uniform",
        sampling_stride: int | None = None,
        deterministic: bool = False,
        seed: int = 42,
        transform_config: dict | None = None,
        debug_max_samples: int | None = None,
    ):
        self.metadata = pd.read_csv(csv_file)
        self.processor = processor
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.sampling_stride = sampling_stride
        self.deterministic = deterministic
        self.seed = seed
        self.transform_config = transform_config or {}
        self.samples = self._build_samples()
        if debug_max_samples is not None:
            self.samples = self.samples[:debug_max_samples]
        self.task_to_index = {task_id: index for index, task_id in enumerate(sorted({sample.task_id for sample in self.samples}))}
        self.scene_to_index = {
            scene_id: index for index, scene_id in enumerate(sorted({sample.scene_id for sample in self.samples}))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_seed = self.seed + idx if self.deterministic else None
        human_frames = sample_video_frames(
            sample.human_video_path,
            num_frames=self.num_frames,
            strategy=self.sampling_strategy,
            seed=sample_seed,
            stride=self.sampling_stride,
        )
        robot_frames = sample_video_frames(
            sample.robot_video_path,
            num_frames=self.num_frames,
            strategy=self.sampling_strategy,
            seed=sample_seed,
            stride=self.sampling_stride,
        )
        if human_frames is None or robot_frames is None:
            return self.__getitem__((idx + 1) % len(self))
        inputs = self.processor([human_frames, robot_frames], return_tensors="pt")
        pixel_values = extract_pixel_values(inputs)
        return {
            "sample_id": sample.sample_id,
            "pair_id": sample.pair_id,
            "dataset_name": sample.dataset_name,
            "task_id": sample.task_id,
            "scene_id": sample.scene_id,
            "camera_id": sample.camera_id,
            "task_index": self.task_to_index[sample.task_id],
            "scene_index": self.scene_to_index[sample.scene_id],
            "human_video_path": sample.human_video_path,
            "robot_video_path": sample.robot_video_path,
            "human_pixel_values": apply_video_transforms(pixel_values[0], self.transform_config, seed=sample_seed),
            "robot_pixel_values": apply_video_transforms(pixel_values[1], self.transform_config, seed=sample_seed),
            "metadata": dict(sample.metadata),
        }

    def _build_samples(self):
        samples = []
        for index, row in self.metadata.iterrows():
            task_id = str(row.get("task_id", Path(row["robot_video_path"]).stem))
            scene_id = str(row.get("scene_id", f"{task_id}_scene_0"))
            camera_id = str(row.get("camera_id", f"cam_{index}"))
            sample_id = str(row.get("sample_id", f"whirl::{task_id}::{scene_id}::{camera_id}"))
            samples.append(
                VideoPairSample(
                    sample_id=sample_id,
                    pair_id=str(row.get("pair_id", sample_id)),
                    dataset_name="whirl",
                    task_id=task_id,
                    scene_id=scene_id,
                    camera_id=camera_id,
                    human_video_path=str(row["human_video_path"]),
                    robot_video_path=str(row["robot_video_path"]),
                    metadata={"csv_row_index": int(index)},
                )
            )
        return samples
