from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

from bise.common.schemas import VideoPairSample
from bise.modalities.video.batch import extract_pixel_values
from bise.modalities.video.frame_sampling import sample_video_frames
from bise.modalities.video.transforms import apply_video_transforms

from .scanner import scan_task_scenes


class RH20TVideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        processor,
        num_frames: int = 16,
        sampling_strategy: str = "uniform",
        sampling_stride: int | None = None,
        deterministic: bool = False,
        seed: int = 42,
        transform_config: Dict | None = None,
        max_pairs_per_scene: int | None = None,
        debug_max_samples: int | None = None,
    ):
        self.root_dir = root_dir
        self.processor = processor
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.sampling_stride = sampling_stride
        self.deterministic = deterministic
        self.seed = seed
        self.transform_config = transform_config or {}
        self.max_pairs_per_scene = max_pairs_per_scene
        self.samples = self._build_samples(scan_task_scenes(root_dir))

        if self.processor is None:
            raise ValueError("processor is required for video preprocessing")
        if not self.samples:
            raise ValueError(f"No valid video pairs found in {root_dir}")
        if debug_max_samples is not None:
            self.samples = self.samples[:debug_max_samples]

        self.task_to_index = {task_id: index for index, task_id in enumerate(sorted({sample.task_id for sample in self.samples}))}
        self.scene_to_index = {
            scene_id: index for index, scene_id in enumerate(sorted({sample.scene_id for sample in self.samples}))
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
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
        human_pixel_values = apply_video_transforms(pixel_values[0], self.transform_config, seed=sample_seed)
        robot_pixel_values = apply_video_transforms(pixel_values[1], self.transform_config, seed=sample_seed)
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
            "human_pixel_values": human_pixel_values,
            "robot_pixel_values": robot_pixel_values,
            "metadata": dict(sample.metadata),
        }

    def _build_samples(self, tasks: List[List]) -> List[VideoPairSample]:
        samples: List[VideoPairSample] = []
        for task_scenes in tasks:
            for scene in task_scenes:
                task_id = Path(scene.scene_path).parent.name
                scene_name = Path(scene.scene_path).name
                # RH20T 的 scene_1 / scene_2 会在不同 task 下重复。
                # 评估 scene-level 检索时必须把 task 也纳入 scene 标识，避免标签碰撞。
                scene_id = f"{task_id}/{scene_name}"
                video_pairs = list(scene.video_pairs)
                if self.max_pairs_per_scene is not None:
                    video_pairs = video_pairs[: self.max_pairs_per_scene]
                for human_path, robot_path in video_pairs:
                    camera_id = Path(human_path).name.replace("_human.mp4", "")
                    sample_id = f"rh20t::{task_id}::{scene_name}::{camera_id}"
                    samples.append(
                        VideoPairSample(
                            sample_id=sample_id,
                            pair_id=sample_id,
                            dataset_name="rh20t",
                            task_id=task_id,
                            scene_id=scene_id,
                            camera_id=camera_id,
                            human_video_path=str(human_path),
                            robot_video_path=str(robot_path),
                            metadata={"scene_path": scene.scene_path, "scene_name": scene_name},
                        )
                    )
        return samples
