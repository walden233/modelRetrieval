import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video.batch import collate_video_pairs


def test_collate_video_pairs_shapes_and_metadata():
    batch = [
        {
            "sample_id": "a",
            "pair_id": "p1",
            "dataset_name": "rh20t",
            "task_id": "task_1",
            "scene_id": "scene_1",
            "camera_id": "cam_0",
            "task_index": 0,
            "scene_index": 0,
            "human_video_path": "/tmp/h1.mp4",
            "robot_video_path": "/tmp/r1.mp4",
            "human_pixel_values": torch.randn(4, 3, 8, 8),
            "robot_pixel_values": torch.randn(4, 3, 8, 8),
            "metadata": {"dataset": "rh20t"},
        },
        {
            "sample_id": "b",
            "pair_id": "p2",
            "dataset_name": "whirl",
            "task_id": "task_2",
            "scene_id": "scene_2",
            "camera_id": "cam_1",
            "task_index": 1,
            "scene_index": 1,
            "human_video_path": "/tmp/h2.mp4",
            "robot_video_path": "/tmp/r2.mp4",
            "human_pixel_values": torch.randn(4, 3, 8, 8),
            "robot_pixel_values": torch.randn(4, 3, 8, 8),
            "metadata": {"dataset": "whirl"},
        },
    ]

    result = collate_video_pairs(batch)
    assert result["human_pixel_values"].shape == (2, 4, 3, 8, 8)
    assert result["robot_pixel_values"].shape == (2, 4, 3, 8, 8)
    assert result["task_indices"].tolist() == [0, 1]
    assert result["scene_indices"].tolist() == [0, 1]
    assert result["sample_ids"] == ["a", "b"]
