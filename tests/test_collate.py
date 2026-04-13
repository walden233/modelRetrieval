import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.data.rh20t.collate import collate_trajectories


def test_collate_trajectories_shapes():
    batch = [
        {
            "human_poses": [torch.randn(3, 6, 3)],
            "tcp_bases": [torch.randn(5, 7)],
            "scene_idx": 0,
            "task_idx": 1,
        },
        {
            "human_poses": [torch.randn(2, 6, 3)],
            "tcp_bases": [torch.randn(4, 7)],
            "scene_idx": 1,
            "task_idx": 2,
        },
    ]

    result = collate_trajectories(batch)
    assert result["human_poses"].shape == (2, 3, 6, 3)
    assert result["tcp_bases"].shape == (2, 5, 7)
    assert result["human_mask"].shape == (2, 3)
    assert result["tcp_mask"].shape == (2, 5)
