import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video import collate_video_pairs, evaluate_video_retrieval


class _ToyVideoDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "sample_id": f"sample_{index}",
            "pair_id": f"pair_{index}",
            "dataset_name": "toy",
            "task_id": f"task_{index}",
            "scene_id": f"scene_{index}",
            "camera_id": f"cam_{index}",
            "task_index": index,
            "scene_index": index,
            "human_video_path": f"/tmp/human_{index}.mp4",
            "robot_video_path": f"/tmp/robot_{index}.mp4",
            "human_pixel_values": torch.full((2, 3, 4, 4), float(index + 1)),
            "robot_pixel_values": torch.full((2, 3, 4, 4), float(index + 1)),
            "metadata": {},
        }


class _ToyModel(torch.nn.Module):
    def forward(self, human_pixel_values, robot_pixel_values):
        human_embeddings = human_pixel_values.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        robot_embeddings = robot_pixel_values.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return {
            "human_embeddings": torch.nn.functional.normalize(human_embeddings, dim=1),
            "robot_embeddings": torch.nn.functional.normalize(robot_embeddings, dim=1),
            "logit_scale_inter": torch.tensor(1.0),
            "logit_scale_intra": torch.tensor(1.0),
        }


def test_evaluate_video_retrieval_reports_bidirectional_metrics():
    dataloader = DataLoader(_ToyVideoDataset(), batch_size=2, shuffle=False, collate_fn=collate_video_pairs)
    result = evaluate_video_retrieval(_ToyModel(), dataloader, device=torch.device("cpu"))
    assert "human_to_robot" in result["metrics"]
    assert "robot_to_human" in result["metrics"]
    assert "task" in result["metrics"]["human_to_robot"]
    assert result["similarity_matrix"].shape == (2, 2)
