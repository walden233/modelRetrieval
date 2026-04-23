import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video.losses import intra_domain_consistency_loss, multi_positive_video_contrastive_loss


def test_multi_positive_video_contrastive_loss_returns_finite_value():
    human = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    robot = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = multi_positive_video_contrastive_loss(human, robot, labels, labels, torch.tensor(10.0))
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_intra_domain_consistency_loss_returns_small_value_for_identical_views():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loss = intra_domain_consistency_loss(embeddings, embeddings, torch.tensor(10.0))
    assert torch.isfinite(loss)
    assert loss.item() < 1.0
