import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video.losses import (
    intra_domain_consistency_loss, multi_positive_video_contrastive_loss,
    negative_task_separation_loss,
)
from bise.modalities.video.trainer import train_video_epoch


def test_multi_positive_video_contrastive_loss_returns_finite_value():
    human = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    robot = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loss = multi_positive_video_contrastive_loss(human, robot, labels, labels, torch.tensor(10.0))
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_intra_domain_consistency_loss_returns_small_value_for_identical_views():
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.arange(2)
    loss = intra_domain_consistency_loss(
        embeddings, embeddings, torch.tensor(10.0), labels, labels,
    )
    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_task_separation_masks_positives_and_leaves_separated_negatives_untouched():
    human = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    robot = human.detach().clone().requires_grad_()
    # Identical same-task cross-domain pairs are positives, never repelled.
    loss = negative_task_separation_loss(human, robot, torch.tensor([0, 1]))
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(human.grad) == torch.count_nonzero(robot.grad) == 0
    identical = torch.tensor([[1., 0.]] * 3, requires_grad=True)
    loss = negative_task_separation_loss(identical, identical, torch.tensor([0, 0, 1]))
    assert loss.item() == pytest.approx(.7 ** 2)
    loss = negative_task_separation_loss(identical, identical, torch.zeros(3))
    assert loss.item() == 0 and loss.requires_grad
    for margin in (-1.1, 1.1):
        with pytest.raises(ValueError, match="margin"):
            negative_task_separation_loss(human, robot, torch.tensor([0, 1]), margin)


def test_task_separation_update_reduces_different_task_cosine_in_both_domains():
    human = torch.nn.Parameter(torch.tensor([[1., 0.], [.8, .6]]))
    robot = torch.nn.Parameter(human.detach().clone())
    optimizer = torch.optim.SGD([human, robot], lr=.2)
    before = negative_task_separation_loss(
        F.normalize(human, dim=-1), F.normalize(robot, dim=-1), torch.tensor([0, 1]))
    before.backward()
    assert human.grad.abs().sum() > 0 and robot.grad.abs().sum() > 0
    optimizer.step()
    after = negative_task_separation_loss(
        F.normalize(human, dim=-1), F.normalize(robot, dim=-1), torch.tensor([0, 1]))
    assert after < before


@pytest.mark.parametrize("scene_weight", [None, 1.0])
def test_video_training_uses_scene_and_task_positives_in_both_domains(monkeypatch, scene_weight):
    from bise.modalities.video import trainer

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(4, 4, bias=False)

        def encode_human(self, videos):
            return F.normalize(self.projection(videos), dim=1)

        def encode_robot(self, videos):
            return F.normalize(self.projection(videos), dim=1)

        def forward(self, human, robot):
            return {"human_embeddings": self.encode_human(human),
                    "robot_embeddings": self.encode_robot(robot),
                    "logit_scale_inter": torch.tensor(2.0),
                    "logit_scale_intra": torch.tensor(2.0)}

    torch.manual_seed(7)
    model = TinyModel()
    batch = {"human_pixel_values": torch.eye(4), "robot_pixel_values": torch.eye(4).roll(1, 1),
             "scene_indices": torch.tensor([0, 0, 1, 2]),
             "task_indices": torch.tensor([0, 0, 0, 1])}
    seen_labels = []
    seen_weights = []
    losses = []

    def capture_loss(z1, z2, scale, scene_labels, task_labels, scene_positive_weight=0.5):
        seen_labels.append((scene_labels.clone(), task_labels.clone()))
        seen_weights.append(scene_positive_weight)
        loss = intra_domain_consistency_loss(
            z1, z2, scale, scene_labels, task_labels,
            scene_positive_weight=scene_positive_weight,
        )
        losses.append(loss.detach())
        return loss

    monkeypatch.setattr(trainer, "intra_domain_consistency_loss", capture_loss)
    monkeypatch.setattr(trainer, "_apply_batch_transform", lambda videos, config: videos)
    before = model.projection.weight.detach().clone()
    loss_config = {"name": "info_nce", "lambda_intra": 0.2}
    if scene_weight is not None:
        loss_config["scene_positive_weight"] = scene_weight
    metrics = train_video_epoch(
        model, [batch], torch.optim.SGD(model.parameters(), lr=0.01), None,
        torch.device("cpu"), loss_config,
        intra_transform_config={"noise_std": 0.01},
    )

    assert len(seen_labels) == 2
    assert seen_weights == [0.5 if scene_weight is None else scene_weight] * 2
    for scenes, tasks in seen_labels:
        torch.testing.assert_close(scenes, batch["scene_indices"])
        torch.testing.assert_close(tasks, batch["task_indices"])
    torch.testing.assert_close(torch.tensor(metrics["intra_loss"]), torch.stack(losses).mean())
    assert not torch.equal(before, model.projection.weight)
