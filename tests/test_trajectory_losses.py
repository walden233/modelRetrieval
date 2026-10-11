import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.trajectory.augmentations import augment_human_poses_rotation, augment_robot_tcp_rotation
from bise.modalities.trajectory.losses import multi_positive_intra_modal_loss
from bise.modalities.video.losses import intra_domain_consistency_loss


@pytest.mark.parametrize("loss_fn", [multi_positive_intra_modal_loss, intra_domain_consistency_loss])
def test_intra_loss_uses_weighted_scene_task_targets_and_gradients(loss_fn):
    scenes = torch.tensor([0, 0, 1, 2])
    tasks = torch.tensor([0, 0, 0, 1])
    weights = torch.tensor([[1.0, 0.5, 0.25, 0.0],
                            [0.5, 1.0, 0.25, 0.0],
                            [0.25, 0.25, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    targets = weights / weights.sum(dim=1, keepdim=True)
    logits = torch.tensor([[1.0, 0.8, 0.2, -0.1],
                           [0.6, 1.2, 0.1, 0.0],
                           [0.3, 0.2, 0.9, -0.2],
                           [-0.1, 0.0, -0.3, 1.0]], requires_grad=True)
    loss = loss_fn(torch.eye(4), logits.t(), torch.tensor(1.0), scenes, tasks)
    loss.backward()

    expected_gradient = ((logits.detach().softmax(1) - targets)
                         + (logits.detach().t().softmax(1) - targets).t()) / 8
    torch.testing.assert_close(logits.grad, expected_gradient)
    entropy_term = (targets * targets.clamp_min(1e-12).log()).sum() / 4
    expected_loss = -(targets * (F.log_softmax(logits, 1)
                                + F.log_softmax(logits.t(), 1))).sum() / 8 + entropy_term
    torch.testing.assert_close(loss, expected_loss)


@pytest.mark.parametrize("loss_fn", [multi_positive_intra_modal_loss, intra_domain_consistency_loss])
def test_intra_loss_with_distinct_tasks_preserves_diagonal_contrastive_loss(loss_fn):
    z1 = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    z2 = torch.tensor([[0.8, 0.6], [0.0, 1.0]])
    labels = torch.arange(2)
    loss = loss_fn(z1, z2, torch.tensor(3.0), labels, labels)
    logits = 3.0 * (z1 @ z2.t())
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    torch.testing.assert_close(loss, expected)


def test_multi_positive_intra_modal_loss_rewards_same_scene_matches():
    z1 = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        dim=1,
    )
    z2 = z1.clone()
    logit_scale = torch.tensor(10.0)

    same_scene_loss = multi_positive_intra_modal_loss(
        z1,
        z2,
        logit_scale,
        scene_labels=torch.tensor([0, 0, 1]),
        task_labels=torch.tensor([0, 0, 1]),
    )
    distinct_scene_loss = multi_positive_intra_modal_loss(
        z1,
        z2,
        logit_scale,
        scene_labels=torch.tensor([0, 1, 2]),
        task_labels=torch.tensor([0, 0, 1]),
    )

    assert same_scene_loss < distinct_scene_loss


def test_multi_positive_intra_modal_loss_supports_weak_task_positives():
    z1 = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        dim=1,
    )
    z2 = z1.clone()
    logit_scale = torch.tensor(10.0)
    strict_loss = multi_positive_intra_modal_loss(
        z1,
        z2,
        logit_scale,
        scene_labels=torch.tensor([0, 1, 2]),
        task_labels=torch.tensor([0, 0, 1]),
        task_positive_weight=0.0,
    )
    weak_task_loss = multi_positive_intra_modal_loss(
        z1,
        z2,
        logit_scale,
        scene_labels=torch.tensor([0, 1, 2]),
        task_labels=torch.tensor([0, 0, 1]),
        task_positive_weight=0.25,
    )

    assert weak_task_loss < strict_loss


def test_human_pose_augmentation_preserves_root_and_z_without_noise():
    torch.manual_seed(0)
    poses = torch.tensor(
        [
            [
                [[1.0, 2.0, 0.5], [2.0, 3.0, 1.5]],
                [[1.5, 2.5, 0.7], [2.5, 3.5, 1.7]],
            ]
        ]
    )
    augmented = augment_human_poses_rotation(poses, noise_std=0.0, max_angle_degrees=10.0)

    assert torch.allclose(augmented[:, :, 0, :], poses[:, :, 0, :], atol=1e-6)
    assert torch.allclose(augmented[..., 2], poses[..., 2], atol=1e-6)


def test_robot_tcp_augmentation_keeps_quaternion_normalized():
    torch.manual_seed(0)
    tcp = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [0.2, 0.1, 0.4, 0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    augmented = augment_robot_tcp_rotation(tcp, noise_std=0.0, max_angle_degrees=10.0)
    quat_norms = torch.norm(augmented[..., 3:], dim=-1)

    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-6)
    assert torch.allclose(augmented[..., 2], tcp[..., 2], atol=1e-6)
