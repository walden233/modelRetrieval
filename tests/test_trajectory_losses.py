import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.trajectory.augmentations import augment_human_poses_rotation, augment_robot_tcp_rotation
from bise.modalities.trajectory.losses import multi_positive_intra_modal_loss


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
