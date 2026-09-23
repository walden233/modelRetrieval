import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bise.modalities.trajectory.geometry import (
    quaternion_to_matrix, robot_pose_features, transform_tcp, validate_rigid_transform,
)
from bise.modalities.trajectory.augmentations import augment_robot_tcp_rotation
from bise.modalities.trajectory.models import CrossModalTrajectoryModel


def poses():
    torch.manual_seed(3)
    tcp = torch.randn(2, 7, 7)
    tcp[..., 3:] /= tcp[..., 3:].norm(dim=-1, keepdim=True)
    return tcp, torch.ones(2, 7, dtype=torch.bool)


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_sign_and_global_rigid_transform_invariance(order):
    tcp, mask = poses()
    if order == "xyzw":
        tcp[..., 3:] = tcp[..., [4, 5, 6, 3]]
    flipped = tcp.clone()
    flipped[..., 3:] *= torch.tensor([1., -1., -1., 1., -1., 1., -1.])[None, :, None]
    base = robot_pose_features(tcp, mask, "rot6d", order)
    torch.testing.assert_close(base, robot_pose_features(flipped, mask, "rot6d", order))
    q = torch.tensor([.5, .5, .5, .5])
    changed = transform_tcp(tcp, q, torch.tensor([.7, -.4, .2]), order)
    torch.testing.assert_close(robot_pose_features(tcp, mask, "relative_rot6d", order),
                               robot_pose_features(changed, mask, "relative_rot6d", order),
                               atol=2e-6, rtol=2e-6)


def test_z90_augmentation_uses_same_rotation_for_pose_and_position():
    tcp = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.], [1., 0., 0., 1., 0., 0., 0.]]])
    rotation = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    q_xyzw = torch.tensor([[0., 0., 2**-.5, 2**-.5]])
    with patch("bise.modalities.trajectory.augmentations._rand_z_rotation",
               return_value=(rotation, q_xyzw)):
        result = augment_robot_tcp_rotation(tcp, noise_std=0, quaternion_order="wxyz")
    torch.testing.assert_close(result[0, 1, :3], torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(quaternion_to_matrix(result[0, 0, 3:]), rotation[0])


def test_tool_offset_rotates_with_tcp_and_precedes_relative_pose():
    tcp = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 2**-.5, 0., 0., 2**-.5]]])
    tool = torch.eye(4)
    tool[0, 3] = .1
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = robot_pose_features(tcp, mask, "rot6d", tcp_to_reference=tool)
    torch.testing.assert_close(result[0, :, :3], torch.tensor([[.1, 0., 0.], [0., .1, 0.]]))
    relative = robot_pose_features(tcp, mask, "relative_rot6d", tcp_to_reference=tool)
    torch.testing.assert_close(relative[0, 1, :3], torch.tensor([-.1, .1, 0.]))
    tool[0, 0] = 2
    with pytest.raises(ValueError, match="proper rotation"):
        validate_rigid_transform(tool)


def test_relative_encoder_invariance_and_padding():
    tcp, mask = poses()
    mask[1, 4:] = False
    model = CrossModalTrajectoryModel(18, 9, 16, 4, 1, 32, 8, dropout=0,
                                      robot_representation="relative_rot6d").eval()
    changed = transform_tcp(tcp, torch.tensor([.5, .5, .5, .5]), torch.tensor([1., 2., 3.]))
    changed[1, 4:] = 123
    with torch.no_grad():
        torch.testing.assert_close(model.forward_robot(tcp, mask), model.forward_robot(changed, mask),
                                   atol=3e-6, rtol=3e-6)
