"""Functional grasp motion and local shape, computed before temporal encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import robot_pose_features


class MotionShapeProjection(nn.Module):
    def __init__(self, shape_dim: int, d_model: int):
        super().__init__()
        self.motion = nn.Sequential(nn.Linear(9, d_model), nn.GELU())
        self.shape = nn.Sequential(nn.Linear(shape_dim, d_model), nn.GELU())
        for branch in (self.motion, self.shape):
            nn.init.xavier_uniform_(branch[0].weight)
            nn.init.zeros_(branch[0].bias)

    def forward(self, features):
        return (self.motion(features[..., :9]) + self.shape(features[..., 9:])) / 2**0.5


def human_grasp_features(points, mask, context=None, keypoint_indices=None):
    """Keep grasp motion; express selected hand points in the current grasp frame."""
    if points.shape[-2:] != (21, 3):
        raise ValueError("Grasp geometry requires all 21 human landmarks")
    observed = mask if context is None else mask & context[..., 2].bool()
    center = (points[..., 4, :] + points[..., 8, :]) / 2
    closing = points[..., 8, :] - points[..., 4, :]
    palm_width = points[..., 17, :] - points[..., 5, :]
    closing = torch.where(closing.norm(dim=-1, keepdim=True) > 1e-4, closing, palm_width)
    x = F.normalize(closing, dim=-1, eps=1e-6)
    forward = points[..., 9, :] - points[..., 0, :]
    y = F.normalize(forward - (forward * x).sum(-1, keepdim=True) * x,
                    dim=-1, eps=1e-6)
    z = F.normalize(torch.cross(x, y, dim=-1), dim=-1, eps=1e-6)
    y = torch.cross(z, x, dim=-1)
    rotation = torch.stack((x, y, z), dim=-1)
    rot6d = rotation[..., :, :2].transpose(-1, -2).flatten(-2)
    length = forward.norm(dim=-1).masked_fill(~observed, float("nan"))
    scale = torch.nanmedian(length, dim=1).values.clamp_min(1e-4)[:, None, None, None]
    shape_points = points if keypoint_indices is None else points[..., keypoint_indices, :]
    local = (shape_points - center.unsqueeze(-2)) @ rotation / scale
    features = torch.cat((center, rot6d, local.flatten(-2)), dim=-1)
    return features.masked_fill(~observed.unsqueeze(-1), 0)


def robot_grasp_features(tcp, mask, quaternion_order="wxyz"):
    """Calibrated grasp pose and two jaw points, scaled by maximum opening."""
    if tcp.shape[-1] != 8:
        raise ValueError("Grasp input requires calibrated XYZ/quaternion and measured opening")
    motion = robot_pose_features(tcp[..., :7], mask, "relative_rot6d", quaternion_order)
    half_width = tcp[..., 7] / 2
    zero = torch.zeros_like(half_width)
    shape = torch.stack((-half_width, zero, zero, half_width, zero, zero), dim=-1)
    return torch.cat((motion, shape), dim=-1).masked_fill(~mask.unsqueeze(-1), 0)
