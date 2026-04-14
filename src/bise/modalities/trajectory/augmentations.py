import math

import torch
import torch.nn.functional as F


def _rand_z_rotation(batch: int, device: torch.device, dtype: torch.dtype, max_angle_degrees: float) -> tuple[torch.Tensor, torch.Tensor]:
    if max_angle_degrees <= 0:
        rotmats = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1).clone()
        quaternions = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).unsqueeze(0).expand(batch, -1).clone()
        return rotmats, quaternions

    max_angle_radians = math.radians(max_angle_degrees)
    angles = (torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0) * max_angle_radians
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)

    zeros = torch.zeros_like(cos_theta)
    ones = torch.ones_like(cos_theta)
    rotmats = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, zeros], dim=1),
            torch.stack([sin_theta, cos_theta, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )

    half_angles = angles / 2.0
    quaternions = torch.stack(
        [
            zeros,
            zeros,
            torch.sin(half_angles),
            torch.cos(half_angles),
        ],
        dim=1,
    )
    return rotmats, quaternions


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    rw = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([rx, ry, rz, rw], dim=-1)


def augment_human_poses_rotation(
    poses_batch: torch.Tensor,
    root_index: int = 0,
    rotate_about_root: bool = True,
    noise_std: float = 0.005,
    max_angle_degrees: float = 10.0,
) -> torch.Tensor:
    batch_size, seq_len, num_keypoints, _ = poses_batch.shape
    device = poses_batch.device
    dtype = poses_batch.dtype

    rotmats, _ = _rand_z_rotation(batch_size, device=device, dtype=dtype, max_angle_degrees=max_angle_degrees)
    rotation = rotmats.unsqueeze(1).expand(-1, seq_len, -1, -1)

    points = poses_batch.view(batch_size, seq_len, num_keypoints, 3)
    if rotate_about_root:
        root_pos = points[..., root_index, :].unsqueeze(2)
        centered = points - root_pos
    else:
        root_pos = None
        centered = points

    rotation_flat = rotation.reshape(batch_size * seq_len, 3, 3)
    centered_flat = centered.permute(0, 1, 3, 2).reshape(batch_size * seq_len, 3, num_keypoints)
    rotated_flat = torch.bmm(rotation_flat, centered_flat)
    rotated = rotated_flat.view(batch_size, seq_len, 3, num_keypoints).permute(0, 1, 3, 2).contiguous()

    if root_pos is not None:
        rotated = rotated + root_pos
    if noise_std > 0:
        rotated = rotated + torch.randn_like(rotated) * noise_std
    return rotated


def augment_robot_tcp_rotation(
    tcp_batch: torch.Tensor,
    noise_std: float = 0.005,
    max_angle_degrees: float = 10.0,
    rotate_about_first_position: bool = True,
) -> torch.Tensor:
    batch_size, seq_len, _ = tcp_batch.shape
    device = tcp_batch.device
    dtype = tcp_batch.dtype
    positions = tcp_batch[..., :3]
    quaternions = tcp_batch[..., 3:]

    rotation, q_rot = _rand_z_rotation(batch_size, device=device, dtype=dtype, max_angle_degrees=max_angle_degrees)
    expanded_rotation = rotation.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(batch_size * seq_len, 3, 3)

    if rotate_about_first_position:
        origin = positions[:, :1, :]
        centered_positions = positions - origin
    else:
        origin = None
        centered_positions = positions

    pos_flat = centered_positions.reshape(batch_size * seq_len, 3).unsqueeze(-1)
    rotated_positions = torch.bmm(expanded_rotation, pos_flat).squeeze(-1).view(batch_size, seq_len, 3)
    if origin is not None:
        rotated_positions = rotated_positions + origin
    if noise_std > 0:
        rotated_positions = rotated_positions + torch.randn_like(rotated_positions) * noise_std

    q_rot_expanded = q_rot.unsqueeze(1).expand(-1, seq_len, -1)
    new_quaternions = _quat_mul(
        q_rot_expanded.reshape(batch_size * seq_len, 4),
        quaternions.reshape(batch_size * seq_len, 4),
    ).view(batch_size, seq_len, 4)
    new_quaternions = F.normalize(new_quaternions, dim=-1)

    return torch.cat([rotated_positions, new_quaternions], dim=-1)
