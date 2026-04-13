import torch
import torch.nn.functional as F


def _rand_unit_quaternion(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    quaternion = torch.randn(batch, 4, device=device, dtype=dtype)
    return F.normalize(quaternion, dim=1)


def _quat_to_rotmat(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quaternion.unbind(dim=1)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)
    m10 = 2 * (xy + zw)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - xw)
    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = ww - xx - yy + zz

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=1),
            torch.stack([m10, m11, m12], dim=1),
            torch.stack([m20, m21, m22], dim=1),
        ],
        dim=1,
    ).permute(1, 0, 2).contiguous()


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
    same_rotation_per_sample: bool = True,
) -> torch.Tensor:
    batch_size, seq_len, num_keypoints, _ = poses_batch.shape
    device = poses_batch.device
    dtype = poses_batch.dtype

    if same_rotation_per_sample:
        rotmats = _quat_to_rotmat(_rand_unit_quaternion(batch_size, device=device, dtype=dtype))
    else:
        rotmats = _quat_to_rotmat(_rand_unit_quaternion(batch_size * seq_len, device=device, dtype=dtype)).view(
            batch_size, seq_len, 3, 3
        )

    points = poses_batch.view(batch_size, seq_len, num_keypoints, 3)
    if rotate_about_root:
        root_pos = points[..., root_index, :].unsqueeze(2)
        centered = points - root_pos
    else:
        root_pos = None
        centered = points

    rotation = rotmats.unsqueeze(1).expand(-1, seq_len, -1, -1) if same_rotation_per_sample else rotmats
    rotation_flat = rotation.reshape(batch_size * seq_len, 3, 3)
    centered_flat = centered.permute(0, 1, 3, 2).reshape(batch_size * seq_len, 3, num_keypoints)
    rotated_flat = torch.bmm(rotation_flat, centered_flat)
    rotated = rotated_flat.view(batch_size, seq_len, 3, num_keypoints).permute(0, 1, 3, 2).contiguous()

    if root_pos is not None:
        rotated = rotated + root_pos
    return rotated


def augment_robot_tcp_rotation(tcp_batch: torch.Tensor, same_rotation_per_sample: bool = True) -> torch.Tensor:
    batch_size, seq_len, _ = tcp_batch.shape
    device = tcp_batch.device
    dtype = tcp_batch.dtype
    positions = tcp_batch[..., :3]
    quaternions = tcp_batch[..., 3:]

    if same_rotation_per_sample:
        q_rot = _rand_unit_quaternion(batch_size, device=device, dtype=dtype)
        rotation = _quat_to_rotmat(q_rot)
        expanded_rotation = rotation.unsqueeze(1).expand(-1, seq_len, -1, -1).reshape(batch_size * seq_len, 3, 3)
        pos_flat = positions.reshape(batch_size * seq_len, 3).unsqueeze(-1)
        rotated_positions = torch.bmm(expanded_rotation, pos_flat).squeeze(-1).view(batch_size, seq_len, 3)
        q_rot_expanded = q_rot.unsqueeze(1).expand(-1, seq_len, -1)
        new_quaternions = _quat_mul(q_rot_expanded.reshape(batch_size * seq_len, 4), quaternions.reshape(batch_size * seq_len, 4)).view(
            batch_size, seq_len, 4
        )
    else:
        q_rot = _rand_unit_quaternion(batch_size * seq_len, device=device, dtype=dtype)
        rotation = _quat_to_rotmat(q_rot)
        pos_flat = positions.reshape(batch_size * seq_len, 3).unsqueeze(-1)
        rotated_positions = torch.bmm(rotation, pos_flat).squeeze(-1).view(batch_size, seq_len, 3)
        new_quaternions = _quat_mul(q_rot, quaternions.reshape(batch_size * seq_len, 4)).view(batch_size, seq_len, 4)

    return torch.cat([rotated_positions, new_quaternions], dim=-1)
