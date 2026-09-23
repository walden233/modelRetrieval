"""Explicit quaternion conventions and shared robot pose representations."""

import torch


def quaternion_to_matrix(quaternion: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    if order not in {"wxyz", "xyzw"}:
        raise ValueError(f"Unsupported quaternion order: {order}")
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if order == "xyzw":
        q = q[..., [3, 0, 1, 2]]
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def robot_pose_features(
    tcp: torch.Tensor,
    mask: torch.Tensor,
    representation: str = "raw7",
    quaternion_order: str = "wxyz",
    tcp_to_reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if representation == "raw7":
        if tcp_to_reference is not None:
            raise ValueError("TCP-to-reference requires a rotation-matrix representation")
        return tcp
    if representation not in {"rot6d", "relative_rot6d"}:
        raise ValueError(f"Unknown robot representation: {representation}")
    rotation = quaternion_to_matrix(tcp[..., 3:], quaternion_order)
    position = tcp[..., :3]
    if tcp_to_reference is not None:
        # The offset lives in the moving TCP frame, not in the base frame.
        position = position + (rotation @ tcp_to_reference[:3, 3, None]).squeeze(-1)
        rotation = rotation @ tcp_to_reference[:3, :3]
    if representation == "relative_rot6d":
        first = mask.long().argmax(dim=1)
        rows = torch.arange(tcp.shape[0], device=tcp.device)
        r0_inv = rotation[rows, first].transpose(-1, -2).unsqueeze(1)
        p0 = position[rows, first].unsqueeze(1)
        position = (r0_inv @ (position - p0).unsqueeze(-1)).squeeze(-1)
        rotation = r0_inv @ rotation
    rot6d = rotation[..., :, :2].transpose(-1, -2).reshape(*tcp.shape[:-1], 6)
    result = torch.cat([position, rot6d], dim=-1)
    return result.masked_fill(~mask.unsqueeze(-1), 0)


def validate_rigid_transform(matrix) -> torch.Tensor:
    transform = torch.as_tensor(matrix, dtype=torch.float32)
    if transform.shape != (4, 4) or not torch.isfinite(transform).all():
        raise ValueError("TCP-to-reference must be a finite 4x4 rigid transform")
    rotation = transform[:3, :3]
    if (not torch.allclose(transform[3], torch.tensor([0., 0., 0., 1.]), atol=1e-5)
            or not torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)
            or not torch.isclose(torch.det(rotation), torch.tensor(1.), atol=1e-5)):
        raise ValueError("TCP-to-reference must contain a proper rotation and homogeneous last row")
    return transform


def transform_tcp(tcp: torch.Tensor, rotation_quaternion: torch.Tensor,
                  translation: torch.Tensor, order: str = "wxyz") -> torch.Tensor:
    """Apply one global rigid transform, with an explicitly wxyz rotation argument."""
    result = tcp.clone()
    rotation = quaternion_to_matrix(rotation_quaternion, "wxyz")
    result[..., :3] = tcp[..., :3] @ rotation.T + translation
    b = tcp[..., 3:] if order == "wxyz" else tcp[..., [6, 3, 4, 5]]
    aw, ax, ay, az = rotation_quaternion.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    q = torch.stack([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw], dim=-1)
    if order == "xyzw":
        q = q[..., [1, 2, 3, 0]]
    elif order != "wxyz":
        raise ValueError("Unknown quaternion order")
    result[..., 3:] = q
    return result
