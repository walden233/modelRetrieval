import torch
import torch.nn.functional as F # --- NEW ---
import kornia # --- NEW ---

# def augment_human_poses_rotation(poses_batch: torch.Tensor) -> torch.Tensor:
#     """
#     对人类姿态批次应用随机 3D 旋转。
#     输入形状: (B, S, 21, 3)
#     输出同形状: (B, S, 21, 3)
#     """
#     B, S, N, _ = poses_batch.shape
#     device = poses_batch.device
    
#     # 1. 随机轴与角
#     rand_axis = F.normalize(torch.randn(B, 3, device=device), dim=-1)     # (B, 3)
#     rand_angle = torch.rand(B, 1, device=device) * 2 * torch.pi           # (B, 1)

#     # 2. 轴角 → 旋转矩阵 (B, 3, 3)
#     rot_mats = kornia.geometry.axis_angle_to_rotation_matrix(rand_axis * rand_angle)

#     # 3. reshape 为 (B, S*N, 3)，一次性旋转
#     poses_flat = poses_batch.view(B, S*N, 3)              # (B, S*N, 3)

#     # 4. 旋转 (B, S*N, 3)
#     rotated = torch.bmm(poses_flat, rot_mats)             # (B, S*N, 3)

#     # 5. reshape 回原样
#     return rotated.view(B, S, N, 3)

# def augment_robot_tcp_rotation(tcp_batch: torch.Tensor) -> torch.Tensor:
#     """
#     对机器人 TCP 批次的位置部分应用随机 3D 旋转。
#     输入形状: (B, S, 7) = [x, y, z, qx, qy, qz, qw]
#     输出同形状
#     """
#     B, S, _ = tcp_batch.shape
#     device = tcp_batch.device
    
#     positions = tcp_batch[..., :3]     # (B, S, 3)
#     quaternions = tcp_batch[..., 3:]   # (B, S, 4)
    
#     # 1. 随机旋转
#     rand_axis = F.normalize(torch.randn(B, 3, device=device), dim=-1)     # (B, 3)
#     rand_angle = torch.rand(B, 1, device=device) * 2 * torch.pi           # (B, 1)

#     # 2. Axis-Angle → Rotation Matrix (B, 3, 3)
#     rot_mats = kornia.geometry.axis_angle_to_rotation_matrix(rand_axis * rand_angle)

#     # 3. 旋转位置 (展平为 B, S, 3)
#     positions_flat = positions.view(B, S, 3)     # (B, S, 3)

#     # 4. batch 矩阵乘法
#     rotated_positions = torch.bmm(positions_flat, rot_mats)  # (B, S, 3)

#     # 5. 合并回 (B, S, 7)
#     return torch.cat([rotated_positions, quaternions], dim=-1)



# ---------- 辅助函数 ----------
def _rand_unit_quaternion(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    在 S^3 上均匀采样 unit quaternion，返回形状 (B,4) ，格式 (x,y,z,w).
    方法：采样 4 维高斯然后归一化（等价于在球面上均匀采样）。
    """
    q = torch.randn(batch, 4, device=device, dtype=dtype)
    q = F.normalize(q, dim=1)  # unit quaternion
    # 保证标量部分为最后一位：我们使用 (x,y,z,w) 约定
    return q


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (B,4) (x,y,z,w) 转为旋转矩阵 (B,3,3)
    公式参考标准四元数->旋转矩阵转换。
    """
    # q: (B,4) as (x,y,z,w)
    x, y, z, w = q.unbind(dim=1)
    B = q.shape[0]
    # compute terms
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

    # build rotation matrix elements
    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)

    m10 = 2 * (xy + zw)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - xw)

    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = ww - xx - yy + zz

    rot = torch.stack([
        torch.stack([m00, m01, m02], dim=1),
        torch.stack([m10, m11, m12], dim=1),
        torch.stack([m20, m21, m22], dim=1),
    ], dim=1)  # (B,3,3) in a transposed stacking, we need shape (B,3,3)
    # above produced shape (3, B, 3) — fix ordering:
    rot = rot.permute(1, 0, 2).contiguous()  # (B,3,3)
    return rot


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    四元数乘法：返回 a ⊗ b
    a: (...,4) (x,y,z,w)
    b: (...,4) (x,y,z,w)
    输出 (...,4) (x,y,z,w)
    """
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)

    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    rw = aw * bw - ax * bx - ay * by - az * bz

    return torch.stack([rx, ry, rz, rw], dim=-1)


# ---------- 主函数：人体姿态旋转增强 ----------
def augment_human_poses_rotation(
    poses_batch: torch.Tensor,
    root_index: int = 0,
    rotate_about_root: bool = True,
    same_rotation_per_sample: bool = True,
) -> torch.Tensor:
    """
    对人类姿态批次应用随机 3D 旋转（增强）。
    - 输入形状: (B, S, N, 3)
    - 输出形状: (B, S, N, 3)
    说明:
      * 默认在每个样本 (batch) 上使用一个随机刚体旋转（即对时间轴 S 保持同一旋转），
        因为我们通常希望增强整个序列的一致旋转。
      * 默认围绕 root_index 关节旋转（保留 root 到其它关节的相对位移不变）。
    参数:
      - root_index: 谁作为旋转中心（默认 0）
      - rotate_about_root: True 则绕 root 旋转，False 则绕世界原点旋转
      - same_rotation_per_sample: True 每个 batch 样本使用一个旋转；False 为每个时间步采样不同旋转
    """
    assert poses_batch.ndim == 4 and poses_batch.shape[-1] == 3, "poses must be (B,S,N,3)"
    B, S, N, _ = poses_batch.shape
    device = poses_batch.device
    dtype = poses_batch.dtype

    # 1) 生成旋转四元数
    if same_rotation_per_sample:
        q_rand = _rand_unit_quaternion(B, device=device, dtype=dtype)  # (B,4)
        rotmats = _quat_to_rotmat(q_rand)  # (B,3,3)
    else:
        # 为每个时间步采样：形成 (B*S,4) -> (B,S,3,3)
        q_rs = _rand_unit_quaternion(B * S, device=device, dtype=dtype)
        rotmats = _quat_to_rotmat(q_rs).view(B, S, 3, 3)  # (B,S,3,3)

    # 2) 中心化（若绕 root 旋转）
    pts = poses_batch.view(B, S, N, 3)
    if rotate_about_root:
        # root position: (B, S, 1, 3)
        root_pos = pts[..., root_index, :].unsqueeze(2)  # (B,S,1,3)
        pts_centered = pts - root_pos  # (B,S,N,3)
    else:
        pts_centered = pts

    # 3) 应用旋转：注意矩阵乘法以列向量形式使用 R @ v
    if same_rotation_per_sample:
        # rotmats: (B,3,3) -> expand to (B,S,3,3)
        R = rotmats.unsqueeze(1).expand(-1, S, -1, -1)  # (B,S,3,3)
    else:
        R = rotmats  # (B,S,3,3)

    # 把点转成列向量形式 (B,S,3,N)
    pts_col = pts_centered.permute(0, 1, 3, 2)  # (B,S,3,N)
    # 进行矩阵乘法： (B,S,3,3) @ (B,S,3,N) -> (B,S,3,N)
    # 使用 batched bmm: 先 reshape
    Bx, Sx = B, S
    R_flat = R.reshape(Bx * Sx, 3, 3)
    pts_col_flat = pts_col.reshape(Bx * Sx, 3, N)
    rotated_col_flat = torch.bmm(R_flat, pts_col_flat)  # (B*S,3,N)
    rotated_col = rotated_col_flat.view(Bx, Sx, 3, N)
    rotated = rotated_col.permute(0, 1, 3, 2).contiguous()  # (B,S,N,3)

    # 4) 取消中心化（若有）
    if rotate_about_root:
        rotated = rotated + root_pos

    return rotated


# ---------- 主函数：机器人 TCP 旋转增强 ----------
def augment_robot_tcp_rotation(
    tcp_batch: torch.Tensor,
    same_rotation_per_sample: bool = True,
) -> torch.Tensor:
    """
    对机器人 TCP 批次的位置与朝向应用随机 3D 旋转（增强）。
    - 输入形状: (B, S, 7)  = [x, y, z, qx, qy, qz, qw]  （四元数为 x,y,z,w 约定）
    - 输出同形状, 同步旋转位置与四元数
    参数:
      - same_rotation_per_sample: True 每个 batch 样本使用同一旋转（同一序列一致性）
    """
    assert tcp_batch.ndim == 3 and tcp_batch.shape[-1] == 7, "tcp_batch must be (B,S,7)"
    B, S, _ = tcp_batch.shape
    device = tcp_batch.device
    dtype = tcp_batch.dtype

    positions = tcp_batch[..., :3]   # (B,S,3)
    quats = tcp_batch[..., 3:]       # (B,S,4)  assumed (x,y,z,w)

    # 1) 生成旋转四元数 q_rot (B,4) or (B,S,4)
    if same_rotation_per_sample:
        q_rot = _rand_unit_quaternion(B, device=device, dtype=dtype)  # (B,4)
        R = _quat_to_rotmat(q_rot)  # (B,3,3)
        # apply to positions: expand R to (B,S,3,3)
        R_exp = R.unsqueeze(1).expand(-1, S, -1, -1).reshape(B * S, 3, 3)
        pos_flat = positions.reshape(B * S, 3).unsqueeze(-1)  # (B*S,3,1)
        # do matrix multiply R @ v
        rotated_pos_flat = torch.bmm(R_exp, pos_flat)  # (B*S,3,1)
        rotated_positions = rotated_pos_flat.squeeze(-1).view(B, S, 3)
        # apply quaternion multiplication to orientations: q_new = q_rot ⊗ q_old
        q_rot_exp = q_rot.unsqueeze(1).expand(-1, S, -1)  # (B,S,4)
        new_quats = _quat_mul(q_rot_exp.reshape(B * S, 4), quats.reshape(B * S, 4))
        new_quats = new_quats.view(B, S, 4)
    else:
        # 每个时间步不同旋转
        q_rs = _rand_unit_quaternion(B * S, device=device, dtype=dtype)  # (B*S,4)
        R_flat = _quat_to_rotmat(q_rs)  # (B*S,3,3)
        pos_flat = positions.reshape(B * S, 3).unsqueeze(-1)  # (B*S,3,1)
        rotated_pos_flat = torch.bmm(R_flat, pos_flat).squeeze(-1)  # (B*S,3)
        rotated_positions = rotated_pos_flat.view(B, S, 3)
        new_quats = _quat_mul(q_rs, quats.reshape(B * S, 4)).view(B, S, 4)

    # 2) 合并并返回 (B,S,7)
    out = torch.cat([rotated_positions, new_quats], dim=-1)
    return out
