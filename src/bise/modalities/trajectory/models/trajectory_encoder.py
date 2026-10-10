import numpy as np
import torch
import torch.nn as nn

from ..geometry import robot_pose_features, validate_rigid_transform
from ..grasp import MotionShapeProjection, human_grasp_features, robot_grasp_features


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.pe[:, : inputs.size(1), :]


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        temporal_features: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.time_proj = nn.Linear(3, d_model) if temporal_features else None
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model)) if temporal_features else None
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor,
                sample_factor: int = 1, context: torch.Tensor | None = None) -> torch.Tensor:
        if sample_factor > 1:
            src = src[:, ::sample_factor, :]
            src_padding_mask = src_padding_mask[:, ::sample_factor]
            if context is not None:
                context = context[:, ::sample_factor]

        src = self.input_proj(src)
        if self.time_proj is not None:
            if context is None or context.shape != (*src.shape[:2], 3):
                raise ValueError("Time-aware trajectory encoding requires aligned time/observation context")
            observed = context[..., 2:3].bool()
            src = torch.where(observed, src, self.missing_token) + self.time_proj(context)
        if src.shape[1] + 1 > self.pos_encoder.pe.shape[1]:
            raise ValueError("Trajectory exceeds positional encoding; configure a dataset frame limit")
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=src.device)
        src_padding_mask = torch.cat((cls_mask, src_padding_mask), dim=1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=~src_padding_mask)
        return output[:, 0, :]


class CrossModalTrajectoryModel(nn.Module):
    def __init__(
        self,
        human_input_dim: int,
        robot_input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        proj_dim: int,
        dropout: float = 0.1,
        tcp_sample_factor: int = 1,
        robot_representation: str = "raw7",
        quaternion_order: str = "wxyz",
        temporal_features: bool = False,
        tcp_to_reference=None,
        shared_transformer: bool = False,
        dual_adapters: bool = False,
        grasp_geometry: bool = False,
        human_keypoint_indices: list[int] | None = None,
        use_gripper: bool = False,
    ):
        super().__init__()
        if quaternion_order not in {"wxyz", "xyzw"}:
            raise ValueError("quaternion_order must be explicit wxyz or xyzw")
        motion_dim = {"raw7": 7, "rot6d": 9, "relative_rot6d": 9}.get(robot_representation)
        if motion_dim is None:
            raise ValueError("Unknown robot_representation")
        expected_dim = 15 if grasp_geometry else motion_dim + int(use_gripper)
        if robot_input_dim != expected_dim:
            raise ValueError("robot_input_dim does not match robot_representation")
        if grasp_geometry and robot_representation != "relative_rot6d":
            raise ValueError("Grasp geometry requires relative_rot6d")
        if grasp_geometry and tcp_to_reference is not None:
            raise ValueError("Grasp inputs are already calibrated; do not apply a second tool transform")
        self.grasp_geometry = grasp_geometry
        self.use_gripper = use_gripper or grasp_geometry
        self.human_keypoint_indices = human_keypoint_indices
        if human_keypoint_indices is not None:
            if (not human_keypoint_indices
                    or len(set(human_keypoint_indices)) != len(human_keypoint_indices)
                    or any(i < 0 or i >= 21 for i in human_keypoint_indices)
                    or human_input_dim != 3 * len(human_keypoint_indices) + 9 * grasp_geometry):
                raise ValueError("Invalid human keypoint selection or input dimension")
            if grasp_geometry and not {0, 4, 5, 8, 9, 17}.issubset(human_keypoint_indices):
                raise ValueError("Grasp keypoint selection must include wrist, palm and pinch anchors")
        elif grasp_geometry and human_input_dim != 72:
            raise ValueError("Full-hand grasp geometry requires human_input_dim=72")
        self.robot_representation = robot_representation
        self.quaternion_order = quaternion_order
        self.register_buffer("tcp_to_reference", validate_rigid_transform(tcp_to_reference)
                             if tcp_to_reference is not None else None, persistent=False)
        if self.tcp_to_reference is not None and robot_representation == "raw7":
            raise ValueError("Tool correction requires rot6d or relative_rot6d")
        self.human_encoder = TrajectoryEncoder(
            input_dim=human_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            temporal_features=temporal_features,
        )
        self.robot_encoder = TrajectoryEncoder(
            input_dim=robot_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            temporal_features=temporal_features,
        )
        if grasp_geometry:
            self.human_encoder.input_proj = MotionShapeProjection(human_input_dim - 9, d_model)
            self.robot_encoder.input_proj = MotionShapeProjection(6, d_model)
        if shared_transformer:
            for name in ("transformer_encoder", "pos_encoder", "cls_token",
                         "time_proj", "missing_token"):
                setattr(self.robot_encoder, name, getattr(self.human_encoder, name))
        self.dual_adapters = dual_adapters
        if dual_adapters:
            def adapter():
                return nn.Sequential(nn.Linear(d_model, max(1, d_model // 2)), nn.ReLU(),
                                     nn.Linear(max(1, d_model // 2), d_model))
            self.human_adapter = adapter()
            self.robot_adapter = adapter()
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )
        self.logit_scale_inter = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        self.logit_scale_intra = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        self.eps = 1e-8
        self.tcp_sample_factor = tcp_sample_factor

    def safe_l2norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp(min=self.eps)

    def forward_human(self, human_poses: torch.Tensor, human_mask: torch.Tensor,
                      context: torch.Tensor | None = None) -> torch.Tensor:
        if self.human_keypoint_indices is not None and not self.grasp_geometry:
            if human_poses.shape[-2] != 21:
                raise ValueError("Keypoint selection requires the common 21-point source")
            human_poses = human_poses[..., self.human_keypoint_indices, :]
        human_poses_flat = (human_grasp_features(human_poses, human_mask, context,
                                                self.human_keypoint_indices)
                            if self.grasp_geometry else
                            human_poses.reshape(human_poses.size(0), human_poses.size(1), -1))
        human_features = self.human_encoder(human_poses_flat, human_mask, context=context)
        if self.dual_adapters:
            human_features = human_features + self.human_adapter(human_features)
        return self.safe_l2norm(self.projection_head(human_features))

    def forward_robot(self, tcp_bases: torch.Tensor, tcp_mask: torch.Tensor,
                      context: torch.Tensor | None = None) -> torch.Tensor:
        if self.grasp_geometry:
            features = robot_grasp_features(tcp_bases, tcp_mask, self.quaternion_order)
        else:
            if self.use_gripper and tcp_bases.shape[-1] != 8:
                raise ValueError("Gripper input requires TCP pose plus measured opening (8 values)")
            pose = tcp_bases[..., :7] if self.use_gripper else tcp_bases
            features = robot_pose_features(pose, tcp_mask, self.robot_representation,
                                           self.quaternion_order, self.tcp_to_reference)
            if self.use_gripper:
                features = torch.cat((features, tcp_bases[..., 7:8]), dim=-1)
                features = features.masked_fill(~tcp_mask[..., None], 0)
        robot_features = self.robot_encoder(features, tcp_mask,
                                            sample_factor=self.tcp_sample_factor, context=context)
        if self.dual_adapters:
            robot_features = robot_features + self.robot_adapter(robot_features)
        return self.safe_l2norm(self.projection_head(robot_features))

    def forward(
        self,
        human_poses: torch.Tensor,
        human_mask: torch.Tensor,
        tcp_bases: torch.Tensor,
        tcp_mask: torch.Tensor,
        human_context: torch.Tensor | None = None,
        robot_context: torch.Tensor | None = None,
    ):
        human_embeddings = self.forward_human(human_poses, human_mask, human_context)
        robot_embeddings = self.forward_robot(tcp_bases, tcp_mask, robot_context)
        return human_embeddings, robot_embeddings, self.logit_scale_inter.exp()
