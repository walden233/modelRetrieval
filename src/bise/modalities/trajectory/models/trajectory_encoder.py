import numpy as np
import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
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

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor, sample_factor: int = 1) -> torch.Tensor:
        if sample_factor > 1:
            src = src[:, ::sample_factor, :]
            src_padding_mask = src_padding_mask[:, ::sample_factor]

        src = self.input_proj(src)
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
    ):
        super().__init__()
        self.human_encoder = TrajectoryEncoder(
            input_dim=human_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.robot_encoder = TrajectoryEncoder(
            input_dim=robot_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
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

    def forward_human(self, human_poses: torch.Tensor, human_mask: torch.Tensor) -> torch.Tensor:
        human_poses_flat = human_poses.view(human_poses.size(0), human_poses.size(1), -1)
        human_features = self.human_encoder(human_poses_flat, human_mask)
        return self.safe_l2norm(self.projection_head(human_features))

    def forward_robot(self, tcp_bases: torch.Tensor, tcp_mask: torch.Tensor) -> torch.Tensor:
        robot_features = self.robot_encoder(tcp_bases, tcp_mask, sample_factor=self.tcp_sample_factor)
        return self.safe_l2norm(self.projection_head(robot_features))

    def forward(
        self,
        human_poses: torch.Tensor,
        human_mask: torch.Tensor,
        tcp_bases: torch.Tensor,
        tcp_mask: torch.Tensor,
    ):
        human_embeddings = self.forward_human(human_poses, human_mask)
        robot_embeddings = self.forward_robot(tcp_bases, tcp_mask)
        return human_embeddings, robot_embeddings, self.logit_scale_inter.exp()
