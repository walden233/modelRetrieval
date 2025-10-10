import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class TrajectoryEncoder(nn.Module):
    """
    用于编码轨迹序列的 Transformer 模型。
    """
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 输入投射层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # CLS 令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, src, src_padding_mask):
        # src shape: (batch_size, seq_len, input_dim)
        # src_padding_mask shape: (batch_size, seq_len)
        
        # 1. 投射到 d_model 维度
        src = self.input_proj(src)
        
        # 2. 添加 CLS 令牌
        batch_size = src.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        src = torch.cat((cls_tokens, src), dim=1)
        
        # 3. 调整 padding mask 以适应 CLS 令牌
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=src.device)
        src_padding_mask = torch.cat((cls_mask, src_padding_mask), dim=1)
        
        # 4. 添加位置编码
        src = self.pos_encoder(src)
        
        # 5. 通过 Transformer 编码器
        # 注意：PyTorch 的 TransformerEncoderLayer 需要的 padding mask 是布尔类型，
        # 其中 True 表示该位置 *不* 被-attention，所以我们需要反转掩码
        output = self.transformer_encoder(src, src_key_padding_mask=~src_padding_mask)
        
        # 6. 提取 CLS 令牌的输出作为序列表示
        cls_output = output[:, 0, :]
        return cls_output

class CrossModalTrajectoryModel(nn.Module):
    """
    完整的跨模态轨迹匹配模型，采用双编码器结构和共享权重。
    """
    def __init__(self, human_input_dim, robot_input_dim, d_model, nhead, num_layers, dim_feedforward, proj_dim, dropout=0.1):
        super().__init__()
        
        # 共享的 Transformer 编码器
        self.human_encoder = TrajectoryEncoder(
            input_dim=human_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.robot_encoder = TrajectoryEncoder(
            input_dim=robot_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 投射头，用于对比学习
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))

    def forward(self, human_poses, human_mask, tcp_bases, tcp_mask):
        # 展平人类姿态数据
        human_poses_flat = human_poses.view(human_poses.size(0), human_poses.size(1), -1)
        
        # 编码
        human_features = self.human_encoder(human_poses_flat, human_mask)
        robot_features = self.robot_encoder(tcp_bases, tcp_mask)
        
        # 通过投射头
        human_embeddings = self.projection_head(human_features)
        robot_embeddings = self.projection_head(robot_features)
        
        # L2 归一化
        human_embeddings = F.normalize(human_embeddings, p=2, dim=-1)
        robot_embeddings = F.normalize(robot_embeddings, p=2, dim=-1)
        
        return human_embeddings, robot_embeddings, self.logit_scale.exp()