import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)  # 类似 BERT 初始化


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
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.)
        
        # CLS 令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, src, src_padding_mask, sample_factor=1):
            """
            Args:
                src (torch.Tensor): (batch_size, seq_len, input_dim)
                src_padding_mask (torch.Tensor): (batch_size, seq_len)
                sample_factor (int, optional): 采样因子. 
                    =1 表示不采样. 
                    =2 表示采样 1/2 (即每隔一个取一个). 
                    =3 表示采样 1/3 (即每隔两个取一个).
                    Defaults to 1.
            """
            # src shape: (batch_size, seq_len, input_dim)
            # src_padding_mask shape: (batch_size, seq_len)
            
            # 0. (新增) 对序列进行降采样
            if sample_factor > 1:
                # 使用步长(stride)对序列和掩码进行采样
                # 必须确保两者以完全相同的方式被采样
                src = src[:, ::sample_factor, :]
                src_padding_mask = src_padding_mask[:, ::sample_factor]
                
                # 采样后 src shape: (batch_size, new_seq_len, input_dim)
                # 采样后 mask shape: (batch_size, new_seq_len)

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
            if((src_padding_mask.sum(dim=1)==0).any()):
                print("有全掩")
            if((src_padding_mask.sum(dim=1)==1).any()):
                print("有1")
            
            src_padding_mask=~src_padding_mask
            output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)
            
            # 6. 提取 CLS 令牌的输出作为序列表示
            cls_output = output[:, 0, :]
            if torch.isinf(cls_output).any():
                print("!!! 警告：TrajectoryEncoder输出中存在 inf !!!")
            if torch.isnan(cls_output).any():
                nan_mask = torch.isnan(cls_output)
                batch_idx_nan = torch.nonzero(nan_mask.any(dim=1)).squeeze().tolist()
                print(f"!!! 警告：TrajectoryEncoder输出中存在 nan，样本索引: {batch_idx_nan}")
            return cls_output

class CrossModalTrajectoryModel(nn.Module):
    """
    (已修改) 完整的跨模态轨迹匹配模型。
    添加了独立的 'forward_human' 和 'forward_robot' 方法，用于模态内训练。
    """
    def __init__(self, human_input_dim, robot_input_dim, d_model, nhead, num_layers, dim_feedforward, proj_dim, dropout=0.1, tcp_sample_factor=1):
        super().__init__()
        
        # 编码器 (不变)
        self.human_encoder = TrajectoryEncoder(
            input_dim=human_input_dim, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.robot_encoder = TrajectoryEncoder(
            input_dim=robot_input_dim, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, dim_feedforward=dim_feedforward, dropout=dropout
        )
        
        # 投射头 (不变)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )
        
        # 用于 Inter-modal (跨模态)
        self.logit_scale_inter = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        # 用于 Intra-modal (模态内)
        self.logit_scale_intra = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))
        
        # L2 Norm 辅助函数 (不变)
        self.eps = 1e-8
        self.tcp_sample_factor = tcp_sample_factor

    def safe_l2norm(self, x):
        return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=self.eps))

    # --- NEW: forward_human ---
    def forward_human(self, human_poses, human_mask):
        """
        处理人类轨迹：展平 -> 编码 -> 投影 -> 归一化
        """
        if torch.isinf(human_poses).any():
            print("!!! 警告：human_poses 输入中存在 inf !!!")
        if torch.isnan(human_poses).any():
            print("!!! 警告：human_poses 输入中存在 nan !!!")

        # 1. 展平人类姿态数据
        # (batch_size, seq_len, 21, 3) -> (batch_size, seq_len, 63)
        human_poses_flat = human_poses.view(human_poses.size(0), human_poses.size(1), -1)
        
        # 2. 编码
        human_features = self.human_encoder(human_poses_flat, human_mask)
        
        # 3. 通过投射头
        human_embeddings = self.projection_head(human_features)
        
        # 4. L2 归一化
        return self.safe_l2norm(human_embeddings)

    # --- NEW: forward_robot ---
    def forward_robot(self, tcp_bases, tcp_mask):
        """
        处理机器人轨迹：(采样) -> 编码 -> 投影 -> 归一化
        """
        if torch.isinf(tcp_bases).any():
            print("!!! 警告：tcp_bases 输入中存在 inf !!!")
        if torch.isnan(tcp_bases).any():
            print("!!! 警告：tcp_bases 输入中存在 nan !!!")

        # 1. 编码 (采样在编码器内部完成)
        robot_features = self.robot_encoder(tcp_bases, tcp_mask, sample_factor=self.tcp_sample_factor)
        
        # 2. 通过投射头
        robot_embeddings = self.projection_head(robot_features)
        
        # 3. L2 归一化
        return self.safe_l2norm(robot_embeddings)

    # --- MODIFIED: forward (now uses sub-methods) ---
    def forward(self, human_poses, human_mask, tcp_bases, tcp_mask):
        """
        标准的前向传播，用于跨模态损失计算和评估。
        """
        human_embeddings = self.forward_human(human_poses, human_mask)
        robot_embeddings = self.forward_robot(tcp_bases, tcp_mask)
        
        return human_embeddings, robot_embeddings, self.logit_scale_inter.exp()