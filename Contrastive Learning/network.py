import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class ContrastiveFineTuner(nn.Module):
    def __init__(self, model, hidden_size=1024,feature_dim=128):
        super().__init__()
        self.model = model
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )

    def forward(self, pixel_values):
        # # 调整维度以匹配模型期望的输入格式 (B, C, T, H, W) -> (B, T, C, H, W)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        video_features = self.model(pixel_values)
        
        # 将特征通过投影头
        projection = self.projection_head(video_features)
        
        # L2 归一化，这对于对比损失至关重要
        projection = nn.functional.normalize(projection, p=2, dim=1)
        
        return projection
    


class vjepaFineTuner(nn.Module):
    def __init__(self, model, hidden_size=1024,feature_dim=128):
        super().__init__()
        self.model = model
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )

    def forward(self, pixel_values):
        video_features = self.model.get_vision_features(pixel_values)

        # --- 新增代码：在序列维度上进行平均池化 ---
        # torch.mean(video_features, dim=1) 会将 [b, 2048, 1024] -> [b, 1024]
        aggregated_features = torch.mean(video_features, dim=1)
        
        # 将聚合后的特征通过投影头
        projection = self.projection_head(aggregated_features)

        # # 将特征通过投影头 [b, 2048, 1024] -> [b, 1024]
        # projection = self.projection_head(video_features[:, 0])
        
        # L2 归一化，这对于对比损失至关重要
        projection = nn.functional.normalize(projection, p=2, dim=1)
        
        return projection