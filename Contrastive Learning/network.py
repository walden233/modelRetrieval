import torch.nn as nn
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

class ContrastiveFineTuner(nn.Module):
    def __init__(self, model_name="OpenGVLab/VideoMAEv2-Large", feature_dim=128):
        super().__init__()
        # 1. 加载预训练模型作为骨干网络
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.videomae = AutoModel.from_pretrained(
            model_name, 
            config=self.config, 
            trust_remote_code=True
        )
        
        # 2. 定义投影头
        #    它将骨干网络的高维输出映射到对比学习的低维空间
        hidden_size = self.config.model_config["embed_dim"] # e.g., 1024 for Large model
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim)
        )

    def forward(self, pixel_values):
        # 调整维度以匹配模型期望的输入格式 (B, C, T, H, W) -> (B, T, C, H, W)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        # 获取骨干网络的输出
        # 根据我们之前的讨论，这个特定模型的输出直接就是 (batch, hidden_size) 的特征张量
        video_features = self.videomae(pixel_values=pixel_values)
        
        # 将特征通过投影头
        projection = self.projection_head(video_features)
        
        # L2 归一化，这对于对比损失至关重要
        projection = nn.functional.normalize(projection, p=2, dim=1)
        
        return projection