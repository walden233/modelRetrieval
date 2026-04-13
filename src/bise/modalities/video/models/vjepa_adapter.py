import torch
import torch.nn as nn


class VJEPAAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int = 1024, feature_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_features = self.backbone.get_vision_features(pixel_values)
        pooled = torch.mean(vision_features, dim=1)
        projection = self.projection_head(pooled)
        return nn.functional.normalize(projection, p=2, dim=1)
