import torch
import torch.nn as nn


class VideoMAEAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int = 1024, feature_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_dim),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        outputs = self.backbone(pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            pooled = outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, tuple):
            pooled = outputs[0]
        else:
            pooled = outputs
        projection = self.projection_head(pooled)
        return nn.functional.normalize(projection, p=2, dim=1)
