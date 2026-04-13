from torch import nn
import torch


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, human_features: torch.Tensor, robot_features: torch.Tensor) -> torch.Tensor:
        batch_size = human_features.shape[0]
        labels = torch.arange(batch_size, device=human_features.device)
        logits = (human_features @ robot_features.T) / self.temperature
        return (self.criterion(logits, labels) + self.criterion(logits.T, labels)) / 2
