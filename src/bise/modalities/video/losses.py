import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    # 最基础的一对一对比损失：假设 batch 内第 i 个 human 只和第 i 个 robot 匹配。
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, human_features: torch.Tensor, robot_features: torch.Tensor) -> torch.Tensor:
        batch_size = human_features.shape[0]
        labels = torch.arange(batch_size, device=human_features.device)
        logits = (human_features @ robot_features.T) / self.temperature
        return (self.criterion(logits, labels) + self.criterion(logits.T, labels)) / 2


def multi_positive_video_contrastive_loss(
    human_embeds: torch.Tensor,
    robot_embeds: torch.Tensor,
    human_labels: torch.Tensor,
    robot_labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    # 多正样本版本：只要标签相同，就都是正样本，不再强行限制一对一。
    # 这更适合“同一 task 下多个 scene / camera 都可视作正样本”的检索设置。
    device = human_embeds.device
    human_labels = human_labels.to(device)
    robot_labels = robot_labels.to(device)
    logits_per_human = logit_scale * (human_embeds @ robot_embeds.t())
    logits_per_robot = logits_per_human.t()

    positive_mask = (human_labels.unsqueeze(1) == robot_labels.unsqueeze(0)).float()
    target_human = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    target_robot = positive_mask.t() / positive_mask.t().sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_human = F.kl_div(F.log_softmax(logits_per_human, dim=1), target_human, reduction="batchmean")
    loss_robot = F.kl_div(F.log_softmax(logits_per_robot, dim=1), target_robot, reduction="batchmean")
    return (loss_human + loss_robot) / 2.0


def intra_domain_consistency_loss(z1: torch.Tensor, z2: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    # 模态内一致性损失：同一个视频经过两次不同增强后，编码结果应尽量接近。
    if z1.shape[0] == 0:
        return torch.tensor(0.0, device=z1.device)
    logits_a = logit_scale * (z1 @ z2.t())
    logits_b = logits_a.t()
    labels = torch.arange(z1.shape[0], device=z1.device)
    return (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels)) / 2.0
