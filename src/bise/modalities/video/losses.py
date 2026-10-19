import torch
import torch.nn.functional as F
from torch import nn

from bise.common.losses import multi_positive_intra_modal_loss as intra_domain_consistency_loss


def negative_task_separation_loss(human_embeds, robot_embeds, task_labels, margin=0.3):
    """Penalize high cosine scores between different tasks in HH, RR and HR pairs."""
    if not -1.0 <= margin <= 1.0:
        raise ValueError("negative_separation.margin must be between -1 and 1")
    embeddings = torch.cat((human_embeds, robot_embeds), dim=0).float()
    labels = task_labels.to(embeddings.device).repeat(2)
    different_tasks = labels[:, None] != labels[None, :]
    if not different_tasks.any():
        return embeddings.sum() * 0.0
    # The penalty uses raw cosine rather than the trainable contrastive logit scale.
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        cosine = embeddings @ embeddings.T
        return F.relu(cosine[different_tasks] - margin).square().mean()


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
