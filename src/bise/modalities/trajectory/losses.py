import torch
import torch.nn.functional as F


def trajectory_symmetric_contrastive_loss(
    human_embeds: torch.Tensor,
    robot_embeds: torch.Tensor,
    human_labels: torch.Tensor,
    robot_labels: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    device = human_embeds.device
    human_labels = human_labels.to(device)
    robot_labels = robot_labels.to(device)
    logits_per_human = logit_scale * human_embeds @ robot_embeds.t()
    logits_per_robot = logits_per_human.t()
    labels = (human_labels.unsqueeze(1) == robot_labels.unsqueeze(0)).float()
    labels_human = labels / labels.sum(dim=1, keepdim=True).clamp(min=1.0)
    labels_robot = labels.t() / labels.t().sum(dim=1, keepdim=True).clamp(min=1.0)
    loss_human = F.kl_div(F.log_softmax(logits_per_human, dim=1), labels_human, reduction="batchmean")
    loss_robot = F.kl_div(F.log_softmax(logits_per_robot, dim=1), labels_robot, reduction="batchmean")
    return (loss_human + loss_robot) / 2


def intra_modal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    num_samples = z1.shape[0]
    if num_samples == 0:
        return torch.tensor(0.0, device=z1.device)

    labels = torch.arange(num_samples, device=z1.device)
    sim_z1_z2 = (z1 @ z2.T) / temperature
    sim_z2_z1 = (z2 @ z1.T) / temperature
    return (F.cross_entropy(sim_z1_z2, labels) + F.cross_entropy(sim_z2_z1, labels)) / 2.0


def multi_positive_intra_modal_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    logit_scale: torch.Tensor,
    scene_labels: torch.Tensor,
    task_labels: torch.Tensor,
    task_positive_weight: float = 0.0,
    scene_positive_weight: float = 1.0,
) -> torch.Tensor:
    num_samples = z1.shape[0]
    if num_samples == 0:
        return torch.tensor(0.0, device=z1.device)

    device = z1.device
    scene_labels = scene_labels.to(device)
    task_labels = task_labels.to(device)

    scene_matches = scene_labels.unsqueeze(1) == scene_labels.unsqueeze(0)
    task_matches = task_labels.unsqueeze(1) == task_labels.unsqueeze(0)

    target_weights = scene_matches.float() * scene_positive_weight
    if task_positive_weight > 0:
        weak_task_matches = task_matches & ~scene_matches
        target_weights = target_weights + weak_task_matches.float() * task_positive_weight

    target_weights.fill_diagonal_(scene_positive_weight)
    target_probs = target_weights / target_weights.sum(dim=1, keepdim=True).clamp(min=1.0)

    logits_z1_z2 = logit_scale * (z1 @ z2.t())
    logits_z2_z1 = logits_z1_z2.t()
    loss_z1 = F.kl_div(F.log_softmax(logits_z1_z2, dim=1), target_probs, reduction="batchmean")
    loss_z2 = F.kl_div(F.log_softmax(logits_z2_z1, dim=1), target_probs, reduction="batchmean")
    return (loss_z1 + loss_z2) / 2.0
