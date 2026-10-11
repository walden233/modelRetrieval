import torch
import torch.nn.functional as F


def multi_positive_intra_modal_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    logit_scale: torch.Tensor,
    scene_labels: torch.Tensor,
    task_labels: torch.Tensor,
    task_positive_weight: float = 0.25,
    scene_positive_weight: float = 0.5,
) -> torch.Tensor:
    """Match augmented views with self, scene and task weights of 1, 0.5 and 0.25."""
    if z1.shape[0] == 0:
        return torch.tensor(0.0, device=z1.device)

    scene_labels = scene_labels.to(z1.device)
    task_labels = task_labels.to(z1.device)
    scene_matches = scene_labels.unsqueeze(1) == scene_labels.unsqueeze(0)
    task_matches = task_labels.unsqueeze(1) == task_labels.unsqueeze(0)

    # Each batch contains distinct scene-camera samples; off-diagonal scene matches are views.
    target_weights = scene_matches.float() * scene_positive_weight
    target_weights += (task_matches & ~scene_matches).float() * task_positive_weight
    target_weights.fill_diagonal_(1.0)
    target_probs = target_weights / target_weights.sum(dim=1, keepdim=True)

    logits = logit_scale * (z1 @ z2.t())
    loss_z1 = F.kl_div(F.log_softmax(logits, dim=1), target_probs, reduction="batchmean")
    loss_z2 = F.kl_div(F.log_softmax(logits.t(), dim=1), target_probs, reduction="batchmean")
    return (loss_z1 + loss_z2) / 2.0
