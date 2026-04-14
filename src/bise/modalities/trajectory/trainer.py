import torch
from tqdm import tqdm

from .augmentations import augment_human_poses_rotation, augment_robot_tcp_rotation
from .losses import multi_positive_intra_modal_loss, trajectory_symmetric_contrastive_loss


def _build_augmentations(human_poses, tcp_bases, noise_std: float, max_rotation_degrees: float):
    human_aug1 = augment_human_poses_rotation(
        human_poses,
        noise_std=noise_std,
        max_angle_degrees=max_rotation_degrees,
    )
    human_aug2 = augment_human_poses_rotation(
        human_poses,
        noise_std=noise_std,
        max_angle_degrees=max_rotation_degrees,
    )
    robot_aug1 = augment_robot_tcp_rotation(
        tcp_bases,
        noise_std=noise_std,
        max_angle_degrees=max_rotation_degrees,
    )
    robot_aug2 = augment_robot_tcp_rotation(
        tcp_bases,
        noise_std=noise_std,
        max_angle_degrees=max_rotation_degrees,
    )
    return human_aug1, human_aug2, robot_aug1, robot_aug2


def _compute_intra_loss(
    model,
    human_aug1,
    human_aug2,
    human_mask,
    robot_aug1,
    robot_aug2,
    tcp_mask,
    human_scene_labels,
    human_task_labels,
    robot_scene_labels,
    robot_task_labels,
    intra_task_positive_weight: float,
):
    human_loss = multi_positive_intra_modal_loss(
        model.forward_human(human_aug1, human_mask),
        model.forward_human(human_aug2, human_mask),
        model.logit_scale_intra.exp(),
        human_scene_labels,
        human_task_labels,
        task_positive_weight=intra_task_positive_weight,
    )
    robot_loss = multi_positive_intra_modal_loss(
        model.forward_robot(robot_aug1, tcp_mask),
        model.forward_robot(robot_aug2, tcp_mask),
        model.logit_scale_intra.exp(),
        robot_scene_labels,
        robot_task_labels,
        task_positive_weight=intra_task_positive_weight,
    )
    return (human_loss + robot_loss) / 2.0


def train_trajectory_epoch(model, dataloader, optimizer, device, use_task_labels: bool = False):
    model.train()
    total_loss = 0.0
    human_label_key = "human_task_indices" if use_task_labels else "human_scene_indices"
    robot_label_key = "robot_task_indices" if use_task_labels else "robot_scene_indices"

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        human_poses = batch["human_poses"].to(device)
        human_mask = batch["human_mask"].to(device)
        tcp_bases = batch["tcp_bases"].to(device)
        tcp_mask = batch["tcp_mask"].to(device)
        human_labels = batch[human_label_key].to(device)
        robot_labels = batch[robot_label_key].to(device)
        human_embeds, robot_embeds, logit_scale = model(human_poses, human_mask, tcp_bases, tcp_mask)
        loss = trajectory_symmetric_contrastive_loss(human_embeds, robot_embeds, human_labels, robot_labels, logit_scale)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def pretrain_intra_modal_epoch(
    model,
    dataloader,
    optimizer,
    device,
    intra_task_positive_weight: float = 0.0,
    augmentation_noise_std: float = 0.005,
    augmentation_max_rotation_degrees: float = 10.0,
):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Stage 1 Pre-training"):
        optimizer.zero_grad()
        human_poses = batch["human_poses"].to(device)
        human_mask = batch["human_mask"].to(device)
        tcp_bases = batch["tcp_bases"].to(device)
        tcp_mask = batch["tcp_mask"].to(device)
        human_scene_labels = batch["human_scene_indices"].to(device)
        human_task_labels = batch["human_task_indices"].to(device)
        robot_scene_labels = batch["robot_scene_indices"].to(device)
        robot_task_labels = batch["robot_task_indices"].to(device)
        human_aug1, human_aug2, robot_aug1, robot_aug2 = _build_augmentations(
            human_poses,
            tcp_bases,
            augmentation_noise_std,
            augmentation_max_rotation_degrees,
        )
        loss = _compute_intra_loss(
            model,
            human_aug1,
            human_aug2,
            human_mask,
            robot_aug1,
            robot_aug2,
            tcp_mask,
            human_scene_labels,
            human_task_labels,
            robot_scene_labels,
            robot_task_labels,
            intra_task_positive_weight,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_augmented_trajectory_epoch(
    model,
    dataloader,
    optimizer,
    device,
    intra_loss_weight: float,
    use_task_labels: bool = False,
    intra_task_positive_weight: float = 0.0,
    augmentation_noise_std: float = 0.005,
    augmentation_max_rotation_degrees: float = 10.0,
):
    model.train()
    total_loss = 0.0
    total_loss_inter = 0.0
    total_loss_intra = 0.0
    human_label_key = "human_task_indices" if use_task_labels else "human_scene_indices"
    robot_label_key = "robot_task_indices" if use_task_labels else "robot_scene_indices"

    for batch in tqdm(dataloader, desc="Stage 2 Finetuning"):
        optimizer.zero_grad()
        human_poses = batch["human_poses"].to(device)
        human_mask = batch["human_mask"].to(device)
        tcp_bases = batch["tcp_bases"].to(device)
        tcp_mask = batch["tcp_mask"].to(device)
        human_labels = batch[human_label_key].to(device)
        robot_labels = batch[robot_label_key].to(device)
        human_scene_labels = batch["human_scene_indices"].to(device)
        human_task_labels = batch["human_task_indices"].to(device)
        robot_scene_labels = batch["robot_scene_indices"].to(device)
        robot_task_labels = batch["robot_task_indices"].to(device)

        human_aug1, human_aug2, robot_aug1, robot_aug2 = _build_augmentations(
            human_poses,
            tcp_bases,
            augmentation_noise_std,
            augmentation_max_rotation_degrees,
        )
        loss_intra = _compute_intra_loss(
            model,
            human_aug1,
            human_aug2,
            human_mask,
            robot_aug1,
            robot_aug2,
            tcp_mask,
            human_scene_labels,
            human_task_labels,
            robot_scene_labels,
            robot_task_labels,
            intra_task_positive_weight,
        )

        human_embeds, robot_embeds, logit_scale = model(human_poses, human_mask, tcp_bases, tcp_mask)
        loss_inter = trajectory_symmetric_contrastive_loss(human_embeds, robot_embeds, human_labels, robot_labels, logit_scale)

        loss = loss_inter + intra_loss_weight * loss_intra
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_inter += loss_inter.item()
        total_loss_intra += loss_intra.item()

    batch_count = len(dataloader)
    return (
        total_loss / batch_count,
        total_loss_inter / batch_count,
        total_loss_intra / batch_count,
    )
