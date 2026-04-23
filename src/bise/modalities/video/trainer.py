import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from .losses import InfoNCELoss, intra_domain_consistency_loss, multi_positive_video_contrastive_loss
from .transforms import apply_video_transforms


def train_video_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    loss_config,
    amp: bool = False,
    scaler=None,
    grad_clip_norm: float | None = None,
    gradient_accumulation_steps: int = 1,
    intra_transform_config=None,
):
    # 一个 epoch 的训练主循环。
    # 输入 batch 后先前向，再算跨域对比损失 / 模态内一致性损失，最后反向更新参数。
    model.train()
    total_loss = 0.0
    total_inter = 0.0
    total_intra = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training Video"), start=1):
        human_videos = batch["human_pixel_values"].to(device)
        robot_videos = batch["robot_pixel_values"].to(device)
        # positive_level 决定“正样本”是按 scene 还是按 task 定义。
        labels = _select_labels(batch, loss_config.get("positive_level", "task")).to(device)

        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(human_videos, robot_videos)
            inter_loss = _compute_inter_loss(outputs, labels, loss_config)
            intra_loss = _compute_intra_loss(
                model=model,
                human_videos=human_videos,
                robot_videos=robot_videos,
                logit_scale=outputs["logit_scale_intra"],
                transform_config=intra_transform_config or loss_config.get("intra_transform"),
            )
            total = inter_loss + float(loss_config.get("lambda_intra", 0.0)) * intra_loss
            scaled_loss = total / max(gradient_accumulation_steps, 1)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % max(gradient_accumulation_steps, 1) == 0 or step == len(dataloader):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += total.item()
        total_inter += inter_loss.item()
        total_intra += intra_loss.item()

    batch_count = max(len(dataloader), 1)
    # 返回的是 epoch 平均统计，后续会写入训练曲线和 best_metrics。
    return {
        "loss": total_loss / batch_count,
        "inter_loss": total_inter / batch_count,
        "intra_loss": total_intra / batch_count,
    }


def _select_labels(batch, positive_level: str):
    normalized = str(positive_level).strip().lower()
    if normalized == "scene":
        return batch["scene_indices"]
    return batch["task_indices"]


def _compute_inter_loss(outputs, labels, loss_config):
    loss_name = str(loss_config.get("name", "multi_positive")).strip().lower()
    if loss_name == "info_nce":
        # 兼容最基础的 baseline。
        temperature = float(loss_config.get("temperature", 0.07))
        return InfoNCELoss(temperature=temperature)(outputs["human_embeddings"], outputs["robot_embeddings"])
    return multi_positive_video_contrastive_loss(
        outputs["human_embeddings"],
        outputs["robot_embeddings"],
        labels,
        labels,
        outputs["logit_scale_inter"],
    )


def _compute_intra_loss(model, human_videos, robot_videos, logit_scale, transform_config):
    # 只有配置里显式打开增强一致性时，才计算这一路损失。
    if not transform_config:
        return torch.tensor(0.0, device=human_videos.device)

    human_view1 = _apply_batch_transform(human_videos, transform_config)
    human_view2 = _apply_batch_transform(human_videos, transform_config)
    robot_view1 = _apply_batch_transform(robot_videos, transform_config)
    robot_view2 = _apply_batch_transform(robot_videos, transform_config)
    human_loss = intra_domain_consistency_loss(
        model.encode_human(human_view1),
        model.encode_human(human_view2),
        logit_scale,
    )
    robot_loss = intra_domain_consistency_loss(
        model.encode_robot(robot_view1),
        model.encode_robot(robot_view2),
        logit_scale,
    )
    return (human_loss + robot_loss) / 2.0


def _apply_batch_transform(batch_tensor: torch.Tensor, transform_config):
    # 对 batch 内每个视频分别做增强，再重新 stack 回来。
    transformed = [apply_video_transforms(sample, transform_config) for sample in batch_tensor]
    return torch.stack(transformed, dim=0)
