import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from .losses import (
    InfoNCELoss, intra_domain_consistency_loss, multi_positive_video_contrastive_loss,
    negative_task_separation_loss,
)
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
    semantic_supervisor=None,
    epoch: int = 0,
    dataset_cfg: str = "cfg2",
):
    # 一个 epoch 的训练主循环。
    # 输入 batch 后先前向，再算跨域对比损失 / 模态内一致性损失，最后反向更新参数。
    model.train()
    total_loss = 0.0
    total_inter = 0.0
    total_intra = 0.0
    total_negative = 0.0
    negative_config = loss_config.get("negative_separation", {})
    negative_weight = float(negative_config.get("weight", 0.0))
    if negative_weight < 0:
        raise ValueError("negative_separation.weight must be nonnegative")
    total_semantic = total_entropy = total_weight = 0.0
    valid_teachers = seen_samples = semantic_queries = 0
    queued_semantic = semantic_supervisor is not None and semantic_supervisor.mode == "queue"
    semantic_candidates = total_semantic_weighted = total_semantic_gradient_ratio = 0.0
    if queued_semantic:
        semantic_supervisor.reset_queue()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc="Training Video"), start=1):
        human_videos = batch["human_pixel_values"].to(device, non_blocking=True)
        robot_videos = batch["robot_pixel_values"].to(device, non_blocking=True)
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
                scene_labels=batch["scene_indices"],
                task_labels=batch["task_indices"],
                scene_positive_weight=float(loss_config.get("scene_positive_weight", 0.5)),
            )
            total = inter_loss + float(loss_config.get("lambda_intra", 0.0)) * intra_loss
            if negative_weight:
                negative_loss = negative_task_separation_loss(
                    outputs["human_embeddings"], outputs["robot_embeddings"],
                    batch["task_indices"], margin=float(negative_config.get("margin", 0.3)),
                )
                total = total + negative_weight * negative_loss
                total_negative += float(negative_loss.detach())
            if semantic_supervisor is not None:
                from .semantic import scene_key
                keys = [scene_key(dataset_cfg, identity) for identity in batch["scene_ids"]]
                semantic_loss, valid, entropy, queries = semantic_supervisor(
                    outputs["human_embeddings"], outputs["robot_embeddings"], keys,
                    getattr(model, "semantic_projector", None),
                )
                weight = semantic_supervisor.loss_weight(epoch + step / len(dataloader))
                total = total + weight * semantic_loss
                total_semantic += float(semantic_loss.detach())
                total_entropy += entropy * queries
                total_weight += weight
                if queued_semantic:
                    semantic_candidates += semantic_supervisor.candidate_count
                    total_semantic_weighted += weight * float(semantic_loss.detach())
                    # Diagnose the actual semantic force on the retrieval embeddings.
                    embeddings = (outputs["human_embeddings"], outputs["robot_embeddings"])
                    sem_grads = torch.autograd.grad(weight * semantic_loss, embeddings,
                                                    retain_graph=True)
                    inter_grads = torch.autograd.grad(inter_loss, embeddings, retain_graph=True)
                    sem_norm = sum(g.float().square().sum() for g in sem_grads).sqrt()
                    inter_norm = sum(g.float().square().sum() for g in inter_grads).sqrt()
                    total_semantic_gradient_ratio += float(sem_norm / inter_norm.clamp_min(1e-8))
                valid_teachers += valid
                semantic_queries += queries
                seen_samples += len(keys)
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
    result = {
        "loss": total_loss / batch_count,
        "inter_loss": total_inter / batch_count,
        "intra_loss": total_intra / batch_count,
    }
    if negative_weight:
        result.update(negative_loss=total_negative / batch_count,
                      negative_weighted_loss=negative_weight * total_negative / batch_count)
    if semantic_supervisor is not None:
        result.update(
            semantic_loss=total_semantic / batch_count,
            semantic_coverage=valid_teachers / max(seen_samples, 1),
            semantic_entropy=total_entropy / max(semantic_queries, 1),
            semantic_queries=semantic_queries,
            semantic_weight=total_weight / batch_count,
        )
    if queued_semantic:
        result.update(
            semantic_candidates=semantic_candidates / batch_count,
            semantic_weighted_loss=total_semantic_weighted / batch_count,
            semantic_gradient_ratio=total_semantic_gradient_ratio / batch_count,
        )
    return result


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


def _compute_intra_loss(
    model, human_videos, robot_videos, logit_scale, transform_config, scene_labels, task_labels,
    scene_positive_weight: float = 0.5,
):
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
        scene_labels,
        task_labels,
        scene_positive_weight=scene_positive_weight,
    )
    robot_loss = intra_domain_consistency_loss(
        model.encode_robot(robot_view1),
        model.encode_robot(robot_view2),
        logit_scale,
        scene_labels,
        task_labels,
        scene_positive_weight=scene_positive_weight,
    )
    return (human_loss + robot_loss) / 2.0


def _apply_batch_transform(batch_tensor: torch.Tensor, transform_config):
    # 对 batch 内每个视频分别做增强，再重新 stack 回来。
    transformed = [apply_video_transforms(sample, transform_config) for sample in batch_tensor]
    return torch.stack(transformed, dim=0)
