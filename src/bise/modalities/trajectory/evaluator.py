import torch
from tqdm import tqdm


def evaluate_retrieval(model, dataloader, device, group_by_task: bool = False):
    model.eval()
    all_human_embeds = []
    all_robot_embeds = []
    all_human_scenes = []
    all_robot_scenes = []
    all_human_tasks = []
    all_robot_tasks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            human_poses = batch["human_poses"].to(device)
            human_mask = batch["human_mask"].to(device)
            tcp_bases = batch["tcp_bases"].to(device)
            tcp_mask = batch["tcp_mask"].to(device)
            human_embeds, robot_embeds, _ = model(human_poses, human_mask, tcp_bases, tcp_mask)
            all_human_embeds.append(human_embeds.cpu())
            all_robot_embeds.append(robot_embeds.cpu())
            all_human_scenes.append(batch["human_scene_indices"])
            all_robot_scenes.append(batch["robot_scene_indices"])
            all_human_tasks.append(batch["human_task_indices"])
            all_robot_tasks.append(batch["robot_task_indices"])

    all_human_embeds = torch.cat(all_human_embeds)
    all_robot_embeds = torch.cat(all_robot_embeds)
    query_labels = torch.cat(all_human_tasks) if group_by_task else torch.cat(all_human_scenes)
    gallery_labels = torch.cat(all_robot_tasks) if group_by_task else torch.cat(all_robot_scenes)

    sim_matrix = all_human_embeds @ all_robot_embeds.t()
    k_values = [1, 5, 10]
    recalls = {k: 0.0 for k in k_values}

    for index in range(len(all_human_embeds)):
        topk_indices = torch.topk(sim_matrix[index], max(k_values)).indices
        retrieved_labels = gallery_labels[topk_indices]
        for k in k_values:
            if (retrieved_labels[:k] == query_labels[index]).any():
                recalls[k] += 1

    for k in k_values:
        recalls[k] /= max(len(all_human_embeds), 1)
    return recalls


def evaluate_retrieval_grouped(model, dataloader, device, group_by_task: bool = False):
    model.eval()
    all_human_embeds = []
    all_robot_embeds = []
    all_human_scenes = []
    all_robot_scenes = []
    all_human_tasks = []
    all_robot_tasks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            human_poses = batch["human_poses"].to(device)
            human_mask = batch["human_mask"].to(device)
            tcp_bases = batch["tcp_bases"].to(device)
            tcp_mask = batch["tcp_mask"].to(device)
            human_embeds, robot_embeds, _ = model(human_poses, human_mask, tcp_bases, tcp_mask)
            all_human_embeds.append(human_embeds.cpu())
            all_robot_embeds.append(robot_embeds.cpu())
            all_human_scenes.append(batch["human_scene_indices"])
            all_robot_scenes.append(batch["robot_scene_indices"])
            all_human_tasks.append(batch["human_task_indices"])
            all_robot_tasks.append(batch["robot_task_indices"])

    all_human_embeds = torch.cat(all_human_embeds).to(device)
    all_robot_embeds = torch.cat(all_robot_embeds).to(device)
    query_labels = torch.cat(all_human_tasks).to(device) if group_by_task else torch.cat(all_human_scenes).to(device)
    gallery_labels = torch.cat(all_robot_tasks).to(device) if group_by_task else torch.cat(all_robot_scenes).to(device)
    sim_matrix = all_human_embeds @ all_robot_embeds.t()

    unique_gallery_labels, inverse_indices = torch.unique(gallery_labels, return_inverse=True)
    scene_sim_matrix = torch.full(
        (sim_matrix.shape[0], len(unique_gallery_labels)),
        -float("inf"),
        device=device,
    )
    scene_indices = inverse_indices.unsqueeze(0).expand_as(sim_matrix)
    scene_sim_matrix.scatter_reduce_(dim=1, index=scene_indices, src=sim_matrix, reduce="amax")

    label_to_col_map = {label.item(): col_idx for col_idx, label in enumerate(unique_gallery_labels)}
    target_cols = torch.tensor([label_to_col_map.get(label.item(), -1) for label in query_labels], device=device)
    valid_queries_mask = target_cols != -1
    scene_sim_matrix = scene_sim_matrix[valid_queries_mask]
    target_cols = target_cols[valid_queries_mask]
    num_valid_queries = int(valid_queries_mask.sum().item())

    if num_valid_queries == 0:
        return {
            "recalls": {1: 0.0, 5: 0.0, 10: 0.0},
            "mean_rank": float("nan"),
            "mrr": float("nan"),
            "mean_percentage_rank": float("nan"),
        }

    recalls = {}
    max_k = min(10, scene_sim_matrix.shape[1])
    topk_scene_indices = torch.topk(scene_sim_matrix, max_k, dim=1).indices
    for k in [1, 5, 10]:
        effective_k = min(k, max_k)
        recalls[k] = (target_cols.unsqueeze(1) == topk_scene_indices[:, :effective_k]).any(dim=1).sum().item() / num_valid_queries

    sorted_indices = torch.argsort(scene_sim_matrix, dim=1, descending=True)
    target_mask = sorted_indices == target_cols.unsqueeze(1)
    ranks = torch.argmax(target_mask.long(), dim=1).float() + 1.0
    mean_rank = ranks.mean().item()
    mrr = (1.0 / ranks).mean().item()
    mean_percentage_rank = mean_rank / len(unique_gallery_labels)

    return {
        "recalls": recalls,
        "mean_rank": mean_rank,
        "mrr": mrr,
        "mean_percentage_rank": mean_percentage_rank,
    }
