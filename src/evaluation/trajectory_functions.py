import torch
from tqdm import tqdm


def evaluate(model, dataloader, device, group_by_task: bool = False):
    """Simple recall@K using either scene ids or task ids as labels."""
    model.eval()
    all_human_embeds = []
    all_robot_embeds = []
    all_human_scenes = []
    all_robot_scenes = []
    all_human_tasks = []
    all_robot_tasks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            human_poses = batch['human_poses'].to(device)
            human_mask = batch['human_mask'].to(device)
            tcp_bases = batch['tcp_bases'].to(device)
            tcp_mask = batch['tcp_mask'].to(device)

            human_embeds, robot_embeds, _ = model(human_poses, human_mask, tcp_bases, tcp_mask)

            all_human_embeds.append(human_embeds.cpu())
            all_robot_embeds.append(robot_embeds.cpu())
            all_human_scenes.append(batch['human_scene_indices'])
            all_robot_scenes.append(batch['robot_scene_indices'])
            all_human_tasks.append(batch['human_task_indices'])
            all_robot_tasks.append(batch['robot_task_indices'])

    all_human_embeds = torch.cat(all_human_embeds)
    all_robot_embeds = torch.cat(all_robot_embeds)
    all_human_scenes = torch.cat(all_human_scenes)
    all_robot_scenes = torch.cat(all_robot_scenes)
    all_human_tasks = torch.cat(all_human_tasks)
    all_robot_tasks = torch.cat(all_robot_tasks)

    # 控制正样本归属：group_by_task=True 时以 task 为标签，否则以 scene 为标签
    query_labels = all_human_tasks if group_by_task else all_human_scenes
    gallery_labels = all_robot_tasks if group_by_task else all_robot_scenes

    sim_matrix = all_human_embeds @ all_robot_embeds.t()
    k_values = [1, 5, 10]
    recalls = {k: 0.0 for k in k_values}

    num_queries = len(all_human_embeds)
    for i in range(num_queries):
        query_label = query_labels[i]
        top_k_indices = torch.topk(sim_matrix[i], max(k_values)).indices
        retrieved_labels = gallery_labels[top_k_indices]

        for k in k_values:
            if (retrieved_labels[:k] == query_label).any():
                recalls[k] += 1

    for k in k_values:
        recalls[k] /= max(num_queries, 1)

    return recalls


def evaluate_gemini(model, dataloader, device, group_by_task: bool = False):
    """Scene/task-level ranking evaluation that collapses duplicate labels."""
    model.eval()
    all_human_embeds = []
    all_robot_embeds = []
    all_human_scenes = []
    all_robot_scenes = []
    all_human_tasks = []
    all_robot_tasks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            human_poses = batch['human_poses'].to(device)
            human_mask = batch['human_mask'].to(device)
            tcp_bases = batch['tcp_bases'].to(device)
            tcp_mask = batch['tcp_mask'].to(device)

            human_embeds, robot_embeds, _ = model(human_poses, human_mask, tcp_bases, tcp_mask)

            all_human_embeds.append(human_embeds.cpu())
            all_robot_embeds.append(robot_embeds.cpu())
            all_human_scenes.append(batch['human_scene_indices'])
            all_robot_scenes.append(batch['robot_scene_indices'])
            all_human_tasks.append(batch['human_task_indices'])
            all_robot_tasks.append(batch['robot_task_indices'])

    all_human_embeds = torch.cat(all_human_embeds).to(device)
    all_robot_embeds = torch.cat(all_robot_embeds).to(device)
    all_human_scenes = torch.cat(all_human_scenes).to(device)
    all_robot_scenes = torch.cat(all_robot_scenes).to(device)
    all_human_tasks = torch.cat(all_human_tasks).to(device)
    all_robot_tasks = torch.cat(all_robot_tasks).to(device)

    # 同样以任务或场景标签控制正负样本
    query_labels = all_human_tasks if group_by_task else all_human_scenes
    gallery_labels = all_robot_tasks if group_by_task else all_robot_scenes

    sim_matrix = all_human_embeds @ all_robot_embeds.t()
    num_human_queries = sim_matrix.shape[0]

    # 对机器人侧的标签去重，确保同一 task/scene 只取一列
    unique_gallery_labels, inverse_indices = torch.unique(gallery_labels, return_inverse=True)
    num_unique_labels = len(unique_gallery_labels)

    scene_indices_broadcasted = inverse_indices.unsqueeze(0).expand_as(sim_matrix)
    scene_sim_matrix = torch.full((num_human_queries, num_unique_labels), -float('inf'), device=device)
    # 对共享标签的列取最大值，得到 label 级别的相似度
    scene_sim_matrix.scatter_reduce_(dim=1, index=scene_indices_broadcasted, src=sim_matrix, reduce='amax')

    label_to_col_map = {label.item(): col_idx for col_idx, label in enumerate(unique_gallery_labels)}
    target_scene_cols = torch.tensor(
        [label_to_col_map.get(label.item(), -1) for label in query_labels],
        device=device,
        dtype=torch.long
    )

    valid_queries_mask = (target_scene_cols != -1)
    if not valid_queries_mask.all():
        missing = num_human_queries - valid_queries_mask.sum()
        print(f"Warning: {missing} human queries have no matching robot labels. Ignoring them.")
        scene_sim_matrix = scene_sim_matrix[valid_queries_mask]
        target_scene_cols = target_scene_cols[valid_queries_mask]
        num_valid_queries = valid_queries_mask.sum().item()
    else:
        num_valid_queries = num_human_queries

    if num_valid_queries == 0:
        print("Error: No valid queries found.")
        recalls = {k: 0.0 for k in [1, 5, 10]}
        return {
            'recalls': recalls,
            'mean_rank': float('nan'),
            'mrr': float('nan'),
            'mean_percentage_rank': float('nan')
        }

    k_values = [1, 5, 10]
    recalls = {k: 0.0 for k in k_values}
    max_k = max(k_values)
    _, topk_scene_indices = torch.topk(scene_sim_matrix, max_k, dim=1)
    correct_in_topk = (target_scene_cols.unsqueeze(1) == topk_scene_indices)

    for k in k_values:
        recall_at_k = correct_in_topk[:, :k].any(dim=1).sum().item()
        recalls[k] = recall_at_k / num_valid_queries

    sorted_indices = torch.argsort(scene_sim_matrix, dim=1, descending=True)
    target_mask = (sorted_indices == target_scene_cols.unsqueeze(1))
    ranks_0_indexed = torch.argmax(target_mask.long(), dim=1)
    ranks_1_indexed = ranks_0_indexed.float() + 1.0

    mean_rank = ranks_1_indexed.mean().item()
    reciprocal_ranks = 1.0 / ranks_1_indexed
    mrr = reciprocal_ranks.mean().item()
    mean_percentage_rank = mean_rank / num_unique_labels

    metrics = {
        'recalls': recalls,
        'mean_rank': mean_rank,
        'mrr': mrr,
        'mean_percentage_rank': mean_percentage_rank
    }
    return metrics
