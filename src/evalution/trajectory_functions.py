import torch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    model.eval()
    all_human_embeds =[]
    all_robot_embeds =[]
    all_human_scenes =[]
    all_robot_scenes =[]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            human_poses = batch['human_poses'].to(device)
            human_mask = batch['human_mask'].to(device)
            tcp_bases = batch['tcp_bases'].to(device)
            tcp_mask = batch['tcp_mask'].to(device)
            
            # 注意：在评估时，我们使用编码器的输出，而不是投射头的输出
            # 这里为了简化，我们继续使用投射头的输出，但在实际应用中应分开
            human_embeds, robot_embeds, _ = model(human_poses, human_mask, tcp_bases, tcp_mask)
            
            all_human_embeds.append(human_embeds.cpu())
            all_robot_embeds.append(robot_embeds.cpu())
            all_human_scenes.append(batch['human_scene_indices'])
            all_robot_scenes.append(batch['robot_scene_indices'])

    all_human_embeds = torch.cat(all_human_embeds)
    all_robot_embeds = torch.cat(all_robot_embeds)
    all_human_scenes = torch.cat(all_human_scenes)
    all_robot_scenes = torch.cat(all_robot_scenes)

    # 计算 Recall@K
    sim_matrix = all_human_embeds @ all_robot_embeds.t()
    
    # 对于每个人类轨迹，找到 top-K 相似的机器人轨迹
    k_values = [1, 5, 10]
    recalls = {k: 0.0 for k in k_values}
    
    num_queries = len(all_human_embeds)
    for i in range(num_queries):
        query_scene = all_human_scenes[i]
        top_k_indices = torch.topk(sim_matrix[i], max(k_values)).indices
        retrieved_scenes = all_robot_scenes[top_k_indices]
        
        for k in k_values:
            if query_scene in retrieved_scenes[:k]:
                recalls[k] += 1
    
    for k in k_values:
        recalls[k] /= num_queries
        
    return recalls


def evaluate_gemini(model, dataloader, device):
    model.eval()
    all_human_embeds = []
    all_robot_embeds = []
    all_human_scenes = []
    all_robot_scenes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            human_poses = batch['human_poses'].to(device)
            human_mask = batch['human_mask'].to(device)
            tcp_bases = batch['tcp_bases'].to(device)
            tcp_mask = batch['tcp_mask'].to(device)
            
            # 注意：在评估时，我们使用编码器的输出，而不是投射头的输出
            # 这里为了简化，我们继续使用投射头的输出，但在实际应用中应分开
            human_embeds, robot_embeds, _ = model(human_poses, human_mask, tcp_bases, tcp_mask)
            
            all_human_embeds.append(human_embeds.cpu())
            all_robot_embeds.append(robot_embeds.cpu())
            all_human_scenes.append(batch['human_scene_indices'])
            all_robot_scenes.append(batch['robot_scene_indices'])

    # --- 原始拼接部分 ---
    all_human_embeds = torch.cat(all_human_embeds).to(device)
    all_robot_embeds = torch.cat(all_robot_embeds).to(device)
    all_human_scenes = torch.cat(all_human_scenes).to(device)
    all_robot_scenes = torch.cat(all_robot_scenes).to(device)


    # --- 修改 1：计算场景级别的相似度 ---

    # 1. 计算原始的 (N_human, N_robot) 相似度矩阵
    # sim_matrix[i, k] = 相似度(human_embed[i], robot_embed[k])
    sim_matrix = all_human_embeds @ all_robot_embeds.t()
    
    num_human_queries = sim_matrix.shape[0]

    # 2. 获取所有唯一的机器人场景，以及每个机器人嵌入对应的唯一场景索引
    # unique_robot_scenes: [scene_id_1, scene_id_2, ...] (已排序)
    # inverse_indices: [0, 2, 1, 0, ...] (长度 N_robot)，
    #                  表示 all_robot_scenes[k] 对应 unique_robot_scenes 中的索引
    unique_robot_scenes, inverse_indices = torch.unique(all_robot_scenes, return_inverse=True)
    num_unique_robot_scenes = len(unique_robot_scenes)

    # 3. 创建一个新的 [N_human, N_unique_scenes] 场景相似度矩阵
    # 我们希望 scene_sim_matrix[i, j] = max(sim_matrix[i, k])
    #                                   for all k where robot_scene[k] == unique_robot_scenes[j]
    
    # 广播 inverse_indices 到 [N_human, N_robot]
    scene_indices_broadcasted = inverse_indices.unsqueeze(0).expand_as(sim_matrix)

    # 初始化场景相似度矩阵为负无穷
    scene_sim_matrix = torch.full((num_human_queries, num_unique_robot_scenes), -float('inf'), device=device)

    # 使用 scatter_reduce_ 和 'amax' (max) 高效计算
    # 它会根据 scene_indices_broadcasted 中的索引（场景索引 j），
    # 从 sim_matrix 中（源）获取值，并计算最大值，存入 scene_sim_matrix（目标）
    scene_sim_matrix.scatter_reduce_(dim=1, index=scene_indices_broadcasted, src=sim_matrix, reduce='amax')

    # --- 修改 2：计算新指标 (Recall@K, Mean Rank, MRR) ---

    # 4. 找到每个 human_query 对应的 *正确* 场景索引（在 scene_sim_matrix 中的列索引）
    
    # 创建一个从 {原始 scene_id -> 唯一场景索引 j} 的映射
    scene_to_col_map = {scene_id.item(): col_idx for col_idx, scene_id in enumerate(unique_robot_scenes)}
    
    # 找到每个 human_scene 对应的列索引
    # .get(s.item(), -1) - 如果 human_scene 不在 robot_scenes 中，标记为 -1
    target_scene_cols = torch.tensor(
        [scene_to_col_map.get(s.item(), -1) for s in all_human_scenes], 
        device=device,
        dtype=torch.long
    )

    # 5. 过滤掉那些在 robot_scenes 中没有对应场景的 human_queries
    valid_queries_mask = (target_scene_cols != -1)
    if not valid_queries_mask.all():
        print(f"Warning: {num_human_queries - valid_queries_mask.sum()} human queries have no matching robot scene. Ignoring them.")
        scene_sim_matrix = scene_sim_matrix[valid_queries_mask]
        target_scene_cols = target_scene_cols[valid_queries_mask]
        num_valid_queries = valid_queries_mask.sum().item()
    else:
        num_valid_queries = num_human_queries

    if num_valid_queries == 0:
        print("Error: No valid queries found.")
        recalls = {k: 0.0 for k in [1, 5, 10]}
        return {'recalls': recalls, 'mean_rank': float('nan'), 'mrr': float('nan'), 'mean_percentage_rank': float('nan')}

    # 6. 计算 Recall@K
    k_values = [1, 5, 10]
    recalls = {k: 0.0 for k in k_values}
    max_k = max(k_values)

    # 对场景相似度进行排序，获取 top-k 场景的 *列索引*
    _, topk_scene_indices = torch.topk(scene_sim_matrix, max_k, dim=1) # [N_valid, max_k]

    # 检查正确的目标场景是否在 top-k 列表中
    # (target_scene_cols.unsqueeze(1) == topk_scene_indices) -> [N_valid, max_k]
    correct_in_topk = (target_scene_cols.unsqueeze(1) == topk_scene_indices)

    for k in k_values:
        # .any(dim=1) 检查每行（每个 query）的前 k 列是否有 True
        recall_at_k = correct_in_topk[:, :k].any(dim=1).sum().item()
        recalls[k] = recall_at_k / num_valid_queries

    # 7. 计算 Mean Rank 和 MRR (Mean Reciprocal Rank)
    
    # 获取所有场景的完整排名
    # sorted_indices[i, r] = 第 i 个 query 的第 r 名的场景的 *列索引*
    sorted_indices = torch.argsort(scene_sim_matrix, dim=1, descending=True) # [N_valid, N_unique_scenes]

    # 找到目标场景在排序列表中的位置（即排名）
    # target_mask[i, r] = True  iff sorted_indices[i, r] == target_scene_cols[i]
    target_mask = (sorted_indices == target_scene_cols.unsqueeze(1))

    # 使用 argmax 找到 *第一个* True 的索引（即最高排名）
    # 排名是从 0 开始的 (0 = 第1名)
    ranks_0_indexed = torch.argmax(target_mask.long(), dim=1)
    
    # 转换为 1-indexed 排名 (1 = 第1名)
    ranks_1_indexed = ranks_0_indexed.float() + 1.0

    # 正确结果的平均排名
    mean_rank = ranks_1_indexed.mean().item()
    
    # 平均倒数排名 (MRR)
    reciprocal_ranks = 1.0 / ranks_1_indexed
    mrr = reciprocal_ranks.mean().item()

    # 排名比例 (Mean Percentage Rank)
    mean_percentage_rank = mean_rank / num_unique_robot_scenes

    metrics = {
        'recalls': recalls,
        'mean_rank': mean_rank,
        'mrr': mrr,
        'mean_percentage_rank': mean_percentage_rank
    }
    
    return metrics