import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('.')  # 添加 src 目录到系统路径
from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel    

def symmetric_contrastive_loss(human_embeds, robot_embeds, human_scenes, robot_scenes, logit_scale):
    """
    计算对称式对比损失。
    Args:
        human_embeds: (N, D) 形状的人类轨迹嵌入
        robot_embeds: (M, D) 形状的机器人轨迹嵌入
        human_scenes: (N,) 形状，表示每个人类轨迹所属的场景索引
        robot_scenes: (M,) 形状，表示每个机器人轨迹所属的场景索引
        logit_scale: 温度参数
    """
    # 计算相似度矩阵
    logits_per_human = logit_scale * human_embeds @ robot_embeds.t()
    logits_per_robot = logits_per_human.t()

    # 创建目标标签
    # 如果 human_i 和 robot_j 来自同一个场景，则它们是正样本对
    # (N, M)
    labels = (human_scenes.unsqueeze(1) == robot_scenes.unsqueeze(0)).float().to(logits_per_human.device)
    
    # 由于一个场景有多个正样本，我们需要对标签进行归一化
    labels_h_r = labels / labels.sum(dim=1, keepdim=True).clamp(min=1.0)
    labels_r_h = labels.t() / labels.t().sum(dim=1, keepdim=True).clamp(min=1.0)

    # 计算交叉熵损失 (使用 KL 散度形式以支持软标签)
    loss_h_r = F.kl_div(F.log_softmax(logits_per_human, dim=1), labels_h_r, reduction='batchmean')
    loss_r_h = F.kl_div(F.log_softmax(logits_per_robot, dim=1), labels_r_h, reduction='batchmean')

    return (loss_h_r + loss_r_h) / 2

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        human_poses = batch['human_poses'].to(device)
        human_mask = batch['human_mask'].to(device)
        tcp_bases = batch['tcp_bases'].to(device)
        tcp_mask = batch['tcp_mask'].to(device)
        human_scenes = batch['human_scene_indices'].to(device)
        robot_scenes = batch['robot_scene_indices'].to(device)

        human_embeds, robot_embeds, logit_scale = model(human_poses, human_mask, tcp_bases, tcp_mask)
        
        loss = symmetric_contrastive_loss(human_embeds, robot_embeds, human_scenes, robot_scenes, logit_scale)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

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

# --- 主执行流程 ---
if __name__ == '__main__':
    # 设置超参数
    DATASET_ROOT = '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg3' # 请替换为您的数据集路径
    BATCH_SIZE = 6 # 每个批次包含的场景数
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型超参数
    # 表 2: 关键模型超参数
    # | 超参数 | 值 | 理由 |
    # |-----------------|------|--------------------------------------------------------------------|
    # | d_model | 256 | 在模型容量和计算成本之间取得良好平衡。 |
    # | nhead | 8 | 标准选择，允许模型关注不同子空间的信息。 |
    # | num_layers | 4 | 对于中等长度的序列，4层提供了足够的深度来学习复杂的时序依赖。 |
    # | dim_feedforward | 1024 | 通常是 d_model 的 4 倍，为模型提供足够的表示能力。 |
    # | proj_dim | 128 | 对比学习中投射头的输出维度，通常小于 d_model。 |
    # | dropout | 0.1 | 标准的正则化手段，防止过拟合。 |
    
    model_params = {
        'human_input_dim': 21 * 3,
        'robot_input_dim': 7,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'proj_dim': 128,
        'dropout': 0.1
    }

    # 1. 初始化数据集和数据加载器
    dataset = RH20TTraceDataset(root_dir=DATASET_ROOT)
    # 假设 80% 训练，20% 验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_trajectories)

    # 2. 初始化模型和优化器
    model = CrossModalTrajectoryModel(**model_params).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. 训练和验证循环
    best_recall_at_1 = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        recalls = evaluate(model, val_loader, DEVICE)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")

        if recalls[1] > best_recall_at_1:
            best_recall_at_1 = recalls[1]
            torch.save(model.state_dict(), 'best_trajectory_model.pth')
            print("Saved new best model.")

    print("训练完成。best R@1:", best_recall_at_1)