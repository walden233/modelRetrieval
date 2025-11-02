import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append('.')  # 添加 src 目录到系统路径
from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel    
from src.evalution.trajectory_functions import evaluate_gemini,evaluate
from src.loss.functions import trajectory_symmetric_contrastive_loss 


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

        #这里设置tcp_sample_factor，通过采样缩短机器人轨迹序列长度
        human_embeds, robot_embeds, logit_scale = model(human_poses, human_mask, tcp_bases, tcp_mask,tcp_sample_factor=4)
        
        loss = trajectory_symmetric_contrastive_loss(human_embeds, robot_embeds, human_scenes, robot_scenes, logit_scale)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


# --- 主执行流程 ---
if __name__ == '__main__':
    # 设置超参数
    DATASET_ROOT = '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg2' 
    BATCH_SIZE = 16 # 每个批次包含的场景数
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_params = {
        'human_input_dim': 21 * 3,
        'robot_input_dim': 7,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'dim_feedforward': 1024,
        'proj_dim': 128,
        'dropout': 0.15
    }

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
    best_result = None
    best_mean_rank_percen = 1.0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        result = evaluate_gemini(model, val_loader, DEVICE)
        recalls = result['recalls']
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
        print(f"Mean Rank: {result['mean_rank']:.2f}, MRR: {result['mrr']:.4f}, Mean Percentage Rank: {result['mean_percentage_rank']:.4f}")

        if result['mean_percentage_rank'] < best_mean_rank_percen:
            best_mean_rank_percen = result['mean_percentage_rank']
            best_result = result
            torch.save(model.state_dict(), 'model_weight/best_trajectory_model.pth')
            print("Saved new best model.")

    print("训练完成。best_result:", best_result)