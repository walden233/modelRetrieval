import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F # --- NEW ---
import kornia # --- NEW ---

import sys
sys.path.append('.')
from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel
from src.evalution.trajectory_functions import evaluate_gemini,evaluate
from src.loss.functions import trajectory_symmetric_contrastive_loss 

# --- NEW: Augmentation Helper Functions ---

def augment_human_poses_rotation(poses_batch: torch.Tensor) -> torch.Tensor:
    """
    对人类姿态批次应用随机 3D 旋转。
    输入形状: (B, S, 21, 3)
    输出同形状: (B, S, 21, 3)
    """
    B, S, N, _ = poses_batch.shape
    device = poses_batch.device
    
    # 1. 随机轴与角
    rand_axis = F.normalize(torch.randn(B, 3, device=device), dim=-1)     # (B, 3)
    rand_angle = torch.rand(B, 1, device=device) * 2 * torch.pi           # (B, 1)

    # 2. 轴角 → 旋转矩阵 (B, 3, 3)
    rot_mats = kornia.geometry.axis_angle_to_rotation_matrix(rand_axis * rand_angle)

    # 3. reshape 为 (B, S*N, 3)，一次性旋转
    poses_flat = poses_batch.view(B, S*N, 3)              # (B, S*N, 3)

    # 4. 旋转 (B, S*N, 3)
    rotated = torch.bmm(poses_flat, rot_mats)             # (B, S*N, 3)

    # 5. reshape 回原样
    return rotated.view(B, S, N, 3)

def augment_robot_tcp_rotation(tcp_batch: torch.Tensor) -> torch.Tensor:
    """
    对机器人 TCP 批次的位置部分应用随机 3D 旋转。
    输入形状: (B, S, 7) = [x, y, z, qx, qy, qz, qw]
    输出同形状
    """
    B, S, _ = tcp_batch.shape
    device = tcp_batch.device
    
    positions = tcp_batch[..., :3]     # (B, S, 3)
    quaternions = tcp_batch[..., 3:]   # (B, S, 4)
    
    # 1. 随机旋转
    rand_axis = F.normalize(torch.randn(B, 3, device=device), dim=-1)     # (B, 3)
    rand_angle = torch.rand(B, 1, device=device) * 2 * torch.pi           # (B, 1)

    # 2. Axis-Angle → Rotation Matrix (B, 3, 3)
    rot_mats = kornia.geometry.axis_angle_to_rotation_matrix(rand_axis * rand_angle)

    # 3. 旋转位置 (展平为 B, S, 3)
    positions_flat = positions.view(B, S, 3)     # (B, S, 3)

    # 4. batch 矩阵乘法
    rotated_positions = torch.bmm(positions_flat, rot_mats)  # (B, S, 3)

    # 5. 合并回 (B, S, 7)
    return torch.cat([rotated_positions, quaternions], dim=-1)


# --- NEW: Intra-modal Contrastive Loss ---

def intra_modal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    计算模态内的对比损失 (SimCLR-style)。
    z1, z2: 两个增强视图的嵌入, 形状 [N, D]
    temperature: 温度参数
    """
    N = z1.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=z1.device)
        
    device = z1.device
    
    # 计算 z1 -> z2 的损失
    sim_z1_z2 = (z1 @ z2.T) / temperature # [N, N]
    
    # 计算 z2 -> z1 的损失
    sim_z2_z1 = (z2 @ z1.T) / temperature # [N, N]
    
    labels = torch.arange(N, device=device)
    
    loss_z1 = F.cross_entropy(sim_z1_z2, labels)
    loss_z2 = F.cross_entropy(sim_z2_z1, labels)
    
    return (loss_z1 + loss_z2) / 2.0


# --- MODIFIED: train_one_epoch ---

def train_one_epoch_augment(model, dataloader, optimizer, device, intra_loss_weight):
    model.train()
    total_loss = 0
    total_loss_inter = 0
    total_loss_intra = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        # 1. 将所有数据移到 GPU
        human_poses = batch['human_poses'].to(device)
        human_mask = batch['human_mask'].to(device)
        tcp_bases = batch['tcp_bases'].to(device)
        tcp_mask = batch['tcp_mask'].to(device)
        human_scenes = batch['human_scene_indices'].to(device)
        robot_scenes = batch['robot_scene_indices'].to(device)

        # 2. (新增) 创建两个不同的增强视图
        # 注意：kornia 的旋转是确定性的，因此我们为 aug1 和 aug2 分别调用它
        # （kornia 内部的随机性是基于 PyTorch 的 RNG 状态的，
        # 但为清晰起见，我们假设每次调用都会产生新的随机参数）
        # 实际上，由于上面我们的实现方式，每次调用都会生成新的随机角度/轴
        human_poses_aug1 = augment_human_poses_rotation(human_poses)
        human_poses_aug2 = augment_human_poses_rotation(human_poses)
        
        tcp_bases_aug1 = augment_robot_tcp_rotation(tcp_bases)
        tcp_bases_aug2 = augment_robot_tcp_rotation(tcp_bases)

        # 3. (新增) 阶段 1：模态内对比 (Intra-modal)
        #    使用模型重构后的 forward_human 和 forward_robot 方法
        h_embed_aug1 = model.forward_human(human_poses_aug1, human_mask)
        h_embed_aug2 = model.forward_human(human_poses_aug2, human_mask)
        
        # 确保对机器人轨迹使用相同的采样因子
        r_embed_aug1 = model.forward_robot(tcp_bases_aug1, tcp_mask)
        r_embed_aug2 = model.forward_robot(tcp_bases_aug2, tcp_mask)

        # 4. 阶段 2：跨模态对比 (Inter-modal)
        #    使用 *原始* 数据和模型的标准 forward 方法
        #    注意：这里的 logit_scale 将被用于 *所有* 损失计算
        human_embeds_orig, robot_embeds_orig, logit_scale = model(
            human_poses, human_mask, tcp_bases, tcp_mask
        )
        
        # 5. 计算所有损失
        
        # 阶段 2 损失 (Inter-modal)
        loss_inter = trajectory_symmetric_contrastive_loss(
            human_embeds_orig, robot_embeds_orig, human_scenes, robot_scenes, logit_scale
        )
        
        # 阶段 1 损失 (Intra-modal)
        loss_intra_human = intra_modal_contrastive_loss(h_embed_aug1, h_embed_aug2, logit_scale)
        loss_intra_robot = intra_modal_contrastive_loss(r_embed_aug1, r_embed_aug2, logit_scale)
        
        loss_intra = (loss_intra_human + loss_intra_robot) / 2.0
        
        # # 6. 组合损失
        total_loss_batch = loss_inter + intra_loss_weight * loss_intra
        # # 7. 反向传播
        # total_loss_batch.backward()

        loss_intra.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_loss_inter += loss_inter.item()
        total_loss_intra += loss_intra.item()
        
    avg_loss = total_loss / len(dataloader)
    avg_loss_inter = total_loss_inter / len(dataloader)
    avg_loss_intra = total_loss_intra / len(dataloader)
    
    return avg_loss, avg_loss_inter, avg_loss_intra


# --- 主执行流程 ---
if __name__ == '__main__':
    # 设置超参数
    DATASET_ROOT = '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg2' 
    BATCH_SIZE = 16 # 每个批次包含的场景数
    NUM_EPOCHS = 60
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INTRA_LOSS_WEIGHT = 0.5 # --- NEW ---: 模态内损失的权重
    
    model_params = {
        'human_input_dim': 21 * 3,
        'robot_input_dim': 7,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 5,
        'dim_feedforward': 512,
        'proj_dim': 128,
        'dropout': 0.15,
        'tcp_sample_factor':4
    }

    # 1. 初始化数据集和数据加载器 (不变)
    dataset = RH20TTraceDataset(root_dir=DATASET_ROOT)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_trajectories)

    # 2. 初始化模型和优化器 (不变)
    model = CrossModalTrajectoryModel(**model_params).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. 训练和验证循环 (--- MODIFIED ---)
    best_result = None
    best_mean_rank_percen = 1.0
    for epoch in range(NUM_EPOCHS):
        # --- MODIFIED ---: 传入权重并接收多个损失
        train_loss, train_loss_inter, train_loss_intra = train_one_epoch_augment(
            model, train_loader, optimizer, DEVICE, INTRA_LOSS_WEIGHT
        )
        
        # 评估部分不变
        result = evaluate_gemini(model, val_loader, DEVICE)
        recalls = result['recalls']
        
        # --- MODIFIED ---: 更新打印信息
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f} "
              f"(Inter: {train_loss_inter:.4f}, Intra: {train_loss_intra:.4f})")
        
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
        print(f"Mean Rank: {result['mean_rank']:.2f}, MRR: {result['mrr']:.4f}, Mean Percentage Rank: {result['mean_percentage_rank']:.4f}")

        if result['mean_percentage_rank'] < best_mean_rank_percen:
            best_mean_rank_percen = result['mean_percentage_rank']
            best_result = result
            torch.save(model.state_dict(), 'model_weight/best_trajectory_model.pth')
            print("Saved new best model.")

    print("训练完成。best_result:", best_result)