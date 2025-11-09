import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F # --- NEW ---
import kornia # --- NEW ---

import os
import sys
sys.path.append('.')
from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel
from src.evaluation.trajectory_functions import evaluate_gemini,evaluate
from src.loss.functions import trajectory_symmetric_contrastive_loss 
from src.utils import save_trial_results,augment_human_poses_rotation, augment_robot_tcp_rotation
from src.loss import intra_modal_contrastive_loss

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
        logit_scale_intra_val = model.logit_scale_intra.exp()
        loss_intra_human = intra_modal_contrastive_loss(h_embed_aug1, h_embed_aug2, logit_scale_intra_val)
        loss_intra_robot = intra_modal_contrastive_loss(r_embed_aug1, r_embed_aug2, logit_scale_intra_val)
        
        loss_intra = (loss_intra_human + loss_intra_robot) / 2.0
        
        # # 6. 组合损失
        total_loss_batch = loss_inter + intra_loss_weight * loss_intra
        # # 7. 反向传播
        total_loss_batch.backward()

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
    NUM_EPOCHS = 55
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    INTRA_LOSS_WEIGHT = 2 # --- NEW ---: 模态内损失的权重
    OUTPUT_DIR = './results/trajectory_augment_results'

    model_params = {
        'human_input_dim': 6 * 3,
        'robot_input_dim': 7,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 5,
        'dim_feedforward': 512,
        'proj_dim': 128,
        'dropout': 0.15,
        'tcp_sample_factor':5
    }
    history = {'train_loss': [],'train_loss_inter': [],'train_loss_intra': [], 'val_mean_p_rank': []}
    run_name = f"augment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    trial_config = {
        'run_name': run_name,
        'device': str(DEVICE),
        'num_epochs': NUM_EPOCHS,
        'model_params': model_params,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'intra_loss_weight': INTRA_LOSS_WEIGHT
    }
    os.makedirs(run_dir, exist_ok=True)

    # 1. 初始化数据集和数据加载器 (不变)
    dataset = RH20TTraceDataset(root_dir=DATASET_ROOT, use_6_keypoints=True)
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
        history['train_loss'].append(train_loss)
        history['train_loss_inter'].append(train_loss_inter)
        history['train_loss_intra'].append(train_loss_intra)
        history['val_mean_p_rank'].append(result['mean_percentage_rank'])
        
        # --- MODIFIED ---: 更新打印信息
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f} "
              f"(Inter: {train_loss_inter:.4f}, Intra: {train_loss_intra:.4f})")
        
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
        print(f"Mean Rank: {result['mean_rank']:.2f}, MRR: {result['mrr']:.4f}, Mean Percentage Rank: {result['mean_percentage_rank']:.4f}")

        if result['mean_percentage_rank'] < best_mean_rank_percen:
            best_mean_rank_percen = result['mean_percentage_rank']
            best_result = result
            torch.save(model.state_dict(), os.path.join(run_dir,'best_trajectory_model.pth'))
            print("Saved new best model.")

    print("训练完成。best_result:", best_result)
    save_trial_results(run_dir, trial_config, history, best_result, partten=2)
    print(f"本次试验的结果已保存至: {run_dir}")