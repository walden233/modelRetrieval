import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import sys
sys.path.append('.')
from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel
from src.evaluation.trajectory_functions import evaluate_gemini,evaluate
from src.loss.functions import trajectory_symmetric_contrastive_loss 
from src.utils import save_trial_results
from src.utils import augment_human_poses_rotation, augment_robot_tcp_rotation
from src.loss import intra_modal_contrastive_loss

# --- NEW: 阶段 1 训练函数 (仅 Intra) ---
def train_one_epoch_intra_only(model, dataloader, optimizer, device):
    model.train()
    total_loss_intra = 0
    
    for batch in tqdm(dataloader, desc="Stage 1 Pre-training (Intra-Only)"):
        optimizer.zero_grad()
        
        # 1. 数据到 GPU
        human_poses = batch['human_poses'].to(device)
        human_mask = batch['human_mask'].to(device)
        tcp_bases = batch['tcp_bases'].to(device)
        tcp_mask = batch['tcp_mask'].to(device)

        # 2. 创建增强视图
        human_poses_aug1 = augment_human_poses_rotation(human_poses)
        human_poses_aug2 = augment_human_poses_rotation(human_poses)
        tcp_bases_aug1 = augment_robot_tcp_rotation(tcp_bases)
        tcp_bases_aug2 = augment_robot_tcp_rotation(tcp_bases)

        # 3. 阶段 1：模态内对比 (Intra-modal)
        h_embed_aug1 = model.forward_human(human_poses_aug1, human_mask)
        h_embed_aug2 = model.forward_human(human_poses_aug2, human_mask)
        
        r_embed_aug1 = model.forward_robot(tcp_bases_aug1, tcp_mask)
        r_embed_aug2 = model.forward_robot(tcp_bases_aug2, tcp_mask)

        # 4. 计算 Intra 损失
        # 假设模型有 'logit_scale_intra'
        logit_scale_intra_val = model.logit_scale_intra.exp() 
        loss_intra_human = intra_modal_contrastive_loss(h_embed_aug1, h_embed_aug2, logit_scale_intra_val)
        loss_intra_robot = intra_modal_contrastive_loss(r_embed_aug1, r_embed_aug2, logit_scale_intra_val)
        
        loss_intra = (loss_intra_human + loss_intra_robot) / 2.0
        
        # 5. 反向传播 (仅 intra 损失)
        loss_intra.backward()
        optimizer.step()
        
        total_loss_intra += loss_intra.item()
        
    avg_loss_intra = total_loss_intra / len(dataloader)
    return avg_loss_intra


# --- NEW: 阶段 2 训练函数 (Combined) ---
def train_one_epoch_combined(model, dataloader, optimizer, device, intra_loss_weight, use_task_labels: bool = False):
    model.train()
    total_loss = 0
    total_loss_inter = 0
    total_loss_intra = 0
    human_label_key = 'human_task_indices' if use_task_labels else 'human_scene_indices'
    robot_label_key = 'robot_task_indices' if use_task_labels else 'robot_scene_indices'
    
    for batch in tqdm(dataloader, desc="Stage 2 Finetuning (Combined)"):
        optimizer.zero_grad()
        
        # 1. 数据到 GPU
        human_poses = batch['human_poses'].to(device)
        human_mask = batch['human_mask'].to(device)
        tcp_bases = batch['tcp_bases'].to(device)
        tcp_mask = batch['tcp_mask'].to(device)
        human_labels = batch[human_label_key].to(device)
        robot_labels = batch[robot_label_key].to(device)

        # 2. 创建增强视图
        human_poses_aug1 = augment_human_poses_rotation(human_poses)
        human_poses_aug2 = augment_human_poses_rotation(human_poses)
        tcp_bases_aug1 = augment_robot_tcp_rotation(tcp_bases)
        tcp_bases_aug2 = augment_robot_tcp_rotation(tcp_bases)

        # 3. 阶段 1：模态内对比 (Intra-modal)
        h_embed_aug1 = model.forward_human(human_poses_aug1, human_mask)
        h_embed_aug2 = model.forward_human(human_poses_aug2, human_mask)
        
        r_embed_aug1 = model.forward_robot(tcp_bases_aug1, tcp_mask)
        r_embed_aug2 = model.forward_robot(tcp_bases_aug2, tcp_mask)

        # 4. 阶段 2：跨模态对比 (Inter-modal)
        human_embeds_orig, robot_embeds_orig, logit_scale = model(
            human_poses, human_mask, tcp_bases, tcp_mask
        )
        
        # 5. 计算所有损失
        
        # 阶段 2 损失 (Inter-modal)
        loss_inter = trajectory_symmetric_contrastive_loss(
            human_embeds_orig, robot_embeds_orig, human_labels, robot_labels, logit_scale
        )
        
        # 阶段 1 损失 (Intra-modal)
        logit_scale_intra_val = model.logit_scale_intra.exp()
        loss_intra_human = intra_modal_contrastive_loss(h_embed_aug1, h_embed_aug2, logit_scale_intra_val)
        loss_intra_robot = intra_modal_contrastive_loss(r_embed_aug1, r_embed_aug2, logit_scale_intra_val)
        loss_intra = (loss_intra_human + loss_intra_robot) / 2.0
        
        total_loss_batch = loss_inter + intra_loss_weight * loss_intra
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_loss_inter += loss_inter.item()
        total_loss_intra += loss_intra.item()
        
    avg_loss = total_loss / len(dataloader)
    avg_loss_inter = total_loss_inter / len(dataloader)
    avg_loss_intra = total_loss_intra / len(dataloader)
    
    return avg_loss, avg_loss_inter, avg_loss_intra


# --- MODIFIED: 主执行流程 ---
if __name__ == '__main__':
    # --- 设置超参数 ---
    DATASET_ROOT = '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg2' 
    BATCH_SIZE = 16 # 每个批次包含的场景数
    LEARNING_RATE = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR = './results/trajectory_augment_results'
    
    # --- MODIFIED: 两阶段训练超参数 ---
    PRETRAIN_EPOCHS = 30  # 阶段 1: 仅 Intra-modal 预训练
    FINETUNE_EPOCHS = 56  # 阶段 2: 组合损失微调 (你原来的 NUM_EPOCHS)
    INTRA_LOSS_WEIGHT = 3.0 # 阶段 2 中 intra-loss 的权重
    TRAIN_TASK_POSITIVES = False
    EVALUATE_TASK_POSITIVES = False
    USE_6_KEYPOINTS=False
    # --- MODIFIED: model_params ---
    model_params = {
        'human_input_dim': 6 * 3,
        'robot_input_dim': 7,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 5,
        'dim_feedforward': 512,
        'proj_dim': 128,
        'dropout': 0.15,
        'tcp_sample_factor':4
    }
    
    history = {'train_loss': [],'train_loss_inter': [],'train_loss_intra': [], 'val_mean_p_rank': []}
    run_name = f"augment_2stage_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    
    # --- MODIFIED: trial_config ---
    trial_config = {
        'run_name': run_name,
        'device': str(DEVICE),
        'pretrain_epochs': PRETRAIN_EPOCHS,
        'finetune_epochs': FINETUNE_EPOCHS,
        'model_params': model_params,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'intra_loss_weight': INTRA_LOSS_WEIGHT,
        'train_task_positives': TRAIN_TASK_POSITIVES,
        'evaluate_task_positives': EVALUATE_TASK_POSITIVES,
        'use_6_keypoints': USE_6_KEYPOINTS
    }
    os.makedirs(run_dir, exist_ok=True)

    # 1. 初始化数据集和数据加载器 (不变)
    dataset = RH20TTraceDataset(root_dir=DATASET_ROOT,use_6_keypoints=USE_6_KEYPOINTS)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_trajectories)

    # 2. 初始化模型和优化器 (不变)
    model = CrossModalTrajectoryModel(**model_params).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    pretrain_model_path = os.path.join(run_dir, 'pretrain_model.pth')

    # --- 3. 阶段 1: Intra-modal 预训练 ---
    print("--- 开始阶段 1: 模态内(Intra-modal)预训练 ---")
    for epoch in range(PRETRAIN_EPOCHS):
        # 使用 'train_one_epoch_intra_only'
        train_loss_intra = train_one_epoch_intra_only(
            model, train_loader, optimizer, DEVICE
        )
        print(f"Pre-train Epoch {epoch+1}/{PRETRAIN_EPOCHS}, Intra-Loss: {train_loss_intra:.4f}")
        
        # 记录预训练的损失
        history['train_loss_intra'].append(train_loss_intra)
        # 用 None 填充其他历史记录，以保持列表长度一致
        history['train_loss'].append(None)
        history['train_loss_inter'].append(None)
        history['val_mean_p_rank'].append(None)

    torch.save(model.state_dict(), pretrain_model_path)
    print(f"阶段 1 预训练完成。模型已保存到: {pretrain_model_path}")

    # --- 4. 阶段 2: 组合损失微调 ---
    print(f"\n--- 开始阶段 2: 组合损失微调 (共 {FINETUNE_EPOCHS} epochs) ---")
    print(f"加载预训练模型: {pretrain_model_path}")
    model.load_state_dict(torch.load(pretrain_model_path)) # 重新加载模型
    
    # (可选) 你可以在这里重置优化器状态，或使用较低的学习率
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE / 10) 
    
    best_result = None
    best_mean_rank_percen = 1.0

    for epoch in range(FINETUNE_EPOCHS):
        # 使用 'train_one_epoch_combined'
        train_loss, train_loss_inter, train_loss_intra = train_one_epoch_combined(
            model, train_loader, optimizer, DEVICE, INTRA_LOSS_WEIGHT, use_task_labels=TRAIN_TASK_POSITIVES
        )
        
        # 评估部分不变
        result = evaluate_gemini(model, val_loader, DEVICE, group_by_task=EVALUATE_TASK_POSITIVES)
        recalls = result['recalls']
        
        # 记录历史
        # 注意：索引会接着 Pre-train 的
        history['train_loss'].append(train_loss)
        history['train_loss_inter'].append(train_loss_inter)
        history['train_loss_intra'].append(train_loss_intra)
        history['val_mean_p_rank'].append(result['mean_percentage_rank'])
        
        # --- MODIFIED ---: 更新打印信息
        print(f"Finetune Epoch {epoch+1}/{FINETUNE_EPOCHS}, Train Loss: {train_loss:.4f} "
              f"(Inter: {train_loss_inter:.4f}, Intra: {train_loss_intra:.4f})")
        
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
        print(f"Mean Rank: {result['mean_rank']:.2f}, MRR: {result['mrr']:.4f}, Mean Percentage Rank: {result['mean_percentage_rank']:.4f}")

        if result['mean_percentage_rank'] < best_mean_rank_percen:
            best_mean_rank_percen = result['mean_percentage_rank']
            best_result = result
            best_model_path = os.path.join(run_dir, 'best_finetune_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    print("训练完成。best_result:", best_result)
    save_trial_results(run_dir, trial_config, history, best_result, partten=2)
    print(f"本次试验的结果已保存至: {run_dir}")
