import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('.')  # 添加 src 目录到系统路径
from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel    
from src.evaluation.trajectory_functions import evaluate_gemini,evaluate
from src.pipelines.trajectoryTrain import train_one_epoch

# --- 主执行流程 ---
if __name__ == '__main__':
    DATASET_DIRS = ['/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg2', 
                    '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg3']
    BATCH_SIZE = 16
    NUM_EPOCHS_CFG2 = 60
    NUM_EPOCHS_CFG3 = NUM_EPOCHS_CFG2 // 2 # cfg3 的 epoch 取 cfg2 的一半
    LEARNING_RATE = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BEST_MODEL_PATH = 'model_weight/best_trajectory_model.pth' # 统一的最佳模型保存路径

    model_params = {
        'human_input_dim': 21 * 3,
        'robot_input_dim': 7,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 5, 
        'dim_feedforward': 1024,
        'proj_dim': 128,
        'dropout': 0.15 
    }

    # --- 辅助函数：用于创建数据集和加载器 ---
    def create_dataloaders(root_dir, batch_size):
        print(f"\nLoading dataset from: {root_dir}")
        dataset = RH20TTraceDataset(root_dir=root_dir)
        # 假设 80% 训练，20% 验证
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectories)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectories)
        
        print(f"Dataset loaded. Train size: {train_size}, Val size: {val_size}")
        return train_loader, val_loader

    # --- 2. 初始化模型和优化器 ---
    # (模型和优化器在 cfg2 训练开始时初始化)
    model = CrossModalTrajectoryModel(**model_params).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


    # --- 3. 阶段一：在 CFG2 上训练 ---
    print(f"--- 阶段 1：开始在 {DATASET_DIRS[0]} 上训练 ---")
    train_loader_cfg2, val_loader_cfg2 = create_dataloaders(DATASET_DIRS[0], BATCH_SIZE)

    best_mean_rank_percen_cfg2 = 1.0
    best_result_cfg2 = None

    for epoch in range(NUM_EPOCHS_CFG2):
        train_loss = train_one_epoch(model, train_loader_cfg2, optimizer, DEVICE)
        result = evaluate_gemini(model, val_loader_cfg2, DEVICE)
        recalls = result['recalls']
        
        print(f"[CFG2] Epoch {epoch+1}/{NUM_EPOCHS_CFG2}, Train Loss: {train_loss:.4f}")
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
        print(f"Mean Rank: {result['mean_rank']:.2f}, MRR: {result['mrr']:.4f}, Mean Percentage Rank: {result['mean_percentage_rank']:.4f}")

        if result['mean_percentage_rank'] < best_mean_rank_percen_cfg2:
            best_mean_rank_percen_cfg2 = result['mean_percentage_rank']
            best_result_cfg2 = result
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model (from CFG2) to {BEST_MODEL_PATH}")

    print(f"--- 阶段 1 (CFG2) 训练完成 ---")
    print(f"CFG2 上的最佳结果: {best_result_cfg2}")


    # --- 4. 阶段二：加载模型并在 CFG3 上微调 ---
    print(f"\n--- 阶段 2：加载最佳模型并开始在 {DATASET_DIRS[1]} 上微调 ---")

    # 加载 cfg2 训练出的最佳模型
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"成功加载最佳模型 {BEST_MODEL_PATH}。")
    except Exception as e:
        print(f"警告：加载模型失败 ({e})。将使用 CFG2 的最终权重继续训练。")

    # （可选）重置优化器或使用更小的学习率进行微调
    # 为简单起见，我们这里重新创建优化器，这在实践中很常见
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE) 
    # 或者使用更小的学习率: lr=LEARNING_RATE / 10

    # 为 cfg3 创建新的数据加载器
    train_loader_cfg3, val_loader_cfg3 = create_dataloaders(DATASET_DIRS[1], BATCH_SIZE)

    best_mean_rank_percen_cfg3 = 1.0
    best_result_cfg3 = None

    for epoch in range(NUM_EPOCHS_CFG3):
        train_loss = train_one_epoch(model, train_loader_cfg3, optimizer, DEVICE)
        result = evaluate_gemini(model, val_loader_cfg3, DEVICE)
        recalls = result['recalls']
        
        print(f"[CFG3] Epoch {epoch+1}/{NUM_EPOCHS_CFG3}, Train Loss: {train_loss:.4f}")
        print(f"Validation Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
        print(f"Mean Rank: {result['mean_rank']:.2f}, MRR: {result['mrr']:.4f}, Mean Percentage Rank: {result['mean_percentage_rank']:.4f}")

        # 注意：这里的最佳模型是基于 *cfg3 验证集* 的表现
        if result['mean_percentage_rank'] < best_mean_rank_percen_cfg3:
            best_mean_rank_percen_cfg3 = result['mean_percentage_rank']
            best_result_cfg3 = result
            # 覆盖保存的最佳模型
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model (from CFG3) to {BEST_MODEL_PATH}")

    print(f"--- 阶段 2 (CFG3) 微调完成 ---")
    print(f"CFG3 上的最佳结果: {best_result_cfg3}")


    # --- 5. 最终评估 ---
    print(f"\n--- 阶段 3：在 CFG2 验证集上进行最终评估 ---")

    # 加载在 cfg3 训练期间保存的最终最佳模型
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"成功加载最终最佳模型 {BEST_MODEL_PATH}。")
    except Exception as e:
        print(f"错误：无法加载最终模型 {BEST_MODEL_PATH} 进行评估。{e}")
        # （根据需要处理错误）

    # 使用 cfg2 的验证加载器进行评估
    print(f"在 {DATASET_DIRS[0]} 的验证集上评估最终模型...")
    final_result_on_cfg2 = evaluate_gemini(model, val_loader_cfg2, DEVICE)

    print("\n--- 最终评估结果 (最终模型 vs CFG2 验证集) ---")
    recalls = final_result_on_cfg2['recalls']
    print(f"Recalls: R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
    print(f"Mean Rank: {final_result_on_cfg2['mean_rank']:.2f}, MRR: {final_result_on_cfg2['mrr']:.4f}, Mean Percentage Rank: {final_result_on_cfg2['mean_percentage_rank']:.4f}")


    print("\n--- 训练总结 ---")
    print(f"阶段 1 (CFG2) 最佳结果 (保存在 {BEST_MODEL_PATH} 之前): {best_result_cfg2}")
    print(f"阶段 2 (CFG3) 最佳结果 (最终保存的模型): {best_result_cfg3}")
    print(f"阶段 3 (最终模型在 CFG2 上的表现): {final_result_on_cfg2}")
