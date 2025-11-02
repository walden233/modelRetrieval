import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import json
import random
import datetime
import matplotlib.pyplot as plt

# 确保 'src' 目录在路径中
sys.path.append('.')  # 添加 src 目录到系统路径

# 假设这些模块在 'src' 目录下是可用的

from src.data import RH20TTraceDataset, collate_trajectories
from src.models import CrossModalTrajectoryModel
from src.evalution.trajectory_functions import evaluate_gemini #, evaluate
from src.loss.functions import trajectory_symmetric_contrastive_loss


# --- 训练和评估函数 (来自您的代码) ---
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    # 使用tqdm显示训练进度
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        
        human_poses = batch['human_poses'].to(device)
        human_mask = batch['human_mask'].to(device)
        tcp_bases = batch['tcp_bases'].to(device)
        tcp_mask = batch['tcp_mask'].to(device)
        human_scenes = batch['human_scene_indices'].to(device)
        robot_scenes = batch['robot_scene_indices'].to(device)

        human_embeds, robot_embeds, logit_scale = model(human_poses, human_mask, tcp_bases, tcp_mask, tcp_sample_factor=4)
        
        loss = trajectory_symmetric_contrastive_loss(human_embeds, robot_embeds, human_scenes, robot_scenes, logit_scale)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)


# --- 新增的辅助函数 ---

def sample_hyperparameters():
    """
    从定义的搜索空间中随机采样一套超参数。
    """
    # 模型参数搜索空间
    d_model = random.choice([128, 256])
    nhead = random.choice([4, 8])
    num_layers = random.choice([4, 5])
    dim_feedforward = d_model * 4  # 保持为 d_model 的 4 倍
    proj_dim = random.choice([64, 128])
    dropout = random.uniform(0.1, 0.25) # 在 0.1 到 0.3 之间采样

    model_params = {
        'human_input_dim': 21 * 3,
        'robot_input_dim': 7,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'dim_feedforward': dim_feedforward,
        'proj_dim': proj_dim,
        'dropout': dropout
    }
    
    # 训练参数搜索空间
    batch_size = random.choice([10, 24]) # 批次大小
    learning_rate = 10**random.uniform(-5, -3) # 学习率 (1e-5 到 1e-3)

    return model_params, batch_size, learning_rate

def save_trial_results(run_dir, config, history, best_result):
    """
    将单次试验的结果保存到指定目录。
    """
    # 1. 保存参数
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        # 自定义一个转换器以处理非序列化类型
        def default_converter(o):
            if isinstance(o, (torch.device, datetime.datetime)):
                return str(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        
        json.dump(config, f, indent=4, default=default_converter)
    
    # 2. 保存最终指标
    if best_result:
        with open(os.path.join(run_dir, 'best_metrics.json'), 'w') as f:
            json.dump(best_result, f, indent=4)
            
    # 3. 绘制并保存曲线
    try:
        plt.figure(figsize=(14, 6))
        
        # 绘制训练损失
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制 Mean Percentage Rank
        plt.subplot(1, 2, 2)
        plt.plot(history['val_mean_p_rank'], label='Val Mean % Rank', color='orange')
        plt.title('Validation Mean Percentage Rank')
        plt.xlabel('Epoch')
        plt.ylabel('Mean % Rank (Lower is Better)')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(f"Run: {os.path.basename(run_dir)}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(run_dir, 'curves.png'))
        plt.close() # 关闭图像，释放内存
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")


# --- 主执行流程 ---
if __name__ == '__main__':
    # 设置基本参数
    DATASET_ROOT = '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg2' 
    NUM_EPOCHS = 56 # 每次试验的 Epochs 数
    NUM_TRIALS = 20 # 总共要进行的试验次数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设置超参数搜索的根目录
    SEARCH_OUTPUT_DIR = 'hyperparam_search_results'
    os.makedirs(SEARCH_OUTPUT_DIR, exist_ok=True)
    
    # 存放全局最佳模型的路径
    BEST_MODEL_SAVE_DIR = 'model_weight'
    os.makedirs(BEST_MODEL_SAVE_DIR, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(BEST_MODEL_SAVE_DIR, 'best_trajectory_model_overall.pth')

    print(f"设备: {DEVICE}")
    print(f"数据集根目录: {DATASET_ROOT}")
    print(f"将执行 {NUM_TRIALS} 次试验，每次 {NUM_EPOCHS} 个 Epochs。")
    print(f"结果将保存在: {SEARCH_OUTPUT_DIR}")
    print(f"最佳模型将保存在: {BEST_MODEL_PATH}")
    print("-" * 30)

    # 1. 初始化数据集 (只需要执行一次)
    try:
        dataset = RH20TTraceDataset(root_dir=DATASET_ROOT)
        # 固定训练集和验证集的划分，确保所有试验都在相同的数据上进行
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                                 [train_size, val_size],
                                                                 generator=torch.Generator().manual_seed(42)) # 固定种子
        print(f"数据集加载成功: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本。")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("退出程序。")
        sys.exit(1)


    # 2. 初始化全局最佳指标追踪器
    global_best_mean_rank_percen = 1.0 # 初始为最差情况

    # 3. 开始超参数搜索循环
    for trial_idx in range(NUM_TRIALS):
        
        # --- 试验设置 ---
        run_name = f"trial_{trial_idx+1:03d}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(SEARCH_OUTPUT_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        print(f"\n--- [试验 {trial_idx+1}/{NUM_TRIALS}] ---")
        print(f"结果目录: {run_dir}")

        # 采样超参数
        model_params, batch_size, learning_rate = sample_hyperparameters()
        
        # 记录本次试验的配置
        trial_config = {
            'run_name': run_name,
            'trial_index': trial_idx + 1,
            'device': str(DEVICE),
            'num_epochs': NUM_EPOCHS,
            'model_params': model_params,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        print("试验配置:")
        print(json.dumps(trial_config, indent=2))

        # --- 试验执行 ---
        
        # 1. 初始化数据加载器 (使用当前试验的 batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_trajectories)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_trajectories)

        # 2. 初始化模型和优化器 (使用当前试验的参数)
        model = CrossModalTrajectoryModel(**model_params).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # 3. 训练和验证循环 (针对本次试验)
        history = {'train_loss': [], 'val_mean_p_rank': []}
        trial_best_result = None
        trial_best_mean_rank_percen = 1.0
        
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            
            # 使用 no_grad 进行评估
            with torch.no_grad():
                model.eval() # 切换到评估模式
                result = evaluate_gemini(model, val_loader, DEVICE)
            
            # 记录历史数据
            history['train_loss'].append(train_loss)
            history['val_mean_p_rank'].append(result['mean_percentage_rank'])

            recalls = result['recalls']
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val R@1: {recalls[1]:.4f}, R@5: {recalls[5]:.4f}, R@10: {recalls[10]:.4f}")
            print(f"  Val MRR: {result['mrr']:.4f}, Mean % Rank: {result['mean_percentage_rank']:.4f}")

            # 检查这是否是本次试验 (trial) 的最佳结果
            if result['mean_percentage_rank'] < trial_best_mean_rank_percen:
                trial_best_mean_rank_percen = result['mean_percentage_rank']
                trial_best_result = result
                print(f"  > 新的 [试验内] 最佳: Mean % Rank {trial_best_mean_rank_percen:.4f}")

                # 检查这是否是全局 (global) 最佳结果
                if trial_best_mean_rank_percen < global_best_mean_rank_percen:
                    global_best_mean_rank_percen = trial_best_mean_rank_percen
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"  >>> ★★★ 新的 [全局] 最佳模型已保存! ★★★")
                    # 同时保存全局最佳的配置
                    with open(os.path.join(BEST_MODEL_SAVE_DIR, 'best_model_params.json'), 'w') as f:
                        json.dump(trial_config, f, indent=4)


        print(f"--- 试验 {trial_idx+1} 完成 ---")
        
        # 4. 保存本次试验的日志、指标和图表
        save_trial_results(run_dir, trial_config, history, trial_best_result)
        print(f"本次试验的结果已保存至: {run_dir}")

    print("\n" + "="*30)
    print("超参数搜索已全部完成。")
    print(f"全局最佳 Mean Percentage Rank: {global_best_mean_rank_percen:.4f}")
    print(f"最佳模型权重已保存至: {BEST_MODEL_PATH}")
    print(f"最佳模型的参数配置已保存至: {os.path.join(BEST_MODEL_SAVE_DIR, 'best_model_params.json')}")