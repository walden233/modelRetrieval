import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor,AutoVideoProcessor
from tqdm.auto import tqdm
import pandas as pd
from decord import VideoReader, cpu
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rh20t import RH20TDataset
from dataset import HumanRobotDataset
from network import ContrastiveFineTuner, vjepaFineTuner
from loss import InfoNCELoss

# --- 设置参数 ---

BATCH_SIZE = 2  # 可以设置得比训练时大一些，因为不需要存储梯度
FEATURE_DIM = 256
TEMPERATURE = 0.07

# --- 初始化 ---
print("Initializing components...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # VideoMAE
# MODEL_NAME = "OpenGVLab/VideoMAEv2-Large"
# processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME,trust_remote_code=True)
# hidden_size = model.model_config["embed_dim"] 
# model = ContrastiveFineTuner(model,hidden_size,FEATURE_DIM).to(device)

# vjepa2
MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME,trust_remote_code=True)
hidden_size = model.config.hidden_size 
model = vjepaFineTuner(model,hidden_size,FEATURE_DIM).to(device)




DATASET_ROOT = './RH20T_subset/RH20T_cfg2' 
SCENE_NUM = 1
CAM_NUM = 1
dataset = RH20TDataset(
        root_dir=DATASET_ROOT,
        scene_num=SCENE_NUM,
        cam_num=CAM_NUM,
        processor=processor,
        num_frames=16 
    )

# 在评估时，shuffle=False 可以确保每次运行结果一致
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
loss_fn = InfoNCELoss(temperature=TEMPERATURE)

# --- 开始评估 ---
print("Starting baseline evaluation...")
# **关键**：将模型设置为评估模式
# 这会关闭 Dropout 等层，确保结果的一致性
model.eval()

total_loss = 0.0
total_batches = 0

# 用于存储所有特征的列表
all_human_features_list = []
all_robot_features_list = []

# **关键**：使用 torch.no_grad() 上下文管理器
# 这会告诉 PyTorch 不需要计算梯度，从而节省大量内存和计算资源
with torch.no_grad():
    progress_bar = tqdm(dataloader, desc="Evaluating Baseline")
    for batch in progress_bar:
        # human_videos = batch["human_pixel_values"].to(device)
        # robot_videos = batch["robot_pixel_values"].to(device)
        
        # all_videos = torch.cat([human_videos, robot_videos], dim=0)

        human_videos = [t.to(device) for t in batch["human_pixel_values"]]
        robot_videos = [t.to(device) for t in batch["robot_pixel_values"]]
        
        all_videos = torch.cat(human_videos + robot_videos, dim=0)
        all_features = model(all_videos)

        human_features, robot_features = torch.chunk(all_features, 2, dim=0)
        all_human_features_list.append(human_features.cpu())
        all_robot_features_list.append(robot_features.cpu())

        loss = loss_fn(human_features, robot_features)
        
        # 累加损失值
        # .item() 将 tensor 转换为 python 数字
        total_loss += loss.item()
        print("batch:",total_batches,"  loss:",loss.item())
        total_batches += 1
        progress_bar.set_postfix({"current_batch_loss": loss.item()})

all_human_features = torch.cat(all_human_features_list, dim=0)
all_robot_features = torch.cat(all_robot_features_list, dim=0)

similarity_matrix = torch.matmul(all_human_features, all_robot_features.T)


# 5. Visualization: Use a heatmap to visualize the matrix
# Convert the tensor to a NumPy array for plotting
similarity_matrix_np = similarity_matrix.numpy()


def calculate_retrieval_metrics(similarity_matrix):
    """
    根据相似度矩阵计算检索评估指标。
    假设第 i 行对应第 i 列是正样本。
    """
    num_queries = similarity_matrix.shape[0]
    ranks = []

    for i in range(num_queries):
        # 获取当前人类视频与所有机器人视频的相似度分数
        scores = similarity_matrix[i, :]
        
        # 获取正样本（正确配对）的分数
        ground_truth_score = scores[i]
        
        # 对所有分数进行降序排序
        sorted_scores = np.sort(scores)[::-1]
        
        # 找到正样本分数在排序后的列表中的位置（排名）
        # np.where 返回一个元组，我们需要第一个数组的第一个元素
        # 排名从1开始，所以需要 +1
        rank = np.where(sorted_scores == ground_truth_score)[0][0] + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    # 计算各项指标
    r_at_1 = np.mean(ranks <= 1)
    r_at_5 = np.mean(ranks <= 5)
    r_at_10 = np.mean(ranks <= 10)
    mean_rank = np.mean(ranks)
    mrr = np.mean(1.0 / ranks)
    
    metrics = {
        "R@1": r_at_1,
        "R@5": r_at_5,
        "R@10": r_at_10,
        "Mean Rank": mean_rank,
        "MRR": mrr
    }
    
    return metrics

def calculate_retrieval_metrics_grouped(similarity_matrix, group_size):
    """
    根据相似度矩阵计算检索评估指标，支持分组的正样本。
    
    Args:
        similarity_matrix (np.array): (N, N) 的相似度矩阵。
        group_size (int): 每个正样本组的大小 (n)。
    """
    num_queries = similarity_matrix.shape[0]
    if num_queries % group_size != 0:
        raise ValueError("The total number of samples must be divisible by group_size.")
        
    # 用于存储每个查询的最高排名正样本的排名
    ranks_of_best_positive = []

    # 遍历每个人类视频（查询）
    for i in range(num_queries):
        # 1. 确定当前查询所属的组和该组的正样本索引范围
        group_start_idx = (i // group_size) * group_size
        positive_indices = set(range(group_start_idx, group_start_idx + group_size))
        
        # 2. 获取当前查询与所有机器人视频的相似度分数
        scores = similarity_matrix[i, :]
        
        # 3. 获取按分数降序排列的机器人视频的 *索引*
        sorted_candidate_indices = np.argsort(-scores)
        
        # 4. 找到第一个出现在排序列表中的正样本，记录其排名
        best_rank_for_query = -1
        # enumerate 从 0 开始，所以 rank 是 0-based，我们需要 +1
        for rank, candidate_idx in enumerate(sorted_candidate_indices):
            if candidate_idx in positive_indices:
                best_rank_for_query = rank + 1
                break  # 找到后即可停止
        
        # 如果因为某些原因没有找到（理论上不应发生），可以跳过
        if best_rank_for_query != -1:
            ranks_of_best_positive.append(best_rank_for_query)
            
    ranks = np.array(ranks_of_best_positive)
    
    # 5. 基于最高排名正样本的排名列表，计算各项指标
    r_at_1 = np.mean(ranks <= 1)
    r_at_5 = np.mean(ranks <= 5)
    r_at_10 = np.mean(ranks <= 10)
    mean_rank = np.mean(ranks)
    mrr = np.mean(1.0 / ranks)
    
    metrics = {
        "R@1": r_at_1,
        "R@5": r_at_5,
        "R@10": r_at_10,
        "Mean Best Positive Rank": mean_rank, 
        "MRR": mrr
    }
    
    return metrics

print("\n--- Retrieval Performance Metrics ---")
# retrieval_results = calculate_retrieval_metrics(similarity_matrix_np)
retrieval_results = calculate_retrieval_metrics_grouped(similarity_matrix_np, SCENE_NUM*CAM_NUM)
for name, value in retrieval_results.items():
    if "Rank" in name:
        print(f"{name:<12}: {value:.2f}") # 排名保留两位小数
    else:
        print(f"{name:<12}: {value:.4f}") # R@K 和 MRR 保留四位小数


plt.figure(figsize=(15, 12))
sns.heatmap(similarity_matrix_np, cmap='viridis') # 'viridis', 'coolwarm', 'YlGnBu' are good colormaps
plt.title("Cosine Similarity between Human and Robot Video Features")
plt.xlabel("Robot Videos")
plt.ylabel("Human Videos")
plt.show()

# print(f"Shape of all human features: {all_human_features.shape}")
# print(f"Shape of all robot features: {all_robot_features.shape}")

# --- 报告结果 ---
average_loss = total_loss / total_batches
print("\n--- Baseline Evaluation Complete ---")
print(f"Total data pairs evaluated: {len(dataset)}")
print(f"Number of batches: {total_batches}")
print(f"Average InfoNCE Loss on pre-trained model: {average_loss:.4f}")