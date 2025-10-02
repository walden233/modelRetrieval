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

from src.data import RH20TVideoDataset, WhirlDataset 
from src.models import VideomaeFineTuner, vjepaFineTuner, InfoNCELoss
from src.evalution import calculate_retrieval_metrics, calculate_retrieval_metrics_grouped


if __name__ == "__main__":
    BATCH_SIZE = 4  # 可以设置得比训练时大一些，因为不需要存储梯度
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
    # model = VideomaeFineTuner(model,hidden_size,FEATURE_DIM).to(device)

    # vjepa2
    MODEL_NAME = "facebook/vjepa2-vitl-fpc64-256"
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME,trust_remote_code=True)
    hidden_size = model.config.hidden_size 
    model = vjepaFineTuner(model,hidden_size,FEATURE_DIM).to(device)




    DATASET_ROOT = './dataset/RH20T_subset/RH20T_cfg3' 
    SCENE_NUM = 1
    CAM_NUM = 1
    dataset = RH20TVideoDataset(
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
