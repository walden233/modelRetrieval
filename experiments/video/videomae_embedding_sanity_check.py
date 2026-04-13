from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification,VideoMAEPreTrainedModel
import numpy as np
import torch
import torch.nn.functional as F

video = list(np.random.rand(16, 3, 224, 224))

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
inputs = processor(video, return_tensors="pt")

def get_video_embedding(video, processor, model):
    """
    输入一个视频（帧列表），返回其特征向量。
    """
    # 1. 预处理视频
    inputs = processor(video, return_tensors="pt")

    # 2. 将数据送入模型并提取特征
    with torch.no_grad():
        # 这里是关键：我们不直接调用 model(**inputs)
        # 而是调用 model.videomae(**inputs) 来获取最后一层的隐藏状态
        # 或者，更通用的方法是让模型输出所有隐藏层，然后我们取最后一层
        outputs = model(**inputs, output_hidden_states=True)
        
        # hidden_states 是一个元组，包含了所有层的输出
        # 我们通常取最后一层的输出作为特征
        last_hidden_state = outputs.hidden_states[-1]
        
        # last_hidden_state 的形状是 (batch_size, num_patches, hidden_size)
        # 我们需要一个单一的向量来代表整个视频，通常的做法是取所有 patch 特征的平均值
        # 这是一种常见的池化（Pooling）操作
        embedding = torch.mean(last_hidden_state, dim=1)

    return embedding
def calculate_similarity(embedding1, embedding2):
    """
    计算两个特征向量之间的余弦相似度。
    """
    # 使用 PyTorch 的 cosine_similarity 函数
    # F.cosine_similarity 需要至少2维的输入，所以我们用 unsqueeze(0) 增加一个维度
    # 如果你的 embedding 已经是 [1, hidden_size] 的形状，就不需要 unsqueeze
    if embedding1.ndim == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.ndim == 1:
        embedding2 = embedding2.unsqueeze(0)
        
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item() # .item() 将 tensor 转换为 python 数字


# --- 3. 创建模拟视频数据 ---
# 视频帧数、通道数、高、宽
num_frames = 16
num_channels = 3
height = 224
width = 224

# 创建视频1：一个随机视频
video1_data = np.random.rand(num_frames, height, width, num_channels) * 255
video1 = [frame.astype(np.uint8) for frame in video1_data]

# 创建视频2：与视频1非常相似（只加了微小噪声）
noise = np.random.rand(num_frames, height, width, num_channels) * 5 # 小噪声
video2_data = np.clip(video1_data + noise, 0, 255) # 加上噪声并确保值在0-255之间
video2_similar = [frame.astype(np.uint8) for frame in video2_data]

# 创建视频3：一个完全不同的随机视频
video3_data = np.random.rand(num_frames, height, width, num_channels) * 255
video3_different = [frame.astype(np.uint8) for frame in video3_data]

print("\nGenerating embeddings for videos...")
# --- 4. 提取特征向量 ---
embedding1 = get_video_embedding(video1, processor, model)
embedding2_similar = get_video_embedding(video2_similar, processor, model)
embedding3_different = get_video_embedding(video3_different, processor, model)
print("Embeddings generated.")

# --- 5. 计算并比较相似度 ---
print("\nCalculating similarities...")
similarity_1_and_2 = calculate_similarity(embedding1, embedding2_similar)
similarity_1_and_3 = calculate_similarity(embedding1, embedding3_different)

print(f"Similarity between video 1 and video 2 (similar): {similarity_1_and_2:.4f}")
print(f"Similarity between video 1 and video 3 (different): {similarity_1_and_3:.4f}")

# --- 6. 结果分析 ---
print("\n--- Analysis ---")
if similarity_1_and_2 > similarity_1_and_3:
    print("As expected, the similarity between the original video and the slightly modified one is higher.")
    print("This demonstrates that the embeddings capture the semantic content of the videos!")
else:
    print("Something unexpected happened. The similarity scores did not match expectations.")