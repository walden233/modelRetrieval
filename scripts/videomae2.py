from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification, AutoModel, AutoConfig
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu # 引入 decord



# --- 模型和处理器加载 (与之前相同) ---
print("Loading model and processor...")
# processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
# model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
# model.eval()

# config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-giant", trust_remote_code=True)
# processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-giant")
# model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-giant', config=config, trust_remote_code=True)

config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Large", trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Large")
model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Large', config=config, trust_remote_code=True)
print("Model loaded.")

# --- 辅助函数 (与之前相同) ---
def get_video_embedding(video_frames, processor, model):
    inputs = processor(video_frames, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs

def calculate_similarity(embedding1, embedding2):
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# --- 新增：视频加载函数 ---
def load_video(filepath, num_frames=16):
    """
    从文件路径加载视频，并均匀采样指定数量的帧。
    """
    vr = VideoReader(filepath, ctx=cpu(0))
    total_frames = len(vr)
    # 计算采样间隔
    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    # 读取指定帧
    frames = vr.get_batch(indices).asnumpy()
    # 转换为 list of numpy arrays
    return [frame for frame in frames]

# --- 主逻辑：使用真实视频进行测试 ---
try:
    print("\nLoading real videos...")
    # 加载两个内容完全不同的视频
    video_sports = load_video("whirl_videos/human/hat.mp4")
    video_cooking = load_video("whirl_videos/robot/hat.mp4")
    
    # 为了模拟相似视频，我们可以对 sports.mp4 做一点轻微的修改，比如只取稍微不同的帧
    vr_sports = VideoReader("whirl_videos/human/can.mp4", ctx=cpu(0))
    indices_similar = np.linspace(1, len(vr_sports) - 2, num=16, dtype=int) # 采样时稍微错开
    frames_similar = vr_sports.get_batch(indices_similar).asnumpy()
    video_sports_similar = [frame for frame in frames_similar]

    print("Generating embeddings for real videos...")
    # 提取特征向量
    embedding_sports = get_video_embedding(video_sports, processor, model)
    embedding_sports_similar = get_video_embedding(video_sports_similar, processor, model)
    embedding_cooking = get_video_embedding(video_cooking, processor, model)
    print("Embeddings generated.")

    # 计算并比较相似度
    print("\nCalculating similarities...")
    similarity_sports_vs_similar = calculate_similarity(embedding_sports, embedding_sports_similar)
    similarity_sports_vs_cooking = calculate_similarity(embedding_sports, embedding_cooking)

    print(f"Similarity between two similar sports videos: {similarity_sports_vs_similar:.4f}")
    print(f"Similarity between a sports video and a cooking video: {similarity_sports_vs_cooking:.4f}")

    # 结果分析
    print("\n--- Analysis ---")
    if similarity_sports_vs_similar > similarity_sports_vs_cooking:
        print("As expected, the similarity between two related videos is much higher than between two unrelated videos.")
        print("This confirms the model is working correctly on real-world data!")
    else:
        print("Something unexpected happened. The similarity scores did not match expectations.")

except FileNotFoundError:
    print("\nError: Video files not found. Please download 'sports.mp4' and 'cooking.mp4' to the same directory.")