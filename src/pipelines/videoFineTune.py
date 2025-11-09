from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler, VideoMAEImageProcessor
from tqdm.auto import tqdm
import torch

# 修改导入路径
from src.data import WhirlDataset
from src.models import VideomaeFineTuner
from src.loss import InfoNCELoss

# --- 1. 设置超参数 ---
MODEL_NAME = "OpenGVLab/VideoMAEv2-Large"
CSV_PATH = "dataset.csv"
NUM_EPOCHS = 5
BATCH_SIZE = 4  # 根据你的GPU显存调整
LEARNING_RATE = 1e-5
FEATURE_DIM = 128 # 投影头的输出维度
TEMPERATURE = 0.07 # InfoNCE的温度参数

# --- 2. 初始化 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
dataset = WhirlDataset(csv_file=CSV_PATH, processor=processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VideomaeFineTuner(model_name=MODEL_NAME, feature_dim=FEATURE_DIM).to(device)
loss_fn = InfoNCELoss(temperature=TEMPERATURE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# --- 3. 训练循环 ---
progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(NUM_EPOCHS):
    for batch in dataloader:
        # 将数据移到GPU
        human_videos = batch["human_pixel_values"].to(device)
        robot_videos = batch["robot_pixel_values"].to(device)
        
        # 将两种视频拼接在一起，一次性通过模型，提高效率
        all_videos = torch.cat([human_videos, robot_videos], dim=0)
        
        # 前向传播
        all_features = model(all_videos)
        
        # 将特征拆分回人类和机器人
        human_features, robot_features = torch.chunk(all_features, 2, dim=0)
        
        # 计算损失
        loss = loss_fn(human_features, robot_features)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})

# --- 4. 保存微调后的编码器 ---
print("训练完成！正在保存模型...")
# **重要**：我们只保存骨干网络（编码器）的权重，因为投影头在推理时不需要
torch.save(model.videomae.state_dict(), "finetuned_videomae_encoder.pth")
