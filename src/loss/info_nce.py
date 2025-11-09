from torch import nn
import matplotlib.pyplot as plt
import torch

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, human_features, robot_features):
        # human_features, robot_features: [batch_size, feature_dim]
        batch_size = human_features.shape[0]
        device = human_features.device
        
        # 计算相似度矩阵
        # human_features @ robot_features.T -> [batch_size, batch_size]
        logits = (human_features @ robot_features.T) / self.temperature
        
        # # 可视化
        # logits_np = logits.detach().cpu().numpy()
        # plt.figure(figsize=(6, 5))
        # plt.imshow(logits_np, cmap="viridis", aspect="auto")  # 热力图
        # plt.colorbar(label="Similarity")
        # plt.title("Logits Heatmap (Human vs Robot Features)")
        # plt.xlabel("Robot Index")
        # plt.ylabel("Human Index")
        # plt.show()

        # 正样本在对角线上，所以我们的目标标签是 [0, 1, 2, ..., batch_size-1]
        labels = torch.arange(batch_size, device=device)
        
        # 计算对称损失
        loss_human_vs_robot = self.criterion(logits, labels)
        loss_robot_vs_human = self.criterion(logits.T, labels)
        
        return (loss_human_vs_robot + loss_robot_vs_human) / 2