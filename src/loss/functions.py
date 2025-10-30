import torch.nn.functional as F
def trajectory_symmetric_contrastive_loss(human_embeds, robot_embeds, human_scenes, robot_scenes, logit_scale):
    """
    计算对称式对比损失。
    Args:
        human_embeds: (N, D) 形状的人类轨迹嵌入
        robot_embeds: (M, D) 形状的机器人轨迹嵌入
        human_scenes: (N,) 形状，表示每个人类轨迹所属的场景索引
        robot_scenes: (M,) 形状，表示每个机器人轨迹所属的场景索引
        logit_scale: 温度参数
    """
    # 计算相似度矩阵
    logits_per_human = logit_scale * human_embeds @ robot_embeds.t()
    logits_per_robot = logits_per_human.t()

    # 创建目标标签
    # 如果 human_i 和 robot_j 来自同一个场景，则它们是正样本对
    # (N, M)
    labels = (human_scenes.unsqueeze(1) == robot_scenes.unsqueeze(0)).float().to(logits_per_human.device)
    
    # 由于一个场景有多个正样本，我们需要对标签进行归一化
    labels_h_r = labels / labels.sum(dim=1, keepdim=True).clamp(min=1.0)
    labels_r_h = labels.t() / labels.t().sum(dim=1, keepdim=True).clamp(min=1.0)

    # 计算交叉熵损失 (使用 KL 散度形式以支持软标签)
    loss_h_r = F.kl_div(F.log_softmax(logits_per_human, dim=1), labels_h_r, reduction='batchmean')
    loss_r_h = F.kl_div(F.log_softmax(logits_per_robot, dim=1), labels_r_h, reduction='batchmean')

    return (loss_h_r + loss_r_h) / 2