import math

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_trajectories(batch):
    all_human_poses = []
    all_tcp_bases = []
    human_scene_indices = []
    robot_scene_indices = []
    human_task_indices = []
    robot_task_indices = []
    max_robot_len = 999

    for item in batch:
        task_idx = item.get("task_idx")
        if task_idx is None or task_idx < 0:
            task_idx = -(item["scene_idx"] + 1)

        all_human_poses.extend(item["human_poses"])
        human_scene_indices.extend([item["scene_idx"]] * len(item["human_poses"]))
        human_task_indices.extend([task_idx] * len(item["human_poses"]))

        sampled_tcp_bases = []
        for trajectory in item["tcp_bases"]:
            original_length = len(trajectory)
            if original_length == 0:
                sampled_tcp_bases.append(trajectory)
                continue

            stride = max(1, int(math.ceil(original_length / max_robot_len)))
            sampled_tcp_bases.append(trajectory[::stride])

        all_tcp_bases.extend(sampled_tcp_bases)
        robot_scene_indices.extend([item["scene_idx"]] * len(item["tcp_bases"]))
        robot_task_indices.extend([task_idx] * len(item["tcp_bases"]))

    human_lengths = [len(trajectory) for trajectory in all_human_poses]
    tcp_lengths = [len(trajectory) for trajectory in all_tcp_bases]
    padded_human_poses = pad_sequence(all_human_poses, batch_first=True, padding_value=0.0)
    padded_tcp_bases = pad_sequence(all_tcp_bases, batch_first=True, padding_value=0.0)

    human_mask = torch.arange(padded_human_poses.size(1))[None, :] < torch.tensor(human_lengths)[:, None]
    tcp_mask = torch.arange(padded_tcp_bases.size(1))[None, :] < torch.tensor(tcp_lengths)[:, None]

#   - human_poses: 补齐后的人手轨迹张量，形状大致是 [N_human, T_h, K, 3]
#   - human_mask: 人手轨迹的有效时间步掩码，True 表示真实帧，False 表示 padding
#   - tcp_bases: 补齐后的机器人 TCP 轨迹张量，形状大致是 [N_robot, T_r, 7]
#   - tcp_mask: 机器人轨迹的有效时间步掩码
#   - human_scene_indices: 每条 human 轨迹属于哪个 scene
#   - robot_scene_indices: 每条 robot 轨迹属于哪个 scene
#   - human_task_indices: 每条 human 轨迹属于哪个 task
#   - robot_task_indices: 每条 robot 轨迹属于哪个 task

#   这些索引的用途是：
#   - 训练时，如果按 scene 定义正样本，就用 *_scene_indices
#   - 训练时，如果按 task 定义正样本，就用 *_task_indices
    return {
        "human_poses": padded_human_poses,
        "human_mask": human_mask,
        "tcp_bases": padded_tcp_bases,
        "tcp_mask": tcp_mask,
        "human_scene_indices": torch.tensor(human_scene_indices, dtype=torch.long),
        "robot_scene_indices": torch.tensor(robot_scene_indices, dtype=torch.long),
        "human_task_indices": torch.tensor(human_task_indices, dtype=torch.long),
        "robot_task_indices": torch.tensor(robot_task_indices, dtype=torch.long),
    }
