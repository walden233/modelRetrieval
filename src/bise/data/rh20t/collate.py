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
    human_scene_ids = []
    robot_scene_ids = []
    human_task_ids = []
    robot_task_ids = []
    human_camera_ids = []
    robot_camera_ids = []
    human_scene_paths = []
    robot_scene_paths = []
    human_times, robot_times, human_observed, human_frame_indices = [], [], [], []
    max_robot_len = 999

    for item in batch:
        task_idx = item.get("task_idx")
        if task_idx is None or task_idx < 0:
            task_idx = -(item["scene_idx"] + 1)

        all_human_poses.extend(item["human_poses"])
        human_scene_indices.extend([item["scene_idx"]] * len(item["human_poses"]))
        human_task_indices.extend([task_idx] * len(item["human_poses"]))
        human_scene_ids.extend([item.get("scene_id", str(item["scene_idx"]))] * len(item["human_poses"]))
        human_task_ids.extend([item.get("task_id", str(task_idx))] * len(item["human_poses"]))
        human_camera_ids.extend(item.get("camera_ids", [""] * len(item["human_poses"])))
        human_scene_paths.extend([item.get("scene_path", "")] * len(item["human_poses"]))

        for i, trajectory in enumerate(item["human_poses"]):
            human_times.append(item.get("human_times", [])[i] if "human_times" in item
                               else torch.arange(len(trajectory), dtype=torch.float32))
            human_observed.append(item.get("human_observed", [])[i] if "human_observed" in item
                                  else torch.ones(len(trajectory), dtype=torch.bool))
            human_frame_indices.append(item.get("human_frame_indices", [])[i]
                                       if "human_frame_indices" in item
                                       else torch.arange(len(trajectory)))
        sampled_tcp_bases = []
        for i, trajectory in enumerate(item["tcp_bases"]):
            original_length = len(trajectory)
            if original_length == 0:
                raise ValueError("Cannot collate an empty robot trajectory")

            stride = (1 if item.get("sampled_once") else
                      max(1, int(math.ceil(original_length / max_robot_len))))
            times = (item["robot_times"][i] if "robot_times" in item
                     else torch.arange(original_length, dtype=torch.float32))
            robot_times.append(times[::stride])
            sampled_tcp_bases.append(trajectory[::stride])

        all_tcp_bases.extend(sampled_tcp_bases)
        robot_scene_indices.extend([item["scene_idx"]] * len(item["tcp_bases"]))
        robot_task_indices.extend([task_idx] * len(item["tcp_bases"]))
        robot_scene_ids.extend([item.get("scene_id", str(item["scene_idx"]))] * len(item["tcp_bases"]))
        robot_task_ids.extend([item.get("task_id", str(task_idx))] * len(item["tcp_bases"]))
        robot_camera_ids.extend(item.get("camera_ids", [""] * len(item["tcp_bases"])))
        robot_scene_paths.extend([item.get("scene_path", "")] * len(item["tcp_bases"]))

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
        "human_times": pad_sequence(human_times, batch_first=True),
        "robot_times": pad_sequence(robot_times, batch_first=True),
        "human_observed": pad_sequence(human_observed, batch_first=True, padding_value=False),
        "human_frame_indices": pad_sequence(human_frame_indices, batch_first=True, padding_value=-1),
        "human_poses": padded_human_poses,
        "human_mask": human_mask,
        "tcp_bases": padded_tcp_bases,
        "tcp_mask": tcp_mask,
        "human_scene_indices": torch.tensor(human_scene_indices, dtype=torch.long),
        "robot_scene_indices": torch.tensor(robot_scene_indices, dtype=torch.long),
        "human_task_indices": torch.tensor(human_task_indices, dtype=torch.long),
        "robot_task_indices": torch.tensor(robot_task_indices, dtype=torch.long),
        "human_scene_ids": human_scene_ids,
        "robot_scene_ids": robot_scene_ids,
        "human_task_ids": human_task_ids,
        "robot_task_ids": robot_task_ids,
        "human_camera_ids": human_camera_ids,
        "robot_camera_ids": robot_camera_ids,
        "human_scene_paths": human_scene_paths,
        "robot_scene_paths": robot_scene_paths,
    }


def trajectory_context(batch: dict, domain: str, device) -> torch.Tensor | None:
    """Progress, log elapsed interval, observation flag; units follow dataset times."""
    key = "human_times" if domain == "human" else "robot_times"
    if key not in batch:
        return None
    times = batch[key].to(device)
    mask = batch["human_mask" if domain == "human" else "tcp_mask"].to(device)
    duration = times.masked_fill(~mask, 0).amax(dim=1, keepdim=True).clamp_min(1e-6)
    progress = times / duration
    delta = torch.zeros_like(times)
    delta[:, 1:] = (times[:, 1:] - times[:, :-1]).clamp_min(0)
    observed = batch["human_observed"].to(device) if domain == "human" else mask
    context = torch.stack([progress, torch.log1p(delta), observed.float()], dim=-1)
    return context.masked_fill(~mask.unsqueeze(-1), 0)


def trajectory_context_kwargs(batch: dict, device) -> dict:
    return {"human_context": trajectory_context(batch, "human", device),
            "robot_context": trajectory_context(batch, "robot", device)}
