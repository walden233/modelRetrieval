from typing import Any, Dict, List

import torch


def extract_pixel_values(processor_outputs: Dict[str, Any]) -> torch.Tensor:
    # HuggingFace 不同视频处理器返回字段名不完全一致，这里做一层兼容。
    pixel_values = processor_outputs.get("pixel_values")
    if pixel_values is None:
        pixel_values = processor_outputs.get("pixel_values_videos")
    if pixel_values is None:
        raise KeyError("processor output must contain `pixel_values` or `pixel_values_videos`.")
    return pixel_values


def collate_video_pairs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Dataset 可能因为视频损坏返回 None，这里先过滤掉无效样本。
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        raise ValueError("Received an empty batch after filtering invalid video samples.")

    # 这里把“一个样本 = 一对 human/robot 视频”的字典列表，
    # 拼成训练器/评估器可以直接消费的 batch 字典。
    return {
        "human_pixel_values": torch.stack([item["human_pixel_values"] for item in valid_batch], dim=0),
        "robot_pixel_values": torch.stack([item["robot_pixel_values"] for item in valid_batch], dim=0),
        "task_indices": torch.tensor([item["task_index"] for item in valid_batch], dtype=torch.long),
        "scene_indices": torch.tensor([item["scene_index"] for item in valid_batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in valid_batch],
        "pair_ids": [item["pair_id"] for item in valid_batch],
        "dataset_names": [item["dataset_name"] for item in valid_batch],
        "task_ids": [item["task_id"] for item in valid_batch],
        "scene_ids": [item["scene_id"] for item in valid_batch],
        "camera_ids": [item["camera_id"] for item in valid_batch],
        "human_video_paths": [item["human_video_path"] for item in valid_batch],
        "robot_video_paths": [item["robot_video_path"] for item in valid_batch],
        "metadata": [item.get("metadata", {}) for item in valid_batch],
    }
