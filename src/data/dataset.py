import torch
from torch.utils.data import Dataset
import pandas as pd
from decord import VideoReader, cpu
import numpy as np


def sample_frames(video_path, num_frames=16):
    """
    从给定的视频路径中均匀采样指定数量的帧。
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # 均匀采样帧的索引
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return list(frames)
    except Exception as e:
        print(f"Error reading video file {video_path}: {e}")
        return None


class WhirlDataset(Dataset):
    def __init__(self, csv_file, processor, num_frames=16):
        self.metadata = pd.read_csv(csv_file)
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # 加载并采样视频帧
        human_frames = sample_frames(row['human_video_path'], self.num_frames)
        robot_frames = sample_frames(row['robot_video_path'], self.num_frames)

        # 使用 processor 进行预处理
        # 它会处理归一化、尺寸调整等
        inputs = self.processor(
            [human_frames, robot_frames], 
            return_tensors="pt"
        )
        
        # 返回成对的视频数据
        return {
            "human_pixel_values": inputs.pixel_values[0],
            "robot_pixel_values": inputs.pixel_values[1]
        }