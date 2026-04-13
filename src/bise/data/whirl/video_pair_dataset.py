import pandas as pd
from torch.utils.data import Dataset

from bise.data.rh20t.scanner import sample_video_frames


class VideoPairDataset(Dataset):
    def __init__(self, csv_file: str, processor, num_frames: int = 16):
        self.metadata = pd.read_csv(csv_file)
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        human_frames = sample_video_frames(row["human_video_path"], self.num_frames)
        robot_frames = sample_video_frames(row["robot_video_path"], self.num_frames)
        inputs = self.processor([human_frames, robot_frames], return_tensors="pt")
        pixel_values = inputs.get("pixel_values") or inputs.get("pixel_values_videos")
        return {
            "human_pixel_values": pixel_values[0],
            "robot_pixel_values": pixel_values[1],
        }
