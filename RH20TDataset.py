import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple
from transformers import VideoMAEImageProcessor

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


ALLOWED_CAMERAS = [
    'cam_f0172289',
    'cam_038522062288',
    'cam_104122063550',
    'cam_104122062295',
    'cam_104122062823',
    'cam_104422070011'
]

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

# ----------------- 数据集对象化实现 -----------------

@dataclass
class Scene:
    """用于存储单个场景信息的数据类"""
    scene_path: str
    video_pairs: List[Tuple[str, str]] = field(default_factory=list)
    human_pose_path: str = ""
    tcp_base_path: str = ""

class RH20TDataset(Dataset):
    """
    用于 RH20T 数据集的自定义 PyTorch Dataset 类。
    每个 item 对应一个 task，并从中采样指定数量的 scenes 和 cameras。
    """
    def __init__(self, root_dir, scene_num, cam_num, processor=None, num_frames=16):
        """
        Args:
            root_dir (str): 数据集根目录 (e.g., './RH20T_subset')。
            scene_num (int): 每个 task 需要采样的 scene 数量。
            cam_num (int): 每个 scene 需要采样的 camera 数量。
            processor (object, optional): 用于视频帧预处理的对象 (例如来自 huggingface 的 processor)。
            num_frames (int): 每个视频需要采样的帧数。
        """
        self.root_dir = root_dir
        self.scene_num = scene_num
        self.cam_num = cam_num
        self.processor = processor
        self.num_frames = num_frames
        
        self.tasks = self._find_tasks()

    def _find_tasks(self):
        tasks = []
        task_folders = sorted([d for d in os.listdir(self.root_dir) if d.startswith('task_')])
        
        for task_folder in task_folders:
            task_path = os.path.join(self.root_dir, task_folder)
            if not os.path.isdir(task_path):
                continue

            current_task_scenes = []
            scene_folders = sorted([d for d in os.listdir(task_path) if d.startswith('scene_')])
            
            for scene_folder in scene_folders:
                scene_path = os.path.join(task_path, scene_folder)
                
                # 创建 Scene 对象
                scene_obj = Scene(scene_path=scene_path)
                scene_obj.human_pose_path = os.path.join(scene_path, 'human_pose.npy')
                scene_obj.tcp_base_path = os.path.join(scene_path, 'tcp_base.npy')

                # 查找并配对视频文件
                human_videos = {}
                robot_videos = {}
                for f in os.listdir(scene_path):
                    if f.endswith('_human.mp4'):
                        cam_id = f.replace('_human.mp4', '')
                        if cam_id in ALLOWED_CAMERAS:
                            human_videos[cam_id] = os.path.join(scene_path, f)
                    elif f.endswith('_robot.mp4'):
                        cam_id = f.replace('_robot.mp4', '')
                        if cam_id in ALLOWED_CAMERAS:
                            robot_videos[cam_id] = os.path.join(scene_path, f)
                
                # 确保 human 和 robot 视频成对存在
                for cam_id, human_path in human_videos.items():
                    if cam_id in robot_videos:
                        robot_path = robot_videos[cam_id]
                        scene_obj.video_pairs.append((human_path, robot_path))
                
                if scene_obj.video_pairs:
                    current_task_scenes.append(scene_obj)

            if current_task_scenes:
                tasks.append(current_task_scenes)
                
        return tasks

    def __len__(self):
        """返回 task 的总数"""
        return len(self.tasks)

    def __getitem__(self, idx):
        """
        根据 task 索引 idx, 获取数据。
        返回一个包含 scene_num * cam_num 个视频对以及相应轨迹文件的字典。
        """
        task_scenes = self.tasks[idx]

        # 1. 从 task 中采样指定数量的 scene
        # 如果请求的 scene 数量超过拥有的数量，则返回所有sence
        if self.scene_num > len(task_scenes):
            print(f"警告: 请求的 scene 数量 ({self.scene_num}) 超过了 task {idx} 中的实际数量 ({len(task_scenes)})。")
            selected_scenes = task_scenes
        else:
            selected_scenes = random.sample(task_scenes, self.scene_num)

        # 准备用于收集数据的列表
        batch_human_frames = []
        batch_robot_frames = []
        batch_human_poses = []
        batch_tcp_bases = []

        for scene in selected_scenes:
            # 2. 从每个 scene 中采样指定数量的 camera 视频对
            if self.cam_num > len(scene.video_pairs):
                selected_video_pairs = scene.video_pairs
            else:
                selected_video_pairs = random.sample(scene.video_pairs, self.cam_num)
            
            # 3. 加载轨迹文件
            human_pose = np.load(scene.human_pose_path,allow_pickle=True).item()
            tcp_base = np.load(scene.tcp_base_path,allow_pickle=True).item()
            batch_human_poses.append(human_pose)
            batch_tcp_bases.append(tcp_base)
            
            # 4. 加载并采样视频帧
            for human_path, robot_path in selected_video_pairs:
                human_frames = sample_frames(human_path, self.num_frames)
                robot_frames = sample_frames(robot_path, self.num_frames)
                
                if human_frames is not None and robot_frames is not None:
                    batch_human_frames.append(human_frames)
                    batch_robot_frames.append(robot_frames)

        if not batch_human_frames: # 如果所有视频都读取失败
            return None

        # 5. 使用 processor (如果提供) 进行预处理
        # 假设 processor 能处理一个视频对的帧列表
        # 注意: 您的 processor 示例是一次处理一对 [human_frames, robot_frames]
        # 这里我们将所有视频对分别处理并收集结果
        if self.processor:
            processed_human_videos = []
            processed_robot_videos = []
            for h_frames, r_frames in zip(batch_human_frames, batch_robot_frames):          
                inputs = self.processor(
                    [h_frames, r_frames], 
                    return_tensors="pt"
                )
                processed_human_videos.append(inputs.pixel_values[0])
                processed_robot_videos.append(inputs.pixel_values[1])
            
            # 将处理后的张量堆叠成一个 batch
            # final_human_pixel_values = torch.stack(processed_human_videos)
            # final_robot_pixel_values = torch.stack(processed_robot_videos)
            final_human_pixel_values = processed_human_videos
            final_robot_pixel_values = processed_robot_videos
        else:
            raise ValueError("必须提供一个有效的 processor 用于视频帧预处理")


        return {
            "human_pixel_values": final_human_pixel_values,
            "robot_pixel_values": final_robot_pixel_values,
            "human_poses": batch_human_poses, # 返回 numpy 数组的列表
            "tcp_bases": batch_tcp_bases      # 返回 numpy 数组的列表
        }

if __name__ == '__main__':
    # ----------------- 使用示例 -----------------
    
    # # 假设你有一个像 huggingface transformers 库中的 processor
    # # 这里我们创建一个模拟的 processor 用于演示
    # class MockProcessor:
    #     def __call__(self, videos, return_tensors="pt"):
    #         # 模拟 processor 的行为：归一化、调整尺寸、转换为 PyTorch 张量
    #         # 假设输入视频帧是 (T, H, W, C) 的 numpy 数组
    #         processed_videos = []
    #         for video_frames in videos:
    #             # (T, H, W, C) -> (C, T, H, W)
    #             tensor = torch.tensor(np.array(video_frames), dtype=torch.float32).permute(3, 0, 1, 2)
    #             # 假设进行归一化
    #             tensor = tensor / 255.0
    #             processed_videos.append(tensor)
            
    #         from collections import namedtuple
    #         Output = namedtuple("Output", ["pixel_values"])
    #         return Output(pixel_values=processed_videos)

    

    DATASET_ROOT = './RH20T_subset' 
    dataset = RH20TDataset(
        root_dir=DATASET_ROOT,
        scene_num=2,
        cam_num=1,
        processor=VideoMAEImageProcessor(),
        num_frames=16 
    )

    # 4. 验证 Dataset
    print(f"发现的任务 (Task) 总数: {len(dataset)}")
    
    if len(dataset) > 0:
        # 获取第一个 task 的数据
        # 注意：这会调用 sample_frames，在真实数据上才能成功
        data_item = dataset[0] 
        
        if data_item:
            print("成功获取一个数据项 (来自 task 0)！")
            print("数据结构:")
            for key, value in data_item.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: torch.Tensor with shape {value.shape}")
                elif isinstance(value, list):
                    # 打印列表内元素的类型和数量
                    item_type = type(value[0]) if value else 'N/A'
                    print(f"  - {key}: list of {len(value)} items, item type: {item_type}")

            # 验证输出的维度
            # B, C, T, H, W (B = scene_num * cam_num = 2 * 1 = 2)
            assert len(data_item['human_pixel_values']) == 2 
            assert len(data_item['human_poses']) == 2 # scene_num = 2
            print(f"shape:{data_item['human_pixel_values'][0].shape}")
            print("\n数据维度验证通过！")
