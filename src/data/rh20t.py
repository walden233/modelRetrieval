import os
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
# from transformers import VideoMAEImageProcessor # 取消注释以在实际项目中使用

# ----------------- 辅助函数和数据类 (保持不变) -----------------

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

@dataclass
class Scene:
    """用于存储单个场景信息的数据类"""
    scene_path: str
    video_pairs: List[Tuple[str, str]] = field(default_factory=list)
    human_pose_path: str = ""
    tcp_base_path: str = ""


# ----------------- 1. 共享逻辑的基类 -----------------

class _RH20TBaseDataset(Dataset):
    """
    RH20T 数据集的内部基类。
    
    主要功能是扫描数据集目录，构建一个包含所有 task 和 scene 信息的结构。
    这个类不应该被直接实例化。
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.tasks = self._find_tasks()

    def _find_tasks(self) -> List[List[Scene]]:
        """扫描根目录，找到所有 tasks 及其包含的 scenes。"""
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
                
                scene_obj = Scene(scene_path=scene_path)
                scene_obj.human_pose_path = os.path.join(scene_path, 'human_pose.npy')
                scene_obj.tcp_base_path = os.path.join(scene_path, 'tcp_base.npy')

                human_videos, robot_videos = {}, {}
                for f in os.listdir(scene_path):
                    if f.endswith('_human.mp4'):
                        cam_id = f.replace('_human.mp4', '')
                        human_videos[cam_id] = os.path.join(scene_path, f)
                    elif f.endswith('_robot.mp4'):
                        cam_id = f.replace('_robot.mp4', '')
                        robot_videos[cam_id] = os.path.join(scene_path, f)
                
                for cam_id, human_path in human_videos.items():
                    if cam_id in robot_videos:
                        scene_obj.video_pairs.append((human_path, robot_videos[cam_id]))
                
                # 只有当场景包含有效数据时才添加
                # or (os.path.exists(scene_obj.human_pose_path) and os.path.exists(scene_obj.tcp_base_path))
                if scene_obj.video_pairs:
                    current_task_scenes.append(scene_obj)

            if current_task_scenes:
                tasks.append(current_task_scenes)
                
        return tasks

    def __len__(self) -> int:
        """返回 task 的总数"""
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> Dict:
        """子类必须实现这个方法。"""
        raise NotImplementedError("子类必须实现 __getitem__ 方法")


# ----------------- 2. 只处理人机视频对的数据集 -----------------

class RH20TVideoDataset(_RH20TBaseDataset):
    """
    用于 RH20T 数据集的视频对 Dataset。
    每个 item 对应一个 task，从中采样 scenes 和 cameras 的视频数据。
    """
    def __init__(self, root_dir: str, scene_num: int, cam_num: int, processor, num_frames: int = 16):
        super().__init__(root_dir)
        self.scene_num = scene_num
        self.cam_num = cam_num
        self.processor = processor
        self.num_frames = num_frames
        
        if self.processor is None:
            raise ValueError("必须提供一个有效的 processor 用于视频帧预处理")

    def __getitem__(self, idx: int) -> Dict:
        task_scenes = self.tasks[idx]

        # 1. 采样 scenes
        if self.scene_num >= len(task_scenes):
            selected_scenes = task_scenes
        else:
            selected_scenes = random.sample(task_scenes, self.scene_num)

        batch_human_frames, batch_robot_frames = [], []

        for scene in selected_scenes:
            if not scene.video_pairs:
                continue

            # 2. 采样 cameras
            if self.cam_num >= len(scene.video_pairs):
                selected_video_pairs = scene.video_pairs
            else:
                selected_video_pairs = random.sample(scene.video_pairs, self.cam_num)
            
            # 3. 加载并采样视频帧
            for human_path, robot_path in selected_video_pairs:
                human_frames = sample_frames(human_path, self.num_frames)
                robot_frames = sample_frames(robot_path, self.num_frames)
                
                if human_frames is not None and robot_frames is not None:
                    batch_human_frames.append(human_frames)
                    batch_robot_frames.append(robot_frames)

        if not batch_human_frames:
            # 如果所有视频都读取失败，可以返回一个空字典或递归调用自身处理下一个样本
            print(f"警告: Task {idx} 中所有选定视频均读取失败。")
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 0 else None

        # 4. 预处理
        processed_human, processed_robot = [], []
        for h_frames, r_frames in zip(batch_human_frames, batch_robot_frames):
            inputs = self.processor([h_frames, r_frames], return_tensors="pt")
            
            pixel_values = inputs.get('pixel_values') or inputs.get('pixel_values_videos')
            if pixel_values is not None:
                processed_human.append(pixel_values[0])
                processed_robot.append(pixel_values[1])

        return {
            "human_pixel_values": processed_human,
            "robot_pixel_values": processed_robot,
            'task_idx': idx
        }

# ----------------- 3. 只处理人机轨迹对的数据集 -----------------

class RH20TTraceDataset(Dataset):
    """
    用于 RH20T 数据集的轨迹对 Dataset。

    - 每个 item 对应一个 scene。
    - 初始化时，扫描并加载所有有效的 scene 路径。
    - __getitem__ 返回该 scene 内，按 camera_id 聚合的轨迹对。
    
    (修改): 
    - 增加了 use_6_keypoints 参数，
      用于在 __getitem__ 中直接将21个人手关键点采样为6个。
    """
    
    def __init__(self, root_dir: str, use_6_keypoints: bool = False):
        """
        初始化 Dataset。
        
        Args:
            root_dir (str): 数据集根目录。
            use_6_keypoints (bool, optional): 
                是否将21个人手关键点采样为6个 (手腕+5指尖)。
                默认为 True。
        """
        self.root_dir = root_dir
        self.use_6_keypoints = use_6_keypoints
        
        # 0=手腕, 4=拇指尖, 8=食指尖, 12=中指尖, 16=无名指尖, 20=小指尖
        self.KEYPOINT_INDICES = [0, 4, 8, 12, 16, 20]
        
        if self.use_6_keypoints:
            print(f"RH20TTraceDataset: [启用] 6关键点采样 (手腕 + 5个指尖)。")
        else:
            print(f"RH20TTraceDataset: [禁用] 6关键点采样 (使用全部21个关键点)。")

        # self.scenes 是一个扁平化的列表，包含所有有效的 Scene 对象
        self.scenes = self._find_all_valid_scenes()
        
        if not self.scenes:
            raise ValueError(f"在目录 {root_dir} 中没有找到任何有效的场景。")

    def _find_all_valid_scenes(self) -> List[Scene]:
        """扫描根目录，找到所有包含有效轨迹文件的 scenes。"""
        all_scenes = []
        task_folders = sorted([d for d in os.listdir(self.root_dir) if d.startswith('task_')])
        
        for task_folder in task_folders:
            task_path = os.path.join(self.root_dir, task_folder)
            if not os.path.isdir(task_path):
                continue

            scene_folders = sorted([d for d in os.listdir(task_path) if d.startswith('scene_')])
            
            for scene_folder in scene_folders:
                scene_path = os.path.join(task_path, scene_folder)
                
                human_pose_path = os.path.join(scene_path, 'human_pose.npy')
                tcp_base_path = os.path.join(scene_path, 'tcp_base.npy')

                # 预先检查文件是否存在，只将有效的 scene 加入列表
                if os.path.exists(human_pose_path) and os.path.exists(tcp_base_path):
                    scene_obj = Scene(
                        scene_path=scene_path,
                        human_pose_path=human_pose_path,
                        tcp_base_path=tcp_base_path
                    )
                    all_scenes.append(scene_obj)
        
        print(f"初始化完成：共找到 {len(all_scenes)} 个有效场景。")
        return all_scenes

    def __len__(self) -> int:
        """返回数据集中有效场景的总数。"""
        return len(self.scenes)
    
    def __getitem__(self, idx: int) -> Dict[str, List[torch.Tensor]]:
        """
        根据场景索引 idx，加载并处理该场景的轨迹数据。
        返回一个字典，其中包含按 camera_id 对齐的轨迹张量列表。
        """
        scene = self.scenes[idx]
        
        try:
            human_pose_dict = np.load(scene.human_pose_path, allow_pickle=True).item()
            tcp_base_dict = np.load(scene.tcp_base_path, allow_pickle=True).item()
        except Exception as e:
            print(f"警告: 加载场景 {scene.scene_path} 的轨迹文件失败: {e}。将尝试加载下一个样本。")
            # 确保在数据集非空时才递归
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 0 else {}

        # 确定当前 scene 中共有的 camera_id
        common_cam_ids = sorted(list(human_pose_dict.keys() & tcp_base_dict.keys()))

        if not common_cam_ids:
            print(f"警告: 场景 {scene.scene_path} 中无共同相机ID的轨迹数据。将尝试加载下一个样本。")
            return self.__getitem__((idx + 1) % len(self)) if len(self) > 0 else {}

        pose_tensors = []
        tcp_tensors = []

        for cam_id in common_cam_ids:
            # --- 处理 human_pose ---
            valid_landmarks = [
                frame['hands_landmarks'][0] 
                for frame in human_pose_dict[cam_id] 
                if frame.get('hands_landmarks') # 使用 .get() 更安全
            ]
            if not valid_landmarks:
                continue # 如果该相机下没有有效的人手关键点，则跳过此相机

            # --- 处理 tcp_base ---
            all_tcps = [rec['tcp'] for rec in tcp_base_dict[cam_id]]
            if not all_tcps:
                continue # 如果该相机下没有TCP数据，也跳过

            # --- 拼接、采样并转换为 Tensor ---
            
            # 1. 堆叠 (seq_len, 21, 3)
            pose_trajectory = np.stack(valid_landmarks, axis=0)
            
            # 2. 【*** 此处为修改点 ***】
            #    根据 __init__ 中的设置进行条件采样
            if self.use_6_keypoints:
                # 从 (seq_len, 21, 3) 采样为 (seq_len, 6, 3)
                pose_trajectory = pose_trajectory[:, self.KEYPOINT_INDICES, :]
            
            # 3. 堆叠机器人轨迹
            tcp_trajectory = np.stack(all_tcps, axis=0)
            
            # 4. 转换并添加
            pose_tensors.append(torch.from_numpy(pose_trajectory).float())
            tcp_tensors.append(torch.from_numpy(tcp_trajectory).float())

        # 如果遍历完所有相机后，没有任何一对有效数据被处理
        if not pose_tensors:
              print(f"警告: 场景 {scene.scene_path} 中所有共同相机均无有效轨迹对。将尝试加载下一个样本。")
              return self.__getitem__((idx + 1) % len(self)) if len(self) > 0 else {}

        return {
            # "human_poses" 列表中的张量现在是 (L, 6, 3) 或 (L, 21, 3)
            "human_poses": pose_tensors,
            "tcp_bases": tcp_tensors,
            'scene_idx': idx
        }



def collate_trajectories(batch):
    """
    自定义的 collate 函数，用于处理包含可变长度轨迹的批次。
    它将批次内的所有人类轨迹和机器人轨迹分别进行填充，并生成注意力掩码。
    
    (修改)：此版本对长度超过 999 的机器人轨迹进行动态降采样。
    """
    all_human_poses =[]
    all_tcp_bases =[]
    
    # scene_indices 用于标识每个轨迹属于哪个场景
    human_scene_indices =[]
    robot_scene_indices =[]

    # 定义我们的目标最大长度
    # 要求是 "小于 1000"，所以最大允许长度是 999
    MAX_ROBOT_LEN = 999

    for item in batch:
        # --- 人类轨迹 (不变) ---
        # item['scene_idx'] 标识了它在批次中的原始场景
        # 我们用它来构建损失函数的标签
        all_human_poses.extend(item['human_poses'])
        human_scene_indices.extend([item['scene_idx']] * len(item['human_poses']))
        
        # --- 机器人轨迹 (动态采样) ---
        
        # 1. 创建一个临时列表来存放采样后的轨迹
        sampled_tcp_bases_for_item = []
        
        # 2. 遍历此 item 中的每一个机器人轨迹
        for trajectory in item['tcp_bases']:
            original_length = len(trajectory)
            
            if original_length == 0:
                # 处理空轨迹
                sampled_tcp_bases_for_item.append(trajectory)
                continue

            # 3. 计算动态步长 (stride)
            # stride = ceil(L / MAX_LEN)。确保 stride 至少为 1。
            stride = max(1, int(math.ceil(original_length / MAX_ROBOT_LEN)))
            
            # 4. 使用步长进行采样
            # trajectory[::stride] 会从头到尾每隔 stride 个点取一个
            sampled_trajectory = trajectory[::stride]
            
            # 5. 添加到临时列表
            sampled_tcp_bases_for_item.append(sampled_trajectory)

            # (调试) 检查新长度是否符合要求
            # assert len(sampled_trajectory) < 1000

        # 6. 将此 item 中所有 *采样后* 的轨迹添加到主列表
        all_tcp_bases.extend(sampled_tcp_bases_for_item)
        
        # 7. 添加场景索引。
        # 注意：这里的 len(item['tcp_bases']) 必须使用原始列表的长度，
        # 因为采样并没有改变轨迹的 *数量*，只是改变了它们的 *长度*。
        robot_scene_indices.extend([item['scene_idx']] * len(item['tcp_bases']))

    # --- 后续处理 (不变) ---

    # 对人类轨迹进行填充
    # pad_sequence 要求输入是张量列表
    human_lengths = [len(p) for p in all_human_poses]
    padded_human_poses = pad_sequence(all_human_poses, batch_first=True, padding_value=0.0)

    # 对机器人轨迹进行填充
    # 这里的 tcp_lengths 现在是 *采样后* 的长度
    tcp_lengths = [len(t) for t in all_tcp_bases]
    padded_tcp_bases = pad_sequence(all_tcp_bases, batch_first=True, padding_value=0.0)

    # 创建注意力掩码
    # 掩码形状为 (batch_size, seq_len)
    # 在 Transformer 中，通常 padding 的位置为 False 或 0
    human_mask = torch.arange(padded_human_poses.size(1))[None, :] < torch.tensor(human_lengths)[:, None]
    tcp_mask = torch.arange(padded_tcp_bases.size(1))[None, :] < torch.tensor(tcp_lengths)[:, None]

    return {
        'human_poses': padded_human_poses,
        'human_mask': human_mask,
        'tcp_bases': padded_tcp_bases,
        'tcp_mask': tcp_mask,
        'human_scene_indices': torch.tensor(human_scene_indices, dtype=torch.long),
        'robot_scene_indices': torch.tensor(robot_scene_indices, dtype=torch.long)
    }


if __name__ == "__main__":
    # --- 初始化数据集 ---
    DATASET_ROOT = '/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg2' 
    dataset = RH20TTraceDataset(
        root_dir=DATASET_ROOT
    )
    print(f"数据集初始化完成，共有 {len(dataset)} 个任务。")
    sample = dataset[0]
    print("示例数据键:", sample.keys())
    print("人类轨迹数量:", len(sample["human_poses"]))
    print("机器人轨迹数量:", len(sample["tcp_bases"]))
    if sample["human_poses"]:
        print("单个人类轨迹形状:", sample["human_poses"][0].shape)
    if sample["tcp_bases"]:
        print("单个机器人轨迹形状:", sample["tcp_bases"][0].shape)