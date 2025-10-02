import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import argparse
import re
from typing import List, Dict, Tuple, Any

# 初始化MediaPipe解决方案
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 根据数据集描述，定义允许处理的相机序列号列表
ALLOWED_CAMERAS = [
    'f0172289',
    '038522062288',
    '104122063550',
    '104122062295',
    '104122062823',
    '104422070011'
]

def find_scenes_to_process(root_dir: str) -> List:
    """
    遍历数据集根目录，查找所有待处理的场景和视频文件。

    Args:
        root_dir (str): RH20T数据集的根目录路径。

    Returns:
        List: 一个处理任务列表，每个元素为
                                    (场景路径, 视频文件路径, 相机序列号)。
    """
    tasks = []
    cam_pattern = re.compile(r'cam_([a-zA-Z0-9]+)_human\.mp4')
    
    print(f"[*] 开始扫描目录: {root_dir}")
    for dirpath, _, filenames in os.walk(root_dir):
        # 确认这是一个场景目录 (包含视频和tcp_base.npy)
        if any(f.endswith('_human.mp4') for f in filenames) and 'tcp_base.npy' in filenames:
            for filename in filenames:
                match = cam_pattern.match(filename)
                if match:
                    camera_serial = match.group(1)
                    if camera_serial in ALLOWED_CAMERAS:
                        video_path = os.path.join(dirpath, filename)
                        tasks.append((dirpath, video_path, camera_serial))
    print(f"[+] 发现 {len(tasks)} 个有效视频任务。")
    return tasks

def structure_frame_data(frame_index: int, mediapipe_results: Any) -> Dict[str, Any]:
    """
    将单帧的MediaPipe输出结果构造成标准格式的字典。

    Args:
        frame_index (int): 当前帧的索引。
        mediapipe_results (Any): MediaPipe Hands模型的process()方法返回的结果对象。

    Returns:
        Dict[str, Any]: 符合设计规范的单帧数据字典。
    """
    frame_landmarks = []
    if mediapipe_results.multi_hand_world_landmarks:
        for hand_world_landmarks in mediapipe_results.multi_hand_world_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark], dtype=np.float32)
            frame_landmarks.append(landmarks)
            
    return {'frame_index': frame_index, 'hands_landmarks': frame_landmarks}

def extract_hand_poses_from_video(video_path: str) -> List:
    """
    从单个视频文件中提取所有帧的手部姿态轨迹。

    Args:
        video_path (str): 待处理的视频文件路径。

    Returns:
        List]: 包含该视频所有帧姿态数据的列表。
    """
    trajectory = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] 错误: 无法打开视频文件 {video_path}")
        return trajectory

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands_model:
        
        for frame_idx in tqdm(range(total_frames), desc=f"处理 {os.path.basename(video_path)}"):
            success, image = cap.read()
            if not success:
                break

            # 关键步骤: 将BGR图像转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # 提高性能
            
            results = hands_model.process(image_rgb)
            
            frame_data = structure_frame_data(frame_idx, results)
            trajectory.append(frame_data)
            
    cap.release()
    return trajectory

def group_tasks_by_scene(tasks: List) -> Dict:
    """将任务列表按场景路径进行分组"""
    grouped_tasks = {}
    for scene_path, video_path, camera_serial in tasks:
        if scene_path not in grouped_tasks:
            grouped_tasks[scene_path] = []
        grouped_tasks[scene_path].append((video_path, camera_serial))
    return grouped_tasks


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="从RH20T数据集中提取人类手部姿态。")
    # parser.add_argument('--dataset_root', type=str, required=True, help='RH20T数据集子集的根目录路径。')
    # args = parser.parse_args()
    dataset_root="/home/ttt/BISE/RH20T_subset"

    all_tasks = find_scenes_to_process(dataset_root)
    grouped_tasks = group_tasks_by_scene(all_tasks)

    for scene_path, video_tasks in grouped_tasks.items():
        print(f"\n[*] 开始处理场景: {scene_path}")
        scene_hand_data = {}
        
        for video_path, camera_serial in video_tasks:
            hand_trajectory = extract_hand_poses_from_video(video_path)
            if hand_trajectory:
                scene_hand_data[camera_serial] = hand_trajectory
        
        if scene_hand_data:
            output_path = os.path.join(scene_path, 'human_pose.npy')
            print(f"[*] 正在保存场景数据到: {output_path}")
            np.save(output_path, scene_hand_data)
            print(f"[+] 场景 {os.path.basename(scene_path)} 处理完成。")

    print("\n[+] 所有任务已完成。")