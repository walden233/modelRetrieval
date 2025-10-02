import numpy as np
import cv2
import argparse
import os

def verify_output(scene_path: str):
    """
    验证指定场景下生成的human_pose.npy文件的正确性。
    """
    npy_path = os.path.join(scene_path, 'human_pose.npy')
    if not os.path.exists(npy_path):
        print(f"[!] 错误: 在 {scene_path} 中未找到 human_pose.npy 文件。")
        return

    print(f"--- 开始验证: {npy_path} ---")
    data = np.load(npy_path, allow_pickle=True).item()

    # 1. 结构完整性检查
    print(f"[*] 顶层键 (相机序列号): {list(data.keys())}")
    assert isinstance(data, dict), "顶层结构应为字典"

    for cam_serial, trajectory in data.items():
        print(f"\n--- 正在检查相机: {cam_serial} ---")
        
        # 2. 时间完整性检查
        video_path = os.path.join(scene_path, f'cam_{cam_serial}_human.mp4')
        if not os.path.exists(video_path):
            print(f"[!] 警告: 找不到对应的视频文件 {video_path}")
            continue
            
        cap = cv2.VideoCapture(video_path)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        trajectory_length = len(trajectory)
        print(f"[*] 轨迹长度: {trajectory_length}, 视频帧数: {video_frame_count}")
        assert trajectory_length == video_frame_count, "轨迹长度与视频帧数不匹配"
        print("[+] 时间完整性检查通过。")

        # 3. 数据形状与内容检查
        if trajectory_length > 0:
            # 随机抽查一帧
            sample_record = trajectory[trajectory_length // 2]
            print(f"[*] 抽样检查第 {sample_record['frame_index']} 帧的数据...")
            assert isinstance(sample_record, dict), "记录应为字典"
            assert 'frame_index' in sample_record and 'hands_landmarks' in sample_record, "记录缺少必要键"
            
            hands_landmarks = sample_record['hands_landmarks']
            assert isinstance(hands_landmarks, list), "'hands_landmarks'应为列表"
            
            if hands_landmarks: # 如果检测到了手
                hand_data = hands_landmarks
                print(f"[*] 检测到 {len(hands_landmarks)} 只手。第一只手的关节点数据形状: {hand_data[0].shape}")
                assert isinstance(hand_data[0], np.ndarray), "关节点数据应为NumPy数组"
                assert hand_data[0].shape == (21, 3), "关节点数据形状应为 (21, 3)"
            else:
                print("[*] 该帧未检测到手。")
            print("[+] 数据形状与内容检查通过。")
    
    print("\n--- 所有验证完成，文件格式正确。 ---")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="验证生成的human_pose.npy文件。")
    # parser.add_argument('--scene_path', type=str, required=True, help='包含human_pose.npy的场景目录路径。')
    # args = parser.parse_args()
    scene_path = "/home/ttt/BISE/dataset/RH20T_subset/RH20T_cfg3/task_0001/scene_1"
    verify_output(scene_path)