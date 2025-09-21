import os
import shutil
import glob
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

def collect_scenes_by_task(source_dir):
    """
    扫描源目录，按任务ID收集所有机器人场景的路径。
    
    Args:
        source_dir (str): RH20T数据集的根目录。
        
    Returns:
        defaultdict: 一个字典，键是任务ID (e.g., 'task_0001')，
                     值是该任务下所有场景文件夹的路径列表。
    """
    scenes_by_task = defaultdict(list)
    print("正在扫描数据集，按任务收集场景...")

    # 遍历所有配置文件夹 (RH20T_cfg1, RH20T_cfg2,...)
    cfg_folders = glob.glob(os.path.join(source_dir, 'RH20T_cfg*'))
    if not cfg_folders:
        print(f"警告：在 '{source_dir}' 中未找到任何 'RH20T_cfg*' 文件夹。")
        return scenes_by_task

    for cfg_folder in tqdm(cfg_folders, desc="扫描配置文件夹"):
        # 查找所有场景文件夹，排除人类演示文件夹
        scene_paths = glob.glob(os.path.join(cfg_folder, 'task_*_user_*_scene_*_cfg_*'))
        robot_scene_paths = [p for p in scene_paths if not p.endswith('_human')]

        for scene_path in robot_scene_paths:
            scene_name = os.path.basename(scene_path)
            # 从场景名中提取任务ID，格式为 'task_xxxx'
            task_id = scene_name.split('_user_')[0]
            scenes_by_task[task_id].append(scene_path)
            
    print(f"扫描完成！共找到 {len(scenes_by_task)} 个独立任务。")
    return scenes_by_task

def extract_data(source_dir, target_dir, n_scenes, m_cameras, allowed_cameras):
    """
    执行数据提取、采样和拷贝的核心函数。
    """
    # 1. 按任务收集所有场景
    scenes_by_task = collect_scenes_by_task(source_dir)
    if not scenes_by_task:
        print("未收集到任何场景，程序退出。")
        return

    task_count=0
    # 2. 创建目标根目录
    os.makedirs(target_dir, exist_ok=True)
    print(f"数据将被提取到: {target_dir}")

    # 3. 遍历每个任务并提取数据
    for task_id, scene_paths in tqdm(scenes_by_task.items(), desc="处理任务"):
        task_count+=1
        # 为当前任务创建子目录
        task_output_dir = os.path.join(target_dir, "task_" + f"{task_count:04d}")
        os.makedirs(task_output_dir, exist_ok=True)

        if len(scene_paths) > n_scenes:
            # selected_scenes = random.sample(scene_paths, n_scenes)
            selected_scenes=scene_paths[0:n_scenes]
        else:
            selected_scenes = scene_paths # 如果场景数不足n，则全部选取
        
        # 遍历选中的场景
        for i,scene_path in enumerate(selected_scenes):
            scene_name = os.path.basename(scene_path)
            human_scene_path = scene_path + '_human'

            # 为当前场景创建子目录
            scene_output_dir = os.path.join(task_output_dir, f"scene_{i+1}")
            os.makedirs(scene_output_dir, exist_ok=True)



            # --- a. 拷贝机器人轨迹文件 ---
            trajectory_src_path = os.path.join(scene_path, 'transformed', 'tcp_base.npy')
            if os.path.exists(trajectory_src_path):
                # 重命名以避免冲突
                trajectory_dst_path = os.path.join(scene_output_dir, f"tcp_base.npy")
                shutil.copy2(trajectory_src_path, trajectory_dst_path)
            else:
                print(f"警告：未找到轨迹文件 {trajectory_src_path}")

            # --- b. 筛选并抽取m个相机视角 ---
            available_cam_dirs = glob.glob(os.path.join(scene_path, 'cam_*'))
            # 从路径中提取相机序列号
            available_cam_serials = [os.path.basename(d) for d in available_cam_dirs]
            
            # 筛选出在允许列表中的相机
            valid_cameras = [cam for cam in available_cam_serials if cam in allowed_cameras]
            # if len(valid_cameras) > m_cameras:
            #     selected_cameras = random.sample(valid_cameras, m_cameras)
            # else:
            #     selected_cameras = valid_cameras # 如果有效相机数不足m，则全部选取

            # --- c. 拷贝人机视频文件 ---
            count=0
            for cam_serial in valid_cameras:
                # 机器人视频路径
                robot_video_src = os.path.join(scene_path, cam_serial, 'color.mp4')
                # 人类演示视频路径
                human_video_src = os.path.join(human_scene_path, cam_serial, 'color.mp4')

                # 拷贝机器人视频
                if os.path.exists(robot_video_src) and os.path.exists(human_video_src):
                    robot_video_dst = os.path.join(scene_output_dir, f"{cam_serial}_robot.mp4")
                    shutil.copy2(robot_video_src, robot_video_dst)
                    human_video_dst = os.path.join(scene_output_dir, f"{cam_serial}_human.mp4")
                    shutil.copy2(human_video_src, human_video_dst)
                    count+=1
                    if count>=m_cameras:
                        break
                # else:
                #     print(f"警告：未找到人或机视频 {scene_name}_{cam_serial}")
            if count<m_cameras:
                print(f"警告：任务 {task_id} 的场景 {scene_name} 中，有效相机数不足 {m_cameras} 个，仅找到 {count} 个。")


    print("\n数据提取完成！")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="从RH20T数据集中提取部分数据用于对比学习。")
    # parser.add_argument('--source_dir', type=str, required=True, 
    #                     help="RH20T数据集的根目录 (例如: /home/ttt/BISE/RH20T)")
    # parser.add_argument('--target_dir', type=str, required=True, 
    #                     help="提取后数据的存放目录 (例如: /home/ttt/RH20T_subset)")
    # parser.add_argument('--n_scenes', type=int, default=5, 
    #                     help="每个任务需要提取的场景数量 (n)。")
    # parser.add_argument('--m_cameras', type=int, default=2, 
    #                     help="每个场景需要提取的相机视角的数量 (m)。")
    
    # args = parser.parse_args()

    # 您指定需要筛选的相机序列号列表
    ALLOWED_CAMERAS = [
        'cam_f0172289',
        'cam_038522062288',
        'cam_104122063550',
        'cam_104122062295',
        'cam_104122062823',
        'cam_104422070011'
    ]

    extract_data("/home/ttt/BISE/RH20T", "/home/ttt/BISE/RH20T_subset", 4, 3, ALLOWED_CAMERAS)