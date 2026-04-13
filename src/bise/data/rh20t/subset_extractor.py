import glob
import os
import shutil
from collections import defaultdict

from tqdm import tqdm


def collect_scenes_by_task(source_dir):
    scenes_by_task = defaultdict(list)
    scene_paths = glob.glob(os.path.join(source_dir, "task_*_user_*_scene_*_cfg_*"))
    robot_scene_paths = [path for path in scene_paths if not (path.endswith("_human") or path.endswith("_human_2"))]
    for scene_path in robot_scene_paths:
        task_id = os.path.basename(scene_path).split("_user_")[0]
        scenes_by_task[task_id].append(scene_path)
    return scenes_by_task


def extract_subset(source_dir, target_dir, n_scenes, m_cameras):
    scenes_by_task = collect_scenes_by_task(source_dir)
    os.makedirs(target_dir, exist_ok=True)

    for task_count, (_, scene_paths) in enumerate(tqdm(scenes_by_task.items(), desc="Processing tasks"), start=1):
        task_output_dir = os.path.join(target_dir, f"task_{task_count:04d}")
        os.makedirs(task_output_dir, exist_ok=True)
        selected_scenes = scene_paths[:n_scenes] if len(scene_paths) > n_scenes else scene_paths

        for index, scene_path in enumerate(selected_scenes, start=1):
            human_scene_path = scene_path + "_human"
            scene_output_dir = os.path.join(task_output_dir, f"scene_{index}")
            os.makedirs(scene_output_dir, exist_ok=True)

            trajectory_src_path = os.path.join(scene_path, "transformed", "tcp_base.npy")
            if os.path.exists(trajectory_src_path):
                shutil.copy2(trajectory_src_path, os.path.join(scene_output_dir, "tcp_base.npy"))

            available_cam_dirs = glob.glob(os.path.join(scene_path, "cam_*"))
            available_cam_serials = [os.path.basename(directory) for directory in available_cam_dirs]
            copied = 0
            for camera_serial in available_cam_serials:
                robot_video_src = os.path.join(scene_path, camera_serial, "color.mp4")
                human_video_src = os.path.join(human_scene_path, camera_serial, "color.mp4")
                if os.path.exists(robot_video_src) and os.path.exists(human_video_src):
                    shutil.copy2(robot_video_src, os.path.join(scene_output_dir, f"{camera_serial}_robot.mp4"))
                    shutil.copy2(human_video_src, os.path.join(scene_output_dir, f"{camera_serial}_human.mp4"))
                    copied += 1
                    if copied >= m_cameras:
                        break
