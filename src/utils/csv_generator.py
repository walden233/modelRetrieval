import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import logging

class HumanRobotCSVGenerator:
    """
    生成HumanRobotDataset所需的CSV文件的工具类
    
    功能：
    - 扫描指定路径下的human和robot文件夹
    - 自动匹配对应的视频文件
    - 生成符合格式的CSV文件
    
    CSV格式：
    human_video_path,robot_video_path,task_id
    /data/human/pick_cup_01.mp4,/data/robot/pick_cup_A.mp4,pick_cup
    ...
    """
    
    def __init__(self, dataset_root_path: str, output_csv_path: str = "dataset.csv"):
        """
        初始化CSV生成器
        
        Args:
            dataset_root_path: 数据集根目录路径，包含human和robot子文件夹
            output_csv_path: 输出的CSV文件路径
        """
        self.dataset_root_path = Path(dataset_root_path)
        self.output_csv_path = Path(output_csv_path)
        self.human_dir = self.dataset_root_path / "human"
        self.robot_dir = self.dataset_root_path / "robot"
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 验证目录结构
        self._validate_directory_structure()
    
    def _validate_directory_structure(self):
        """验证目录结构是否正确"""
        if not self.dataset_root_path.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {self.dataset_root_path}")
        
        if not self.human_dir.exists():
            raise FileNotFoundError(f"human文件夹不存在: {self.human_dir}")
        
        if not self.robot_dir.exists():
            raise FileNotFoundError(f"robot文件夹不存在: {self.robot_dir}")
        
        self.logger.info(f"目录结构验证通过: {self.dataset_root_path}")
    
    def _get_video_files(self, directory: Path) -> List[Path]:
        """获取指定目录下的所有视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        video_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def _match_video_pairs(self) -> List[Tuple[str, str, str]]:
        """
        匹配human和robot视频对
        
        Returns:
            List[Tuple[human_path, robot_path, task_id]]: 匹配的视频对列表
        """
        human_videos = self._get_video_files(self.human_dir)
        robot_videos = self._get_video_files(self.robot_dir)
        
        matched_pairs = []
        unmatched_human = []
        unmatched_robot = []
        
        # 创建人类视频的映射字典（文件名 -> 完整路径）
        human_video_map = {video.stem: video for video in human_videos}
        
        # 遍历机器人视频，用机器人视频名作为前缀去匹配人类视频
        for robot_video in robot_videos:
            robot_stem = robot_video.stem
            
            # 尝试匹配人类视频
            # 情况1: 完全匹配（如 bag.mp4 和 bag.mp4）
            if robot_stem in human_video_map:
                human_video = human_video_map[robot_stem]
                task_id = robot_stem
                matched_pairs.append((
                    str(human_video),
                    str(robot_video),
                    task_id
                ))
                del human_video_map[robot_stem]  # 移除已匹配的
            else:
                # 情况2: 前缀匹配（如 bag.mp4 和 bag_anonymized.mp4）
                # 查找以机器人视频名为前缀的人类视频
                matching_human_videos = [
                    (stem, path) for stem, path in human_video_map.items()
                    if stem.startswith(robot_stem)
                ]
                
                if matching_human_videos:
                    # 取第一个匹配的
                    human_stem, human_video = matching_human_videos[0]
                    task_id = robot_stem  # 使用机器人视频名作为task_id
                    matched_pairs.append((
                        str(human_video),
                        str(robot_video),
                        task_id
                    ))
                    del human_video_map[human_stem]  # 移除已匹配的
                else:
                    unmatched_robot.append(robot_video)
        
        # 剩余的人类视频为未匹配的
        unmatched_human = list(human_video_map.values())
        
        # 记录匹配信息
        self.logger.info(f"成功匹配 {len(matched_pairs)} 对视频")
        if unmatched_human:
            self.logger.warning(f"未匹配的人类视频: {len(unmatched_human)} 个")
            for video in unmatched_human[:5]:  # 只显示前5个
                self.logger.warning(f"  - {video.name}")
            if len(unmatched_human) > 5:
                self.logger.warning(f"  ... 还有 {len(unmatched_human) - 5} 个")
        
        if unmatched_robot:
            self.logger.warning(f"未匹配的机器人视频: {len(unmatched_robot)} 个")
            for video in unmatched_robot[:5]:  # 只显示前5个
                self.logger.warning(f"  - {video.name}")
            if len(unmatched_robot) > 5:
                self.logger.warning(f"  ... 还有 {len(unmatched_robot) - 5} 个")
        
        return matched_pairs
    
    def generate_csv(self, overwrite: bool = False) -> bool:
        """
        生成CSV文件
        
        Args:
            overwrite: 是否覆盖已存在的CSV文件
            
        Returns:
            bool: 是否成功生成CSV文件
        """
        # 检查输出文件是否已存在
        if self.output_csv_path.exists() and not overwrite:
            self.logger.warning(f"CSV文件已存在: {self.output_csv_path}")
            self.logger.warning("使用 overwrite=True 来覆盖现有文件")
            return False
        
        try:
            # 匹配视频对
            matched_pairs = self._match_video_pairs()
            
            if not matched_pairs:
                self.logger.error("没有找到匹配的视频对")
                return False
            
            # 创建DataFrame
            df = pd.DataFrame(
                matched_pairs,
                columns=['human_video_path', 'robot_video_path', 'task_id']
            )
            
            # 确保输出目录存在
            self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存CSV文件
            df.to_csv(self.output_csv_path, index=False)
            
            self.logger.info(f"CSV文件生成成功: {self.output_csv_path}")
            self.logger.info(f"共生成 {len(matched_pairs)} 条记录")
            
            # 显示前几条记录作为示例
            self.logger.info("前3条记录示例:")
            for i, (_, row) in enumerate(df.head(3).iterrows()):
                self.logger.info(f"  {i+1}. {row['human_video_path']}")
                self.logger.info(f"     {row['robot_video_path']}")
                self.logger.info(f"     {row['task_id']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"生成CSV文件时发生错误: {str(e)}")
            return False
    
    def get_statistics(self) -> dict:
        """
        获取数据集统计信息
        
        Returns:
            dict: 包含统计信息的字典
        """
        human_videos = self._get_video_files(self.human_dir)
        robot_videos = self._get_video_files(self.robot_dir)
        matched_pairs = self._match_video_pairs()
        
        stats = {
            'human_videos_count': len(human_videos),
            'robot_videos_count': len(robot_videos),
            'matched_pairs_count': len(matched_pairs),
            'unmatched_human_count': len(human_videos) - len(matched_pairs),
            'unmatched_robot_count': len(robot_videos) - len(matched_pairs),
            'match_rate': len(matched_pairs) / max(len(human_videos), len(robot_videos)) * 100
        }
        
        return stats


# 使用示例
if __name__ == "__main__":
    # 示例用法
    generator = HumanRobotCSVGenerator(
        dataset_root_path="whirl_videos",
        output_csv_path="dataset.csv"
    )
    
    # 生成CSV文件
    success = generator.generate_csv(overwrite=True)
    
    if success:
        print("CSV文件生成成功！")
        
        # 获取统计信息
        stats = generator.get_statistics()
        print("\n数据集统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("CSV文件生成失败！")