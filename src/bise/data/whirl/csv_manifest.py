import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd


class HumanRobotCSVGenerator:
    def __init__(self, dataset_root_path: str, output_csv_path: str = "dataset.csv"):
        self.dataset_root_path = Path(dataset_root_path)
        self.output_csv_path = Path(output_csv_path)
        self.human_dir = self.dataset_root_path / "human"
        self.robot_dir = self.dataset_root_path / "robot"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self._validate_directory_structure()

    def _validate_directory_structure(self) -> None:
        if not self.dataset_root_path.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root_path}")
        if not self.human_dir.exists():
            raise FileNotFoundError(f"Missing human directory: {self.human_dir}")
        if not self.robot_dir.exists():
            raise FileNotFoundError(f"Missing robot directory: {self.robot_dir}")

    def _get_video_files(self, directory: Path) -> List[Path]:
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in video_extensions)

    def _match_video_pairs(self) -> List[Tuple[str, str, str]]:
        human_videos = self._get_video_files(self.human_dir)
        robot_videos = self._get_video_files(self.robot_dir)
        human_video_map = {video.stem: video for video in human_videos}
        matched_pairs = []

        for robot_video in robot_videos:
            robot_stem = robot_video.stem
            if robot_stem in human_video_map:
                human_video = human_video_map.pop(robot_stem)
                matched_pairs.append((str(human_video), str(robot_video), robot_stem))
                continue

            matching_humans = [(stem, path) for stem, path in human_video_map.items() if stem.startswith(robot_stem)]
            if matching_humans:
                human_stem, human_video = matching_humans[0]
                matched_pairs.append((str(human_video), str(robot_video), robot_stem))
                human_video_map.pop(human_stem)

        return matched_pairs

    def generate_csv(self, overwrite: bool = False) -> bool:
        if self.output_csv_path.exists() and not overwrite:
            return False

        matched_pairs = self._match_video_pairs()
        if not matched_pairs:
            return False

        dataframe = pd.DataFrame(matched_pairs, columns=["human_video_path", "robot_video_path", "task_id"])
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_csv_path, index=False)
        return True

    def get_statistics(self) -> dict:
        human_videos = self._get_video_files(self.human_dir)
        robot_videos = self._get_video_files(self.robot_dir)
        matched_pairs = self._match_video_pairs()
        return {
            "human_videos_count": len(human_videos),
            "robot_videos_count": len(robot_videos),
            "matched_pairs_count": len(matched_pairs),
            "unmatched_human_count": len(human_videos) - len(matched_pairs),
            "unmatched_robot_count": len(robot_videos) - len(matched_pairs),
            "match_rate": len(matched_pairs) / max(len(human_videos), len(robot_videos)) * 100,
        }
