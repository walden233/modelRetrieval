import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.data.rh20t.scanner import SceneRecord
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset


class _TrajectoryDataset:
    def __init__(self):
        self.scenes = [
            SceneRecord(scene_path="/tmp/rh20t/task_0001/scene_1"),
            SceneRecord(scene_path="/tmp/rh20t/task_0001/scene_2"),
            SceneRecord(scene_path="/tmp/rh20t/task_0002/scene_1"),
        ]

    def __getitem__(self, index):
        return self.scenes[index]

    def __len__(self):
        return len(self.scenes)


def test_split_trajectory_dataset_all_as_test():
    splits = split_trajectory_dataset(_TrajectoryDataset(), {"all_as_test": True})

    assert len(splits["train"]) == 0
    assert len(splits["val"]) == 0
    assert len(splits["test"]) == 3
    assert build_split_manifest(splits) == {
        "train": [],
        "val": [],
        "test": ["task_0001/scene_1", "task_0001/scene_2", "task_0002/scene_1"],
    }
