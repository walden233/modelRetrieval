import json
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


def test_fixed_test_with_seeded_dynamic_train_val(tmp_path):
    dataset = _TrajectoryDataset()
    dataset.scenes = [
        SceneRecord(scene_path=f"/tmp/rh20t/task_{index:04d}/scene_1")
        for index in range(20)
    ]
    fixed_test = ["task_0018/scene_1", "task_0019/scene_1"]
    manifest_path = tmp_path / "fixed_test.json"
    manifest_path.write_text(json.dumps({"test": fixed_test}))
    base = {
        "fixed_test_manifest_path": str(manifest_path),
        "fixed_test_fraction": 0.1,
        "val_ratio_of_remaining": 1.0 / 9.0,
        "unit": "scene",
    }

    split_42 = build_split_manifest(split_trajectory_dataset(dataset, {**base, "seed": 42}))
    split_43 = build_split_manifest(split_trajectory_dataset(dataset, {**base, "seed": 43}))

    assert split_42["test"] == fixed_test
    assert split_43["test"] == fixed_test
    assert len(split_42["train"]) == 16
    assert len(split_42["val"]) == 2
    assert set(split_42["val"]) != set(split_43["val"])
    assert not set(fixed_test) & (set(split_42["train"]) | set(split_42["val"]))


def test_fixed_test_fraction_is_checked(tmp_path):
    dataset = _TrajectoryDataset()
    manifest_path = tmp_path / "fixed_test.json"
    manifest_path.write_text(json.dumps({"test": ["task_0001/scene_1"]}))
    try:
        split_trajectory_dataset(dataset, {
            "fixed_test_manifest_path": str(manifest_path),
            "fixed_test_fraction": 0.5,
        })
    except ValueError as exc:
        assert "expected 2" in str(exc)
    else:
        raise AssertionError("Mismatched fixed test size was accepted")
