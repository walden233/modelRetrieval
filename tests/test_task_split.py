from itertools import product
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.data.rh20t.splits import validate_split
from bise.data.rh20t.task_split import split_task_dataset, task_scene_split
from bise.modalities.trajectory.factory import build_split_manifest
from bise.modalities.video.factory import build_split_manifest as video_manifest


def identities(sizes):
    return [f"task_{task:04d}/scene_{scene}" for task, size in enumerate(sizes)
            for scene in range(size)]


@pytest.mark.parametrize("sizes,train", [([4] * 102 + [3] * 6 + [2], 258),
                                       ([4] * 107 + [3] * 2, 264)])
def test_task_targets_keep_tasks_disjoint_and_cover_the_inventory(sizes, train):
    available = identities(sizes)
    config = {"unit": "task", "seed": 42, "scene_counts": {"test": 120, "val": 50}}
    mapping = task_scene_split(available, config)
    counts = validate_split(mapping, "task")
    assert [counts[name]["scenes"] for name in ("train", "val", "test")] == [train, 50, 120]
    assert set().union(*map(set, mapping.values())) == set(available)
    assert mapping == task_scene_split(list(reversed(available)), config)
    assert mapping != task_scene_split(available, {**config, "seed": 43})


@pytest.mark.parametrize("sizes,test,val", [([3, 5, 7, 4, 2], 8, 6), ([4] * 6, 8, 10)])
def test_joint_task_targets_choose_the_closest_feasible_partition(sizes, test, val):
    config = {"unit": "task", "seed": 42, "scene_counts": {"test": test, "val": val}}
    mapping = task_scene_split(identities(sizes), config)
    validate_split(mapping, "task")
    scores = []
    for choices in product(("train", "val", "test"), repeat=len(sizes)):
        counts = {name: sum(size for size, choice in zip(sizes, choices) if choice == name)
                  for name in ("train", "val", "test")}
        if all(counts.values()):
            scores.append((abs(counts["test"] - test) + abs(counts["val"] - val),
                           abs(counts["test"] - test), (counts["test"], counts["val"])))
    actual = (len(mapping["test"]), len(mapping["val"]))
    score = (abs(actual[0] - test) + abs(actual[1] - val), abs(actual[0] - test), actual)
    assert score == min(scores)


def test_both_modalities_apply_task_targets_and_preserve_video_pairs():
    available = identities([4] * 107 + [3] * 2)
    class Dataset:
        def __len__(self):
            return len(self.scenes) if hasattr(self, "scenes") else len(self.samples)
    trajectory = Dataset()
    trajectory.scenes = [SimpleNamespace(scene_path=f"/data/{scene}") for scene in available]
    video = Dataset()
    video.samples = [SimpleNamespace(scene_id=scene,
                                    sample_id=f"rh20t::{scene.replace('/', '::')}::{camera}")
                     for scene in available for camera in ("0", "1")]
    config = {"unit": "task", "seed": 42, "scene_counts": {"test": 120, "val": 50}}
    traj = build_split_manifest(split_task_dataset(trajectory, config, "trajectory"))
    videos = video_manifest(split_task_dataset(video, config, "video"))
    assert validate_split(traj, "task")["val"]["scenes"] == 50
    assert validate_split(videos, "task")["test"]["samples"] == 240
    for name in traj:
        assert len(videos[name]) == 2 * len(traj[name])
        assert set(traj[name]) == {"/".join(sample.split("::")[1:3]) for sample in videos[name]}


@pytest.mark.parametrize("changes", [{"unit": "scene"},
                                     {"scene_counts": {"test": 0, "val": 4}},
                                     {"scene_counts": {"test": True, "val": 4}},
                                     {"scene_counts": {"test": 5.5, "val": 4}},
                                     {"scene_counts": {"test": 8, "val": 4}}])
def test_invalid_task_targets_are_rejected(changes):
    config = {"unit": "task", "seed": 42, "scene_counts": {"test": 4, "val": 4}}
    with pytest.raises(ValueError):
        task_scene_split(identities([4] * 3), {**config, **changes})
