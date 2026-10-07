"""Choose disjoint tasks whose scene counts approach validation/test targets."""

from pathlib import Path

import numpy as np
from torch.utils.data import Subset


def task_scene_split(available: list[str], config: dict) -> dict[str, list[str]]:
    if config.get("unit") != "task":
        raise ValueError("scene_counts requires unit=task")
    targets = config["scene_counts"]
    test, val = targets["test"], targets["val"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (test, val)):
        raise ValueError("Task test/val scene targets must be positive integers")
    scenes = sorted(set(available))
    groups = {}
    for scene in scenes:
        groups.setdefault(scene.split("/")[0], []).append(scene)
    if len(groups) < 3 or test + val >= len(scenes):
        raise ValueError("Task split requires three non-empty partitions and a training pool")
    tasks = list(groups)
    np.random.default_rng(int(config.get("seed", 42))).shuffle(tasks)
    largest = max(map(len, groups.values()))
    limits = (min(test + largest, len(scenes)), min(val + largest, len(scenes)))

    # Each reachable pair stores its first path, using each task at most once.
    paths = {(0, 0): None}
    for task in tasks:
        size = len(groups[task])
        for current in list(paths):
            for name, state in (("test", (current[0] + size, current[1])),
                                ("val", (current[0], current[1] + size))):
                if (state[0] <= limits[0] and state[1] <= limits[1]
                        and sum(state) < len(scenes) and state not in paths):
                    paths[state] = (current, task, name)
    candidates = [state for state in paths if state[0] and state[1]]
    if not candidates:
        raise ValueError("Cannot form non-empty task train/val/test partitions")
    state = min(candidates, key=lambda x: (abs(x[0] - test) + abs(x[1] - val),
                                          abs(x[0] - test), x))
    selected = {"test": set(), "val": set()}
    while paths[state] is not None:
        state, task, name = paths[state]
        selected[name].add(task)
    selected["train"] = set(tasks) - selected["test"] - selected["val"]
    return {name: [scene for scene in scenes if scene.split("/")[0] in selected[name]]
            for name in ("train", "val", "test")}


def split_task_dataset(dataset, config: dict, modality: str) -> dict:
    """Apply scene targets during training; evaluation reuses the saved manifest."""
    if modality == "trajectory":
        identities = [f"{Path(scene.scene_path).parent.name}/{Path(scene.scene_path).name}"
                      for scene in dataset.scenes]
    elif modality == "video":
        identities = [sample.scene_id for sample in dataset.samples]
    else:
        raise ValueError(f"Unsupported task split modality: {modality}")
    mapping = task_scene_split(identities, config)
    subsets = {}
    for name, scenes in mapping.items():
        selected = set(scenes)
        subsets[name] = Subset(dataset, [i for i, identity in enumerate(identities)
                                        if identity in selected])
    return subsets
