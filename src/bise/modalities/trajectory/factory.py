import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from torch.utils.data import Subset

from bise.common import resolve_path


def split_trajectory_dataset(dataset, split_config: Dict | None):
    if not split_config:
        return {"train": dataset, "val": None, "test": None}

    manifest_path = split_config.get("manifest_path")
    if manifest_path:
        split_map = _load_split_manifest(resolve_path(manifest_path))
        return {split_name: _subset_from_scene_ids(dataset, scene_ids) for split_name, scene_ids in split_map.items()}

    unit = split_config.get("unit", "scene")
    ratios = split_config.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = int(split_config.get("seed", 42))
    return _split_by_group_unit(dataset, unit=unit, ratios=ratios, seed=seed)


def build_split_manifest(split_datasets: Dict[str, object]) -> Dict[str, List[str]]:
    return {split_name: _collect_scene_ids(split_dataset) for split_name, split_dataset in split_datasets.items()}


def scene_id_from_record(scene_record) -> str:
    scene_path = Path(scene_record.scene_path)
    return f"{scene_path.parent.name}/{scene_path.name}"


def task_id_from_record(scene_record) -> str:
    return Path(scene_record.scene_path).parent.name


def _split_by_group_unit(dataset, unit: str, ratios: Dict[str, float], seed: int):
    groups: Dict[str, List[int]] = {}
    for index, scene in enumerate(dataset.scenes):
        key = _group_key(scene, unit)
        groups.setdefault(key, []).append(index)

    group_keys = list(groups.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(group_keys)

    train_ratio = float(ratios.get("train", 0.8))
    val_ratio = float(ratios.get("val", 0.1))
    if train_ratio + val_ratio > 1.0 + 1e-6:
        raise ValueError("The sum of train and val ratios must be <= 1.0.")

    total_groups = len(group_keys)
    train_cutoff = int(round(total_groups * train_ratio))
    val_cutoff = train_cutoff + int(round(total_groups * val_ratio))
    split_groups = {
        "train": group_keys[:train_cutoff],
        "val": group_keys[train_cutoff:val_cutoff],
        "test": group_keys[val_cutoff:],
    }
    return {
        split_name: Subset(dataset, [index for key in keys for index in groups[key]])
        for split_name, keys in split_groups.items()
    }


def _group_key(scene_record, unit: str) -> str:
    normalized = str(unit).strip().lower()
    if normalized == "scene":
        return scene_id_from_record(scene_record)
    if normalized == "task":
        return task_id_from_record(scene_record)
    raise ValueError(f"Unsupported trajectory split unit: {unit}")


def _load_split_manifest(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Trajectory split manifest must be a JSON object with split keys.")
    return {split_name: list(scene_ids) for split_name, scene_ids in payload.items()}


def _subset_from_scene_ids(dataset, scene_ids: Iterable[str]):
    id_to_index = {scene_id_from_record(scene): index for index, scene in enumerate(dataset.scenes)}
    indices: List[int] = []
    missing: List[str] = []
    for scene_id in scene_ids:
        if scene_id not in id_to_index:
            missing.append(scene_id)
        else:
            indices.append(id_to_index[scene_id])
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} scene ids from split manifest are not in the dataset. Examples: {preview}")
    return Subset(dataset, indices)


def _collect_scene_ids(dataset) -> List[str]:
    if dataset is None:
        return []
    if isinstance(dataset, Subset):
        source = dataset.dataset
        return [scene_id_from_record(source.scenes[index]) for index in dataset.indices]
    if hasattr(dataset, "scenes"):
        return [scene_id_from_record(scene) for scene in dataset.scenes]
    raise TypeError(f"Cannot collect scene ids from dataset type: {type(dataset).__name__}")
