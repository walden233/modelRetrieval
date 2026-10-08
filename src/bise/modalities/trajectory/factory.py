import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from torch.utils.data import Subset

from bise.common import resolve_path


def split_trajectory_dataset(dataset, split_config: Dict | None):
    if not split_config:
        return {"train": dataset, "val": None, "test": None}

    if _is_all_test_split(split_config):
        return _all_as_test_split(dataset)

    manifest_path = split_config.get("manifest_path")
    if manifest_path:
        split_map = _load_split_manifest(resolve_path(manifest_path))
        return {split_name: _subset_from_scene_ids(dataset, scene_ids) for split_name, scene_ids in split_map.items()}

    if split_config.get("shared_manifest_path"):
        from bise.data.rh20t.splits import shared_scene_split
        mapping = shared_scene_split([scene_id_from_record(s) for s in dataset.scenes], split_config)
        return {name: _subset_from_scene_ids(dataset, ids) for name, ids in mapping.items()}

    fixed_test_path = split_config.get("fixed_test_manifest_path")
    if fixed_test_path:
        return _split_with_fixed_test(dataset, split_config, resolve_path(fixed_test_path))

    unit = split_config.get("unit", "scene")
    ratios = split_config.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = int(split_config.get("seed", 42))
    return _split_by_group_unit(dataset, unit=unit, ratios=ratios, seed=seed)


def _is_all_test_split(split_config: Dict) -> bool:
    return bool(split_config.get("all_as_test")) or str(split_config.get("unit", "")).strip().lower() == "all_test"


def _all_as_test_split(dataset):
    indices = list(range(len(dataset)))
    return {
        "train": Subset(dataset, []),
        "val": Subset(dataset, []),
        "test": Subset(dataset, indices),
    }


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


def _split_with_fixed_test(dataset, split_config: Dict, fixed_test_path: Path):
    split_map = _load_split_manifest(fixed_test_path)
    fixed_test_ids = split_map.get("test")
    if not fixed_test_ids:
        raise ValueError("Fixed test manifest must contain a non-empty test split")

    id_to_index = {scene_id_from_record(scene): index for index, scene in enumerate(dataset.scenes)}
    missing = [scene_id for scene_id in fixed_test_ids if scene_id not in id_to_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} fixed test scenes are not in the dataset. Examples: {preview}")

    expected_fraction = float(split_config.get("fixed_test_fraction", 0.1))
    if not 0.0 < expected_fraction < 1.0:
        raise ValueError("fixed_test_fraction must be between zero and one")
    expected_count = int(round(len(dataset) * expected_fraction))
    if len(fixed_test_ids) != expected_count:
        raise ValueError(
            f"Fixed test split has {len(fixed_test_ids)} scenes; expected {expected_count} "
            f"for fraction {expected_fraction}"
        )

    test_indices = [id_to_index[scene_id] for scene_id in fixed_test_ids]
    test_index_set = set(test_indices)
    unit = split_config.get("unit", "scene")
    groups: Dict[str, List[int]] = {}
    test_group_keys = {_group_key(dataset.scenes[index], unit) for index in test_indices}
    for index, scene in enumerate(dataset.scenes):
        if index in test_index_set:
            continue
        key = _group_key(scene, unit)
        if key in test_group_keys:
            raise ValueError("A fixed test group also appears in the train/validation pool")
        groups.setdefault(key, []).append(index)

    val_fraction = float(split_config.get("val_ratio_of_remaining", 1.0 / 9.0))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_ratio_of_remaining must be between zero and one")
    group_keys = list(groups)
    rng = np.random.default_rng(int(split_config.get("seed", 42)))
    rng.shuffle(group_keys)
    val_count = int(round(len(group_keys) * val_fraction))
    if val_count < 1 or val_count >= len(group_keys):
        raise ValueError("Dynamic train/validation pool is too small for the requested ratio")
    val_keys = set(group_keys[:val_count])
    train_indices = [index for key, indices in groups.items() if key not in val_keys for index in indices]
    val_indices = [index for key, indices in groups.items() if key in val_keys for index in indices]
    return {
        "train": Subset(dataset, train_indices),
        "val": Subset(dataset, val_indices),
        "test": Subset(dataset, test_indices),
    }


def _group_key(scene_record, unit: str) -> str:
    normalized = str(unit).strip().lower()
    if normalized == "scene":
        return scene_id_from_record(scene_record)
    if normalized == "task":
        return task_id_from_record(scene_record)
    raise ValueError(f"Unsupported trajectory split unit: {unit}")


def _load_split_manifest(path: Path):
    from bise.data.rh20t.splits import load_split_map
    return load_split_map(path)


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


def build_trajectory_dataset(config: dict, checkpoint: str | None = None):
    from bise.data.rh20t.trajectory_dataset import RH20TTrajectoryDataset
    from bise.data.rh20t.hand_3d_trajectory_dataset import RH20THand3DTrajectoryDataset

    params = dict(config.get("data_params") or {})
    human_source = params.pop("human_source", "human_pose")
    geometry_paths = params.pop("geometry_manifests", None)
    if config["model_params"].get("grasp_geometry"):
        from bise.data.rh20t.grasp_geometry import geometry_manifest
        if (human_source != "hand_3d_base" or config.get("use_6_keypoints")
                or not geometry_paths):
            raise ValueError("Grasp geometry needs full 21-point hands and prepared manifests")
        params["geometry_manifest_path"] = str(geometry_manifest(config))
    if human_source not in {"human_pose", "hand_3d_base"}:
        raise ValueError(f"Unknown trajectory human source: {human_source}")
    manifests = config.get("data_manifests") or {}
    root_name = Path(config["data_root"]).name
    if manifests:
        if root_name not in manifests:
            raise ValueError(f"No trajectory data manifest configured for {root_name}")
        params["manifest_path"] = manifests[root_name]
    if checkpoint is not None:
        snapshot = Path(checkpoint).resolve().parent / "data_manifest.json"
        if snapshot.exists():
            frozen = json.loads(snapshot.read_text())
            if Path(frozen["data_root"]).resolve() == Path(config["data_root"]).resolve():
                params["manifest_path"] = str(snapshot)
    if params.get("max_robot_frames") is not None and config["model_params"].get("tcp_sample_factor", 1) != 1:
        raise ValueError("Dataset sampling cannot be combined with model TCP subsampling")
    if bool(params.get("preserve_timing")) != bool(config["model_params"].get("temporal_features")):
        raise ValueError("preserve_timing and temporal_features must be enabled together")
    dataset_type = RH20THand3DTrajectoryDataset if human_source == "hand_3d_base" else RH20TTrajectoryDataset
    return dataset_type(config["data_root"], config.get("use_6_keypoints", False), **params)


def build_trajectory_model(config: dict, device):
    from .models import CrossModalTrajectoryModel

    params = dict(config["model_params"])
    frames = config.get("reference_frames") or {}
    if frames:
        root_name = Path(config["data_root"]).name
        if root_name not in frames or not frames[root_name].get("source"):
            raise ValueError(f"A sourced TCP-to-reference transform is required for {root_name}")
        params["tcp_to_reference"] = frames[root_name]["tcp_to_reference"]
    return CrossModalTrajectoryModel(**params).to(device)
