import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from torch.utils.data import Subset

from bise.common import resolve_path
from bise.data.rh20t.video_dataset import RH20TVideoDataset
from bise.data.whirl.video_pair_dataset import VideoPairDataset

from .models.backbone_registry import build_video_backbone
from .models.cross_domain_video_encoder import CrossDomainVideoEncoder


def build_video_model(model_config: Dict) -> tuple[object, CrossDomainVideoEncoder]:
    # 工厂函数：根据配置先构建底层 backbone，再包装成统一的视频编码器。
    processor, backbone_adapter = build_video_backbone(
        backbone_type=model_config["backbone_type"],
        model_name=model_config["backbone_name"],
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    model = CrossDomainVideoEncoder(
        backbone_adapter=backbone_adapter,
        feature_dim=model_config.get("feature_dim", 128),
        encoder_mode=model_config.get("encoder_mode", "shared"),
        adapter_bottleneck_dim=model_config.get("adapter_bottleneck_dim"),
        dropout=model_config.get("dropout", 0.1),
        temperature=model_config.get("temperature", 0.07),
        intra_temperature=model_config.get("intra_temperature"),
        freeze_backbone=model_config.get("freeze_backbone", False),
        unfreeze_last_n_blocks=model_config.get("unfreeze_last_n_blocks", 0),
        freeze_patch_embed=model_config.get("freeze_patch_embed", False),
        freeze_norm_layers=model_config.get("freeze_norm_layers", False),
    )
    return processor, model


def build_video_dataset(dataset_config: Dict, processor, is_train: bool):
    # 统一不同数据源的构造方式，对外只暴露一个 build 入口。
    dataset_type = str(dataset_config["type"]).strip().lower()
    common_kwargs = {
        "processor": processor,
        "num_frames": dataset_config.get("num_frames", 16),
        "sampling_strategy": dataset_config.get("sampling_strategy", "uniform"),
        "sampling_stride": dataset_config.get("sampling_stride"),
        "deterministic": dataset_config.get("deterministic", not is_train),
        "seed": dataset_config.get("seed", 42),
        "transform_config": dataset_config.get("train_augmentations") if is_train else dataset_config.get("eval_augmentations"),
        "debug_max_samples": dataset_config.get("debug_max_samples"),
    }

    if dataset_type == "rh20t":
        return RH20TVideoDataset(
            root_dir=str(resolve_path(dataset_config["root_dir"])),
            max_pairs_per_scene=dataset_config.get("max_pairs_per_scene"),
            **common_kwargs,
        )
    if dataset_type == "whirl":
        return VideoPairDataset(
            csv_file=str(resolve_path(dataset_config["csv_path"])),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported video dataset type: {dataset_type}")


def split_video_dataset(dataset, split_config: Dict | None):
    # 如果没有 split 配置，就默认整份数据只作为训练集。
    if not split_config:
        return {"train": dataset, "val": None, "test": None}

    manifest_path = split_config.get("manifest_path")
    if manifest_path:
        # 若用户已经准备好了固定划分文件，则优先按 manifest 切分。
        split_map = _load_split_manifest(resolve_path(manifest_path))
        return {
            split_name: _subset_from_sample_ids(dataset, sample_ids)
            for split_name, sample_ids in split_map.items()
        }

    unit = split_config.get("unit", "sample")
    ratios = split_config.get("ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    seed = int(split_config.get("seed", 42))
    return _split_by_group_unit(dataset, unit=unit, ratios=ratios, seed=seed)


def _split_by_group_unit(dataset, unit: str, ratios: Dict[str, float], seed: int):
    # 按 sample / scene / task 三种粒度切分，避免同一组样本泄漏到不同 split。
    # sample：按camera sample_id 划分，同一个 scene 的不同 camera 可能被分到 train 和 test
    # scene：按场景划分，适合同一场景内样本
    # task：按任务划分，最粗粒度，适合评估模型
    groups: Dict[str, List[int]] = {}
    for index, sample in enumerate(dataset.samples):
        key = _group_key(sample, unit)
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
        split_name: Subset(
            dataset,
            [index for key in keys for index in groups[key]],
        )
        for split_name, keys in split_groups.items()
    }


def _group_key(sample, unit: str) -> str:
    # 这里决定“按什么单位做数据划分”。
    if unit == "sample":
        return sample.sample_id
    if unit == "scene":
        return sample.scene_id
    if unit == "task":
        return sample.task_id
    raise ValueError(f"Unsupported split unit: {unit}")


def _load_split_manifest(path: Path):
    # 支持 json / csv / tsv 三种 manifest 格式。
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {name: list(values) for name, values in payload.items()}
        raise ValueError("JSON split manifest must be an object with split keys.")

    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        mapping: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row in reader:
                split_name = str(row["split"]).strip()
                mapping.setdefault(split_name, []).append(str(row["sample_id"]).strip())
        return mapping

    raise ValueError(f"Unsupported split manifest format: {path.suffix}")


def _subset_from_sample_ids(dataset, sample_ids: Iterable[str]):
    # 根据 sample_id 映射回 dataset 中的实际索引。
    sample_id_set = set(sample_ids)
    indices = [index for index, sample in enumerate(dataset.samples) if sample.sample_id in sample_id_set]
    return Subset(dataset, indices)
