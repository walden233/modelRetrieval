from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, load_jsonl
from .schemas import FeatureRecord, GalleryItem, QueryItem


@dataclass
class RetrievalLibrary:
    root_dir: Path
    config: dict[str, Any]

    #如"video_human": "features/video_human.npy",
    build_info: dict[str, Any]

    #特征等数量统计
    coverage: dict[str, Any]
    
    feature_records: dict[str, FeatureRecord]
    gallery_items: list[GalleryItem]
    query_items: list[QueryItem]
    arrays: dict[str, np.ndarray]

    def get_feature(self, feature_id: str) -> np.ndarray | None:
        # 通过 FeatureRecord 定位到 npy 文件和行号；缺失时返回 None，让上层按缺失模态处理。
        record = self.feature_records.get(feature_id)
        if record is None:
            return None
        array = self.arrays.get(record.array_path)
        if array is None:
            return None
        if record.row_index < 0 or record.row_index >= len(array):
            return None
        return np.asarray(array[record.row_index], dtype=np.float32)

    def item_feature_map(self, item: GalleryItem | QueryItem) -> dict[str, np.ndarray]:
        # 把一个 gallery/query item 的 feature_ids 转成真正的向量字典，并统一做 L2 normalize。
        features = {}
        for modality, feature_id in item.feature_ids.items():
            vector = self.get_feature(feature_id)
            if vector is not None:
                features[modality] = _normalize(vector)
        return features


def load_retrieval_library(root_dir: str | Path) -> RetrievalLibrary:
    # 检索库加载只依赖 manifests + features。
    root = Path(root_dir)
    manifest_dir = root / "manifests"
    feature_records = {
        str(record["feature_id"]): FeatureRecord.from_dict(record)
        for record in load_jsonl(manifest_dir / "feature_records.jsonl")
    }
    arrays = _load_feature_arrays(root, feature_records.values())
    return RetrievalLibrary(
        root_dir=root,
        config=load_json(root / "library_config.json", default={}) or {},
        build_info=load_json(root / "build_info.json", default={}) or {},
        coverage=load_json(root / "coverage.json", default={}) or {},
        feature_records=feature_records,
        gallery_items=[GalleryItem.from_dict(record) for record in load_jsonl(manifest_dir / "gallery_robot.jsonl")],
        query_items=[QueryItem.from_dict(record) for record in load_jsonl(manifest_dir / "query_human_eval.jsonl")],
        arrays=arrays,
    )


def _load_feature_arrays(root: Path, records) -> dict[str, np.ndarray]:
    arrays = {}
    for record in records:
        array_path = str(record.array_path)
        if array_path in arrays:
            continue
        path = Path(array_path)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            arrays[array_path] = np.load(path)
    return arrays


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)
