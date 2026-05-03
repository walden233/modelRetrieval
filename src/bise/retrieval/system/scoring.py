from __future__ import annotations

from typing import Any

import numpy as np

from .library import RetrievalLibrary
from .schemas import GalleryItem, RetrievalQuery, RetrievalResult


DEFAULT_MODALITIES = ("video", "trajectory", "semantic_text")


def retrieve_top_k(
    library: RetrievalLibrary,
    query: RetrievalQuery | dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
    top_k: int = 10,
) -> list[RetrievalResult]:
    # 在线检索主入口：query 可以缺任意模态；只对 query 和 gallery 都有的模态打分。
    config = config or {}
    query_id = query.query_id if isinstance(query, RetrievalQuery) else str(config.get("query_id", "query"))
    query_features = _coerce_query_features(query)
    enabled_modalities = _enabled_modalities(config, query_features)
    if not enabled_modalities:
        raise ValueError("At least one query modality must be available.")

    weights = _weights(config, enabled_modalities)
    raw_scores = _score_gallery(library, query_features, enabled_modalities)
    # 各模态分数分布不同，融合前先做 query-wise 校准，避免某个模态因尺度大而主导排序。
    calibrated_scores = _calibrate_by_modality(raw_scores, method=_fusion_config(config).get("calibration", "zscore"))
    fused_scores = _fuse_scores(calibrated_scores, weights, missing_policy=_fusion_config(config).get("missing_policy", "renormalize"))
    modality_ranks = _modality_ranks(calibrated_scores)
    order = np.argsort(-fused_scores)
    results = []
    for gallery_index in order[:top_k]:
        if not np.isfinite(fused_scores[gallery_index]):
            continue
        item = library.gallery_items[int(gallery_index)]
        results.append(
            RetrievalResult(
                query_id=query_id or "query",
                gallery_id=item.gallery_id,
                entity_key=item.entity_key,
                task_id=item.task_id,
                scene_id=item.scene_id,
                camera_id=item.camera_id,
                fused_score=float(fused_scores[gallery_index]),
                modality_scores={
                    modality: float(scores[gallery_index])
                    for modality, scores in calibrated_scores.items()
                    if np.isfinite(scores[gallery_index])
                },
                modality_ranks={
                    modality: int(ranks[gallery_index])
                    for modality, ranks in modality_ranks.items()
                    if ranks[gallery_index] > 0
                },
                metadata=dict(item.metadata),
            )
        )
    return results


def score_query_against_gallery(
    library: RetrievalLibrary,
    query_features: dict[str, np.ndarray],
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = config or {}
    enabled_modalities = _enabled_modalities(config, query_features)
    weights = _weights(config, enabled_modalities)
    raw_scores = _score_gallery(library, query_features, enabled_modalities)
    calibrated_scores = _calibrate_by_modality(raw_scores, method=_fusion_config(config).get("calibration", "zscore"))
    fused_scores = _fuse_scores(calibrated_scores, weights, missing_policy=_fusion_config(config).get("missing_policy", "renormalize"))
    return fused_scores, calibrated_scores


def group_gallery_scores_by_scene(
    library: RetrievalLibrary,
    fused_scores: np.ndarray,
    modality_scores: dict[str, np.ndarray],
    aggregation: str = "max",
) -> tuple[list[GalleryItem], np.ndarray, dict[str, np.ndarray]]:
    # gallery 可以是 camera-level；scene-level 评估时需要把同一 scene 下多个 camera 聚合成一个候选。
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(library.gallery_items):
        groups.setdefault(item.entity_key, []).append(index)
    scene_items = []
    scene_scores = []
    scene_modality_scores = {name: [] for name in modality_scores}
    for entity_key in sorted(groups):
        indices = groups[entity_key]
        representative = library.gallery_items[indices[0]]
        scene_items.append(representative)
        scene_scores.append(_aggregate(fused_scores[indices], aggregation))
        for modality, scores in modality_scores.items():
            scene_modality_scores[modality].append(_aggregate(scores[indices], aggregation))
    return (
        scene_items,
        np.asarray(scene_scores, dtype=np.float32),
        {name: np.asarray(values, dtype=np.float32) for name, values in scene_modality_scores.items()},
    )


def calibrate_scores(scores: np.ndarray, method: str = "zscore") -> np.ndarray:
    # 校准只在当前 query 的 gallery 分数上进行，不使用标签，因此不会造成评估泄漏。
    values = np.asarray(scores, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return values
    normalized = values.copy()
    method = str(method).strip().lower()
    valid = values[finite_mask]
    if method == "none":
        return normalized
    if method == "zscore":
        std = float(valid.std())
        if std <= 1e-12:
            normalized[finite_mask] = 0.0
        else:
            normalized[finite_mask] = (valid - float(valid.mean())) / std
        return normalized
    if method == "minmax":
        span = float(valid.max() - valid.min())
        if span <= 1e-12:
            normalized[finite_mask] = 0.0
        else:
            normalized[finite_mask] = (valid - float(valid.min())) / span
        return normalized
    if method == "rank":
        order = np.argsort(-valid)
        rank_scores = np.zeros_like(valid, dtype=np.float32)
        rank_scores[order] = 1.0 / (np.arange(len(valid), dtype=np.float32) + 1.0)
        normalized[finite_mask] = rank_scores
        return normalized
    raise ValueError(f"Unsupported calibration method: {method}")


def _coerce_query_features(query: RetrievalQuery | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(query, RetrievalQuery):
        raw = query.to_feature_map()
    else:
        raw = query
    features = {}
    for modality, vector in raw.items():
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        features[modality] = array / norm if norm > 0 else array
    return features


def _enabled_modalities(config: dict[str, Any], query_features: dict[str, np.ndarray]) -> list[str]:
    requested = config.get("modalities") or list(DEFAULT_MODALITIES)
    return [name for name in requested if name in query_features]


def _fusion_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("fusion") or {})


def _weights(config: dict[str, Any], modalities: list[str]) -> dict[str, float]:
    raw_weights = dict(_fusion_config(config).get("weights") or {})
    if not raw_weights:
        return {name: 1.0 / len(modalities) for name in modalities}
    return {name: float(raw_weights.get(name, 0.0)) for name in modalities}


def _score_gallery(library: RetrievalLibrary, query_features: dict[str, np.ndarray], modalities: list[str]) -> dict[str, np.ndarray]:
    # 分模态打分：某候选缺少该模态或维度不匹配时记为 -inf，后续融合时自动忽略。
    scores = {name: np.full(len(library.gallery_items), -np.inf, dtype=np.float32) for name in modalities}
    for gallery_index, item in enumerate(library.gallery_items):
        gallery_features = library.item_feature_map(item)
        for modality in modalities:
            if modality not in gallery_features:
                continue
            query_vector = query_features[modality]
            gallery_vector = gallery_features[modality]
            if query_vector.shape != gallery_vector.shape:
                continue
            scores[modality][gallery_index] = float(np.dot(query_vector, gallery_vector))
    return scores


def _calibrate_by_modality(scores: dict[str, np.ndarray], method: str) -> dict[str, np.ndarray]:
    return {modality: calibrate_scores(values, method=method) for modality, values in scores.items()}


def _fuse_scores(scores: dict[str, np.ndarray], weights: dict[str, float], missing_policy: str) -> np.ndarray:
    # 缺失模态采用 renormalize：只在可用模态之间重新归一权重，保证单模态 query 也能检索。
    if not scores:
        return np.asarray([], dtype=np.float32)
    length = len(next(iter(scores.values())))
    fused = np.zeros(length, dtype=np.float32)
    weight_sums = np.zeros(length, dtype=np.float32)
    for modality, values in scores.items():
        weight = float(weights.get(modality, 0.0))
        finite = np.isfinite(values)
        fused[finite] += values[finite] * weight
        weight_sums[finite] += weight
    missing_policy = str(missing_policy).strip().lower()
    valid = weight_sums > 0
    if missing_policy == "renormalize":
        fused[valid] = fused[valid] / weight_sums[valid]
    fused[~valid] = -np.inf
    return fused


def _modality_ranks(scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ranks = {}
    for modality, values in scores.items():
        modality_ranks = np.zeros(len(values), dtype=np.int64)
        finite = np.isfinite(values)
        order = np.argsort(-values[finite])
        finite_indices = np.where(finite)[0]
        for rank, local_index in enumerate(order, start=1):
            modality_ranks[finite_indices[int(local_index)]] = rank
        ranks[modality] = modality_ranks
    return ranks


def _aggregate(values: np.ndarray, method: str) -> float:
    finite_values = np.asarray(values)[np.isfinite(values)]
    if len(finite_values) == 0:
        return float("-inf")
    method = str(method).strip().lower()
    if method == "max":
        return float(finite_values.max())
    if method == "mean":
        return float(finite_values.mean())
    if method == "first":
        return float(finite_values[0])
    raise ValueError(f"Unsupported aggregation method: {method}")
