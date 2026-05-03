from __future__ import annotations

from typing import Any

import numpy as np

from bise.retrieval.metrics import calculate_label_retrieval_metrics

from .library import RetrievalLibrary
from .scoring import group_gallery_scores_by_scene, score_query_against_gallery


def evaluate_retrieval_system(
    library: RetrievalLibrary,
    config: dict[str, Any],
    level: str = "scene",
    require_modalities: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    # 系统级评估模拟真实查询：human eval query 逐条检索 robot gallery，而不是直接复用单模态 final_test。
    queries = _filter_queries(library, config, require_modalities)
    aggregation = str((config.get("aggregation") or {}).get("camera_to_scene", "max"))
    query_labels = []
    query_task_ids = []
    query_scene_ids = []
    similarity_rows = []
    task_score_rows = []
    modality_rows: dict[str, list[np.ndarray]] = {}
    scene_items = None
    cases = []

    for query in queries:
        # 每个 query 先按当前可用模态检索全部 robot gallery，再聚合到 scene 级别计算指标。
        query_features = library.item_feature_map(query)
        fused_scores, modality_scores = score_query_against_gallery(library, query_features, config=config)
        current_scene_items, scene_scores, scene_modality_scores = group_gallery_scores_by_scene(
            library,
            fused_scores,
            modality_scores,
            aggregation=aggregation,
        )
        if scene_items is None:
            scene_items = current_scene_items
        gallery_task_ids = [item.task_id for item in current_scene_items]
        gallery_scene_ids = [item.scene_id for item in current_scene_items]
        query_labels.append(query.scene_id if level == "scene" else query.task_id)
        query_task_ids.append(query.task_id)
        query_scene_ids.append(query.scene_id)
        similarity_rows.append(scene_scores)
        task_score_rows.append(scene_scores)
        for modality, values in scene_modality_scores.items():
            modality_rows.setdefault(modality, []).append(values)
        cases.append(
            _build_case(
                query=query,
                scene_items=current_scene_items,
                scene_scores=scene_scores,
                scene_modality_scores=scene_modality_scores,
                gallery_task_ids=gallery_task_ids,
                gallery_scene_ids=gallery_scene_ids,
                top_k=top_k,
            )
        )

    if scene_items is None:
        return {
            "metrics": _empty_metrics(level),
            "cases": [],
            "query_count": 0,
            "gallery_count": 0,
        }

    similarity_matrix = np.vstack(similarity_rows).astype(np.float32)
    gallery_task_ids = [item.task_id for item in scene_items]
    gallery_scene_ids = [item.scene_id for item in scene_items]
    metrics = {
        "scene": calculate_label_retrieval_metrics(similarity_matrix, query_scene_ids, gallery_scene_ids),
        "task": calculate_label_retrieval_metrics(similarity_matrix, query_task_ids, gallery_task_ids),
        "mixed": calculate_mixed_retrieval_metrics(
            similarity_matrix=similarity_matrix,
            query_task_ids=query_task_ids,
            query_scene_ids=query_scene_ids,
            gallery_task_ids=gallery_task_ids,
            gallery_scene_ids=gallery_scene_ids,
            scene_gain=float((config.get("scene_task_mixed") or {}).get("scene_gain", 1.0)),
            task_gain=float((config.get("scene_task_mixed") or {}).get("task_gain", 0.3)),
        ),
    }
    return {
        "metrics": metrics,
        "selected_level": level,
        "cases": cases,
        "query_count": len(queries),
        "gallery_count": len(scene_items),
        "query_ids": [query.query_id for query in queries],
        "gallery_ids": [item.gallery_id for item in scene_items],
        "similarity_matrix": similarity_matrix,
        "modality_matrices": {name: np.vstack(rows).astype(np.float32) for name, rows in modality_rows.items()},
    }


def calculate_mixed_retrieval_metrics(
    similarity_matrix: np.ndarray,
    query_task_ids: list[str],
    query_scene_ids: list[str],
    gallery_task_ids: list[str],
    gallery_scene_ids: list[str],
    scene_gain: float = 1.0,
    task_gain: float = 0.3,
    ndcg_k: int = 10,
) -> dict[str, float]:
    # mixed 只定义评价相关性：同 scene 满分，同 task 低分；它不参与排序，避免标签泄漏。
    scene_hits_1 = scene_hits_5 = task_hits_5 = task_only_hits_5 = 0
    ndcg_scores = []
    valid_queries = 0
    for query_index, scores in enumerate(similarity_matrix):
        relevance = np.zeros(len(gallery_scene_ids), dtype=np.float32)
        same_scene = np.asarray(gallery_scene_ids) == query_scene_ids[query_index]
        same_task = np.asarray(gallery_task_ids) == query_task_ids[query_index]
        relevance[same_task] = task_gain
        relevance[same_scene] = scene_gain
        if not np.any(relevance > 0):
            continue
        valid_queries += 1
        ranked_indices = np.argsort(-scores)
        top1 = ranked_indices[:1]
        top5 = ranked_indices[:5]
        scene_hits_1 += int(np.any(same_scene[top1]))
        scene_hits_5 += int(np.any(same_scene[top5]))
        task_hits_5 += int(np.any(same_task[top5]))
        task_only_hits_5 += int((not np.any(same_scene[top5])) and np.any(same_task[top5]))
        top_indices = ranked_indices[: min(ndcg_k, len(ranked_indices))]
        discounts = 1.0 / np.log2(np.arange(2, len(top_indices) + 2))
        dcg = float(np.sum(relevance[top_indices] * discounts))
        ideal_relevance = np.sort(relevance)[::-1][: len(top_indices)]
        ideal_dcg = float(np.sum(ideal_relevance * discounts))
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    if valid_queries == 0:
        return {
            "MixedNDCG@10": 0.0,
            "SceneHit@1": 0.0,
            "SceneHit@5": 0.0,
            "TaskHit@5": 0.0,
            "TaskOnlyHit@5": 0.0,
            "valid_queries": 0,
        }
    return {
        "MixedNDCG@10": float(np.mean(ndcg_scores)),
        "SceneHit@1": scene_hits_1 / valid_queries,
        "SceneHit@5": scene_hits_5 / valid_queries,
        "TaskHit@5": task_hits_5 / valid_queries,
        "TaskOnlyHit@5": task_only_hits_5 / valid_queries,
        "valid_queries": int(valid_queries),
    }


def _filter_queries(library: RetrievalLibrary, config: dict[str, Any], require_modalities: list[str] | None):
    # require_modalities 用于实验控制，例如只评估 video/trajectory/semantic_text 都完整的 query。
    enabled = set(config.get("modalities") or [])
    required = set(require_modalities or [])
    queries = []
    for query in library.query_items:
        features = library.item_feature_map(query)
        available = set(features)
        if enabled and not available.intersection(enabled):
            continue
        if required and not required.issubset(available):
            continue
        queries.append(query)
    return queries


def _build_case(
    query,
    scene_items,
    scene_scores,
    scene_modality_scores,
    gallery_task_ids,
    gallery_scene_ids,
    top_k: int,
):
    ranked_indices = np.argsort(-scene_scores)[:top_k]
    return {
        "query_id": query.query_id,
        "query_task_id": query.task_id,
        "query_scene_id": query.scene_id,
        "retrieved": [
            {
                "rank": rank,
                "gallery_id": scene_items[int(index)].gallery_id,
                "task_id": gallery_task_ids[int(index)],
                "scene_id": gallery_scene_ids[int(index)],
                "camera_id": scene_items[int(index)].camera_id,
                "fused_score": float(scene_scores[int(index)]),
                "modality_scores": {
                    modality: float(values[int(index)])
                    for modality, values in scene_modality_scores.items()
                    if np.isfinite(values[int(index)])
                },
                "is_scene_positive": bool(gallery_scene_ids[int(index)] == query.scene_id),
                "is_task_positive": bool(gallery_task_ids[int(index)] == query.task_id),
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ],
    }


def _empty_metrics(level: str) -> dict[str, Any]:
    empty = {
        "R@1": 0.0,
        "R@5": 0.0,
        "R@10": 0.0,
        "Mean Rank": 0.0,
        "MRR": 0.0,
        "Mean Percentage Rank": 0.0,
        "NDCG@10": 0.0,
        "valid_queries": 0,
    }
    return {"scene": dict(empty), "task": dict(empty), "mixed": calculate_mixed_retrieval_metrics(np.zeros((0, 0)), [], [], [], [])}
