from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from bise.modalities.semantic.schemas import DescriptionReviewRecord, LabelEvaluationRecord, SemanticAnnotation
from bise.retrieval.metrics import calculate_retrieval_metrics_grouped


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_label_predictions(
    predictions: Sequence[SemanticAnnotation],
    gold_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_map = {str(record["sample_id"]): {str(tag) for tag in record.get("capability_tags", [])} for record in gold_records}
    rows: List[LabelEvaluationRecord] = []
    total_tp = total_fp = total_fn = 0
    exact_matches = 0
    for prediction in predictions:
        gold_tags = gold_map.get(prediction.sample_id)
        if gold_tags is None:
            continue
        predicted_tags = set(prediction.capability_tags)
        tp = len(predicted_tags & gold_tags)
        fp = len(predicted_tags - gold_tags)
        fn = len(gold_tags - predicted_tags)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact_match = predicted_tags == gold_tags
        rows.append(
            LabelEvaluationRecord(
                sample_id=prediction.sample_id,
                predicted_tags=sorted(predicted_tags),
                gold_tags=sorted(gold_tags),
                precision=precision,
                recall=recall,
                f1=f1,
                exact_match=exact_match,
            )
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        exact_matches += int(exact_match)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "count": len(rows),
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "exact_match": exact_matches / len(rows) if rows else 0.0,
        "rows": [row.to_dict() for row in rows],
    }


def summarize_description_reviews(records: Sequence[Dict[str, Any] | DescriptionReviewRecord]) -> Dict[str, Any]:
    reviews = [record if isinstance(record, DescriptionReviewRecord) else DescriptionReviewRecord(**record) for record in records]
    total = len(reviews)
    if total == 0:
        return {"count": 0, "main_action_ok_rate": 0.0, "object_ok_rate": 0.0, "hallucination_free_rate": 0.0}
    return {
        "count": total,
        "main_action_ok_rate": sum(1 for review in reviews if review.main_action_ok) / total,
        "object_ok_rate": sum(1 for review in reviews if review.object_ok) / total,
        "hallucination_free_rate": sum(1 for review in reviews if review.hallucination_free) / total,
    }


def evaluate_semantic_retrieval(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    group_size: int = 1,
) -> Dict[str, float]:
    if query_embeddings.shape != gallery_embeddings.shape:
        raise ValueError("Query and gallery embedding matrices must share shape for grouped evaluation.")
    similarity_matrix = query_embeddings @ gallery_embeddings.T
    return calculate_retrieval_metrics_grouped(similarity_matrix, group_size=group_size)


def evaluate_semantic_retrieval_by_task(
    query_embeddings: np.ndarray,
    query_task_ids: Sequence[str],
    gallery_embeddings: np.ndarray,
    gallery_task_ids: Sequence[str],
) -> Dict[str, float]:
    if len(query_embeddings) != len(query_task_ids):
        raise ValueError("query_embeddings and query_task_ids length mismatch.")
    if len(gallery_embeddings) != len(gallery_task_ids):
        raise ValueError("gallery_embeddings and gallery_task_ids length mismatch.")
    similarity_matrix = query_embeddings @ gallery_embeddings.T
    ranks: List[int] = []
    for index, query_task_id in enumerate(query_task_ids):
        positives = {candidate_index for candidate_index, task_id in enumerate(gallery_task_ids) if task_id == query_task_id}
        if not positives:
            continue
        sorted_indices = np.argsort(-similarity_matrix[index])
        for rank, candidate_index in enumerate(sorted_indices, start=1):
            if candidate_index in positives:
                ranks.append(rank)
                break
    if not ranks:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0, "Mean Rank": 0.0}
    rank_array = np.asarray(ranks)
    return {
        "R@1": float(np.mean(rank_array <= 1)),
        "R@5": float(np.mean(rank_array <= 5)),
        "R@10": float(np.mean(rank_array <= 10)),
        "MRR": float(np.mean(1.0 / rank_array)),
        "Mean Rank": float(np.mean(rank_array)),
    }


def split_cross_role_annotations(
    annotations: Sequence[SemanticAnnotation],
) -> tuple[list[SemanticAnnotation], list[SemanticAnnotation]]:
    human_annotations = [annotation for annotation in annotations if annotation.video_role == "human"]
    robot_annotations = [annotation for annotation in annotations if annotation.video_role == "robot"]
    if human_annotations and robot_annotations:
        return human_annotations, robot_annotations
    return list(annotations), list(annotations)
