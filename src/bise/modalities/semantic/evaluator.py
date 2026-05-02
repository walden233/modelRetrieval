from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from bise.modalities.semantic.schemas import DescriptionReviewRecord, LabelEvaluationRecord, SemanticAnnotation
from bise.retrieval.metrics import calculate_label_retrieval_metrics, calculate_retrieval_metrics_grouped


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
    gold_map = {str(record["sample_id"]): record for record in gold_records}
    rows: List[LabelEvaluationRecord] = []
    total_tp = total_fp = total_fn = 0
    exact_matches = 0
    task_complexity_matches = 0
    scene_category_matches = 0
    environment_exact_matches = 0
    for prediction in predictions:
        gold_record = gold_map.get(prediction.sample_id)
        if gold_record is None:
            continue
        gold_tags = {str(tag) for tag in gold_record.get("capability_tags", [])}
        predicted_tags = set(prediction.capability_tags)
        tp = len(predicted_tags & gold_tags)
        fp = len(predicted_tags - gold_tags)
        fn = len(gold_tags - predicted_tags)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact_match = predicted_tags == gold_tags
        task_complexity_match = prediction.task_complexity == str(gold_record.get("task_complexity", "unknown"))
        scene_category_match = prediction.scene_category == str(gold_record.get("scene_category", "unknown"))
        gold_environment_tags = {str(tag) for tag in gold_record.get("environment_tags", [])}
        environment_exact_match = set(prediction.environment_tags) == gold_environment_tags
        rows.append(
            LabelEvaluationRecord(
                sample_id=prediction.sample_id,
                predicted_capability_tags=sorted(predicted_tags),
                gold_capability_tags=sorted(gold_tags),
                capability_precision=precision,
                capability_recall=recall,
                capability_f1=f1,
                capability_exact_match=exact_match,
                task_complexity_match=task_complexity_match,
                scene_category_match=scene_category_match,
                environment_exact_match=environment_exact_match,
            )
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        exact_matches += int(exact_match)
        task_complexity_matches += int(task_complexity_match)
        scene_category_matches += int(scene_category_match)
        environment_exact_matches += int(environment_exact_match)
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
        "capability_exact_match": exact_matches / len(rows) if rows else 0.0,
        "task_complexity_match_rate": task_complexity_matches / len(rows) if rows else 0.0,
        "scene_category_match_rate": scene_category_matches / len(rows) if rows else 0.0,
        "environment_exact_match_rate": environment_exact_matches / len(rows) if rows else 0.0,
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


def evaluate_semantic_retrieval_by_key(
    query_embeddings: np.ndarray,
    query_keys: Sequence[str],
    gallery_embeddings: np.ndarray,
    gallery_keys: Sequence[str],
    query_sample_ids: Sequence[str] | None = None,
    gallery_sample_ids: Sequence[str] | None = None,
    exclude_self: bool = False,
) -> Dict[str, float]:
    if len(query_embeddings) != len(query_keys):
        raise ValueError("query_embeddings and query_keys length mismatch.")
    if len(gallery_embeddings) != len(gallery_keys):
        raise ValueError("gallery_embeddings and gallery_keys length mismatch.")
    if query_sample_ids is not None and len(query_sample_ids) != len(query_embeddings):
        raise ValueError("query_sample_ids and query_embeddings length mismatch.")
    if gallery_sample_ids is not None and len(gallery_sample_ids) != len(gallery_embeddings):
        raise ValueError("gallery_sample_ids and gallery_embeddings length mismatch.")
    similarity_matrix = query_embeddings @ gallery_embeddings.T
    if exclude_self and query_sample_ids is not None and gallery_sample_ids is not None:
        similarity_matrix = similarity_matrix.copy()
        for query_index, query_sample_id in enumerate(query_sample_ids):
            for gallery_index, gallery_sample_id in enumerate(gallery_sample_ids):
                if gallery_sample_id == query_sample_id:
                    similarity_matrix[query_index, gallery_index] = -np.inf
    metrics = calculate_label_retrieval_metrics(similarity_matrix, query_keys, gallery_keys)
    for key, value in list(metrics.items()):
        if isinstance(value, float) and not np.isfinite(value):
            metrics[key] = 0.0
    metrics["query_count"] = len(query_keys)
    metrics["valid_query_count"] = metrics["valid_queries"]
    return metrics


def split_cross_role_annotations(
    annotations: Sequence[SemanticAnnotation],
) -> tuple[list[SemanticAnnotation], list[SemanticAnnotation]]:
    return split_annotations_by_role(annotations, query_role="human", gallery_role="robot")


def split_annotations_by_role(
    annotations: Sequence[SemanticAnnotation],
    query_role: str,
    gallery_role: str,
) -> tuple[list[SemanticAnnotation], list[SemanticAnnotation]]:
    normalized_query_role = str(query_role).strip().lower()
    normalized_gallery_role = str(gallery_role).strip().lower()
    query_annotations = [annotation for annotation in annotations if annotation.video_role.strip().lower() == normalized_query_role]
    gallery_annotations = [annotation for annotation in annotations if annotation.video_role.strip().lower() == normalized_gallery_role]
    query_annotations = _sort_annotations_for_retrieval(query_annotations)
    gallery_annotations = _sort_annotations_for_retrieval(gallery_annotations)
    if query_annotations and gallery_annotations:
        return query_annotations, gallery_annotations
    return list(annotations), list(annotations)


def extract_annotation_embeddings(
    annotations: Sequence[SemanticAnnotation],
    mode: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    embeddings: List[np.ndarray] = []
    keys: List[str] = []
    sample_ids: List[str] = []
    normalized_mode = str(mode).strip().lower()
    for annotation in annotations:
        metadata = annotation.metadata
        text_embedding = metadata.get("text_embedding")
        label_embedding = metadata.get("label_embedding")
        if text_embedding is None or label_embedding is None:
            raise ValueError(f"Annotation {annotation.sample_id} is missing text_embedding or label_embedding in metadata.")
        text_vector = _normalize_vector(np.asarray(text_embedding, dtype=np.float32))
        label_vector = _normalize_vector(np.asarray(label_embedding, dtype=np.float32))
        if normalized_mode == "text":
            vector = text_vector
        elif normalized_mode == "label":
            vector = label_vector
        elif normalized_mode == "combined":
            vector = _normalize_vector(text_vector + label_vector)
        else:
            raise ValueError("mode must be one of: text, label, combined")
        embeddings.append(vector)
        sample_ids.append(annotation.sample_id)
        keys.append(annotation.pair_id)
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32), keys, sample_ids
    return np.vstack(embeddings).astype(np.float32), keys, sample_ids


def evaluate_annotation_retrieval(
    query_annotations: Sequence[SemanticAnnotation],
    gallery_annotations: Sequence[SemanticAnnotation],
    mode: str,
    positive_key: str = "pair_id",
    exclude_self: bool = False,
) -> Dict[str, float]:
    query_embeddings, _, query_sample_ids = extract_annotation_embeddings(query_annotations, mode)
    gallery_embeddings, _, gallery_sample_ids = extract_annotation_embeddings(gallery_annotations, mode)
    query_keys = extract_positive_keys(query_annotations, positive_key)
    gallery_keys = extract_positive_keys(gallery_annotations, positive_key)
    return evaluate_semantic_retrieval_by_key(
        query_embeddings=query_embeddings,
        query_keys=query_keys,
        gallery_embeddings=gallery_embeddings,
        gallery_keys=gallery_keys,
        query_sample_ids=query_sample_ids,
        gallery_sample_ids=gallery_sample_ids,
        exclude_self=exclude_self,
    )


def extract_positive_keys(
    annotations: Sequence[SemanticAnnotation],
    positive_key: str,
) -> list[str]:
    key_name = str(positive_key).strip()
    allowed = {"pair_id", "task_id", "scene_id", "sample_id"}
    if key_name not in allowed:
        raise ValueError(f"positive_key must be one of: {', '.join(sorted(allowed))}")
    if key_name == "scene_id":
        return [_task_scoped_scene_id(annotation) for annotation in annotations]
    return [str(getattr(annotation, key_name)) for annotation in annotations]


def _task_scoped_scene_id(annotation: SemanticAnnotation) -> str:
    scene_id = str(annotation.scene_id)
    task_id = str(annotation.task_id)
    if not task_id or scene_id.startswith(f"{task_id}/"):
        return scene_id
    return f"{task_id}/{scene_id}"


def load_semantic_annotations(path: str | Path) -> List[SemanticAnnotation]:
    annotations = [SemanticAnnotation.from_dict(record) for record in load_jsonl(path)]
    return _sort_annotations_for_retrieval([annotation for annotation in annotations if annotation.status == "success"])


def _sort_annotations_for_retrieval(annotations: Sequence[SemanticAnnotation]) -> list[SemanticAnnotation]:
    return sorted(
        annotations,
        key=lambda item: (
            item.pair_id,
            item.video_role,
            item.sample_id,
        ),
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return vector
    return vector / norm
