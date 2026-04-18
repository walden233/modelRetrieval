import argparse
import json
from typing import Iterable, List, Tuple

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from bise.modalities.semantic.evaluator import evaluate_semantic_retrieval_by_task, load_jsonl, split_cross_role_annotations
from bise.modalities.semantic.schemas import SemanticAnnotation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate text/label semantic retrieval.")
    parser.add_argument("--query-annotations", required=True, help="Query annotations JSONL path.")
    parser.add_argument("--gallery-annotations", help="Gallery annotations JSONL path. Defaults to query annotations.")
    parser.add_argument("--exclude-self", action="store_true", help="Exclude identical sample ids from retrieval.")
    return parser.parse_args()


def main():
    args = parse_args()
    query_annotations = _load_annotations(args.query_annotations)
    gallery_annotations = _load_annotations(args.gallery_annotations or args.query_annotations)
    if not args.gallery_annotations:
        query_annotations, gallery_annotations = split_cross_role_annotations(query_annotations)
    text_metrics = _evaluate(query_annotations, gallery_annotations, "text", args.exclude_self)
    label_metrics = _evaluate(query_annotations, gallery_annotations, "label", args.exclude_self)
    combined_metrics = _evaluate(query_annotations, gallery_annotations, "combined", args.exclude_self)
    print(
        json.dumps(
            {
                "text_only": text_metrics,
                "label_only": label_metrics,
                "text_plus_label": combined_metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _load_annotations(path: str) -> List[SemanticAnnotation]:
    return [SemanticAnnotation.from_dict(record) for record in load_jsonl(path)]


def _evaluate(
    query_annotations: List[SemanticAnnotation],
    gallery_annotations: List[SemanticAnnotation],
    mode: str,
    exclude_self: bool,
):
    query_embeddings, query_task_ids, query_sample_ids = _extract_embeddings(query_annotations, mode)
    gallery_embeddings, gallery_task_ids, gallery_sample_ids = _extract_embeddings(gallery_annotations, mode)
    if exclude_self and query_sample_ids == gallery_sample_ids:
        similarity = query_embeddings @ gallery_embeddings.T
        np.fill_diagonal(similarity, -np.inf)
        return _evaluate_from_similarity(similarity, query_task_ids, gallery_task_ids)
    return evaluate_semantic_retrieval_by_task(query_embeddings, query_task_ids, gallery_embeddings, gallery_task_ids)


def _extract_embeddings(annotations: Iterable[SemanticAnnotation], mode: str) -> Tuple[np.ndarray, List[str], List[str]]:
    embeddings: List[List[float]] = []
    task_ids: List[str] = []
    sample_ids: List[str] = []
    for annotation in annotations:
        text_embedding = annotation.metadata["text_embedding"]
        label_embedding = annotation.metadata["label_embedding"]
        if mode == "text":
            vector = np.asarray(text_embedding, dtype=np.float32)
        elif mode == "label":
            vector = np.asarray(label_embedding, dtype=np.float32)
        else:
            vector = 0.5 * (np.asarray(text_embedding, dtype=np.float32) + np.asarray(label_embedding, dtype=np.float32))
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        embeddings.append(vector.tolist())
        task_ids.append(annotation.task_id)
        sample_ids.append(annotation.sample_id)
    return np.asarray(embeddings, dtype=np.float32), task_ids, sample_ids


def _evaluate_from_similarity(similarity: np.ndarray, query_task_ids: List[str], gallery_task_ids: List[str]):
    ranks: List[int] = []
    for index, task_id in enumerate(query_task_ids):
        positives = {gallery_index for gallery_index, candidate_task_id in enumerate(gallery_task_ids) if candidate_task_id == task_id}
        if not positives:
            continue
        sorted_indices = np.argsort(-similarity[index])
        for rank, candidate_index in enumerate(sorted_indices, start=1):
            if candidate_index in positives:
                ranks.append(rank)
                break
    if not ranks:
        return {"R@1": 0.0, "R@5": 0.0, "R@10": 0.0, "MRR": 0.0}
    rank_array = np.asarray(ranks)
    return {
        "R@1": float(np.mean(rank_array <= 1)),
        "R@5": float(np.mean(rank_array <= 5)),
        "R@10": float(np.mean(rank_array <= 10)),
        "MRR": float(np.mean(1.0 / rank_array)),
    }


if __name__ == "__main__":
    main()
