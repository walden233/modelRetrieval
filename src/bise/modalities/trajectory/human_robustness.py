"""Compare transformed human queries against one frozen robot gallery."""

import numpy as np

from bise.retrieval.metrics import calculate_label_retrieval_metrics


def fixed_gallery_comparison(baseline: dict, transformed_human: np.ndarray, top_k: int = 5) -> dict:
    original = np.asarray(baseline["human_embeddings"], dtype=np.float64)
    changed = np.asarray(transformed_human, dtype=np.float64)
    gallery = np.asarray(baseline["robot_embeddings"], dtype=np.float64)
    if changed.shape != original.shape or original.ndim != 2 or gallery.ndim != 2:
        raise ValueError("Human query embeddings must match the baseline shape")
    if original.shape[1] != gallery.shape[1] or not np.isfinite(changed).all():
        raise ValueError("Human and fixed robot embeddings must be finite and aligned")
    baseline_scores = original @ gallery.T
    changed_scores = changed @ gallery.T
    k = min(top_k, len(gallery))
    if k < 1:
        raise ValueError("The fixed robot gallery is empty")
    baseline_order = np.argsort(-baseline_scores, axis=1, kind="stable")[:, :k]
    changed_order = np.argsort(-changed_scores, axis=1, kind="stable")[:, :k]
    overlap = [len(set(a) & set(b)) / k for a, b in zip(baseline_order, changed_order)]
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(changed, axis=1)
    cosine = np.sum(original * changed, axis=1) / np.maximum(denominator, 1e-12)
    metadata = baseline["metadata"]
    labels = {
        level: {
            "baseline": calculate_label_retrieval_metrics(
                baseline_scores, metadata[f"human_{level}_indices"], metadata[f"robot_{level}_indices"]),
            "transformed": calculate_label_retrieval_metrics(
                changed_scores, metadata[f"human_{level}_indices"], metadata[f"robot_{level}_indices"]),
        }
        for level in ("scene", "task")
    }
    changed_top1 = np.flatnonzero(baseline_order[:, 0] != changed_order[:, 0])
    return {
        "query_count": len(original), "gallery_count": len(gallery),
        "max_abs_embedding_delta": float(np.max(np.abs(original - changed))),
        "mean_embedding_cosine": float(np.mean(cosine)),
        "top1_same_candidate_rate": float(np.mean(baseline_order[:, 0] == changed_order[:, 0])),
        f"top{k}_overlap_fraction": float(np.mean(overlap)),
        "changed_top1_query_indices": changed_top1.tolist(),
        "retrieval": labels,
    }
