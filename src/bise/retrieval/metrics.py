import numpy as np


def calculate_retrieval_metrics(similarity_matrix):
    num_queries = similarity_matrix.shape[0]
    ranks = []

    for index in range(num_queries):
        scores = similarity_matrix[index, :]
        ground_truth_score = scores[index]
        sorted_scores = np.sort(scores)[::-1]
        rank = np.where(sorted_scores == ground_truth_score)[0][0] + 1
        ranks.append(rank)

    ranks = np.array(ranks)
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "Mean Rank": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
    }


def calculate_retrieval_metrics_grouped(similarity_matrix, group_size):
    num_queries = similarity_matrix.shape[0]
    if num_queries % group_size != 0:
        raise ValueError("The total number of samples must be divisible by group_size.")

    ranks = []
    for index in range(num_queries):
        group_start_idx = (index // group_size) * group_size
        positive_indices = set(range(group_start_idx, group_start_idx + group_size))
        scores = similarity_matrix[index, :]
        sorted_candidate_indices = np.argsort(-scores)
        best_rank = -1
        for rank, candidate_idx in enumerate(sorted_candidate_indices):
            if candidate_idx in positive_indices:
                best_rank = rank + 1
                break
        if best_rank != -1:
            ranks.append(best_rank)

    ranks = np.array(ranks)
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "Mean Best Positive Rank": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "NDCG": float(calculate_ndcg(ranks)),
    }


def calculate_ndcg(ranks):
    ranks = np.asarray(ranks)
    if len(ranks) == 0:
        return 0.0
    gains = 1.0 / np.log2(ranks + 1)
    ideal = np.ones_like(ranks)
    ideal_gains = 1.0 / np.log2(ideal + 1)
    return float(np.mean(gains / ideal_gains))


def calculate_label_retrieval_metrics(
    similarity_matrix,
    query_labels,
    gallery_labels,
    k_values=(1, 5, 10),
    ndcg_k: int = 10,
):
    similarity_matrix = np.asarray(similarity_matrix)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    if similarity_matrix.ndim != 2:
        raise ValueError("similarity_matrix must be a 2D array.")
    if similarity_matrix.shape[0] != len(query_labels):
        raise ValueError("The number of query labels must match the number of query rows.")
    if similarity_matrix.shape[1] != len(gallery_labels):
        raise ValueError("The number of gallery labels must match the number of gallery columns.")

    valid_ranks = []
    ndcg_scores = []
    recalls = {k: 0 for k in k_values}
    valid_queries = 0
    max_k = min(max(k_values), similarity_matrix.shape[1]) if similarity_matrix.shape[1] else 0

    for query_index, query_label in enumerate(query_labels):
        positive_mask = gallery_labels == query_label
        positive_count = int(positive_mask.sum())
        if positive_count == 0:
            continue

        valid_queries += 1
        scores = similarity_matrix[query_index]
        ranked_indices = np.argsort(-scores)
        ranked_positive_flags = positive_mask[ranked_indices]
        first_positive_rank = int(np.argmax(ranked_positive_flags)) + 1
        valid_ranks.append(first_positive_rank)

        for k in k_values:
            topk_flags = ranked_positive_flags[: min(k, len(ranked_positive_flags))]
            recalls[k] += int(np.any(topk_flags))

        if max_k > 0:
            topk_flags = ranked_positive_flags[: min(ndcg_k, len(ranked_positive_flags))].astype(float)
            discounts = 1.0 / np.log2(np.arange(2, len(topk_flags) + 2))
            dcg = float(np.sum(topk_flags * discounts))
            ideal_hits = min(positive_count, len(topk_flags))
            ideal_dcg = float(np.sum(discounts[:ideal_hits])) if ideal_hits > 0 else 0.0
            ndcg_scores.append(dcg / ideal_dcg if ideal_dcg > 0 else 0.0)

    if valid_queries == 0:
        return {
            "R@1": 0.0,
            "R@5": 0.0,
            "R@10": 0.0,
            "Mean Rank": float("nan"),
            "MRR": float("nan"),
            "Mean Percentage Rank": float("nan"),
            "NDCG@10": float("nan"),
            "valid_queries": 0,
        }

    ranks = np.asarray(valid_ranks, dtype=float)
    metrics = {
        f"R@{k}": recalls[k] / valid_queries for k in k_values
    }
    metrics["Mean Rank"] = float(np.mean(ranks))
    metrics["MRR"] = float(np.mean(1.0 / ranks))
    metrics["Mean Percentage Rank"] = float(np.mean(ranks / len(gallery_labels)))
    metrics[f"NDCG@{ndcg_k}"] = float(np.mean(ndcg_scores)) if ndcg_scores else float("nan")
    metrics["valid_queries"] = int(valid_queries)
    return metrics
