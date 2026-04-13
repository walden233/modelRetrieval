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
