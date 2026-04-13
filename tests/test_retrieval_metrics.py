import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.retrieval.metrics import calculate_ndcg, calculate_retrieval_metrics_grouped


def test_grouped_metrics_include_ndcg():
    similarity_matrix = np.array(
        [
            [0.9, 0.8, 0.1, 0.2],
            [0.85, 0.88, 0.15, 0.1],
            [0.1, 0.2, 0.92, 0.89],
            [0.05, 0.1, 0.88, 0.91],
        ]
    )
    metrics = calculate_retrieval_metrics_grouped(similarity_matrix, group_size=2)
    assert "NDCG" in metrics
    assert metrics["R@1"] == 1.0


def test_ndcg_returns_zero_for_empty_input():
    assert calculate_ndcg([]) == 0.0
