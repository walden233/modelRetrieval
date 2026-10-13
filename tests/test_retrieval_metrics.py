import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.retrieval.metrics import (
    calculate_label_retrieval_metrics, calculate_ndcg, calculate_retrieval_metrics_grouped,
)


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


def test_mpr_all_uses_every_positive_rank_in_percent():
    scores = -np.arange(100)[None, :]
    gallery_labels = np.zeros(100, dtype=int)
    gallery_labels[[1, 9, 17]] = 1

    metrics = calculate_label_retrieval_metrics(scores, [1], gallery_labels)

    assert metrics["MPR-all"] == pytest.approx(10.0)
    assert metrics["Mean Percentage Rank"] == pytest.approx(.02)
    assert metrics["MRR"] == pytest.approx(.5)


def test_mpr_all_weights_queries_equally_and_skips_missing_positives():
    scores = np.tile([4., 3., 2., 1.], (3, 1))
    metrics = calculate_label_retrieval_metrics(scores, ["a", "b", "missing"],
                                                ["a", "b", "b", "b"])

    # Query a has rank 1 (25%); query b has ranks 2/3/4 (75%).
    assert metrics["MPR-all"] == pytest.approx(50.0)
    assert metrics["valid_queries"] == 2
    assert metrics["gallery_size"] == 4


def test_gallery_size_tracks_direction_in_rectangular_retrieval():
    scores = np.array([[.9, .8, .1, .0], [.1, .0, .8, .9], [.2, .3, .4, .5]])
    query_labels = ["a", "b", "missing"]
    gallery_labels = ["a", "a", "b", "b"]

    forward = calculate_label_retrieval_metrics(scores, query_labels, gallery_labels)
    reverse = calculate_label_retrieval_metrics(scores.T, gallery_labels, query_labels)

    assert forward["gallery_size"] == 4
    assert forward["valid_queries"] == 2
    assert reverse["gallery_size"] == 3
    assert reverse["valid_queries"] == 4
    assert forward["Mean Percentage Rank"] == pytest.approx(forward["Mean Rank"] / 4)
    assert reverse["Mean Percentage Rank"] == pytest.approx(reverse["Mean Rank"] / 3)


def test_mpr_all_matches_best_rank_for_one_positive():
    metrics = calculate_label_retrieval_metrics([[.1, .9, .5]], ["a"], ["a", "b", "c"])

    assert metrics["MPR-all"] == pytest.approx(100.0)
    assert metrics["MPR-all"] == pytest.approx(100.0 * metrics["Mean Percentage Rank"])


@pytest.mark.parametrize("scores, query_labels, gallery_labels", [
    (np.zeros((1, 0)), ["a"], []),
    (np.zeros((0, 1)), [], ["a"]),
    (np.zeros((1, 1)), ["a"], ["b"]),
])
def test_mpr_all_is_nan_without_valid_queries(scores, query_labels, gallery_labels):
    metrics = calculate_label_retrieval_metrics(scores, query_labels, gallery_labels)

    assert np.isnan(metrics["MPR-all"])
    assert metrics["valid_queries"] == 0
    assert metrics["gallery_size"] == np.asarray(scores).shape[1]
