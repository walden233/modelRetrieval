import json
from pathlib import Path

import numpy as np
import pytest

from bise.modalities.video.views import analyze_views, export_views, scene_max_similarity


def metadata(scenes=4):
    return {
        'sample_ids': [f'task_{i}/scene/a{j}' for i in range(scenes) for j in range(2)],
        'task_ids': [f'task_{i}' for i in range(scenes) for _ in range(2)],
        'scene_ids': ['scene' for _ in range(scenes * 2)],
        'camera_ids': ['a', 'b'] * scenes,
    }


def test_scene_max_pools_all_four_cells_including_off_diagonal_and_respects_task_ids():
    matrix = np.arange(16).reshape(4, 4)
    meta = metadata(2)
    pooled, ids = scene_max_similarity(matrix, meta)
    np.testing.assert_array_equal(pooled, [[5, 7], [13, 15]])
    assert ids['task_ids'] == ['task_0', 'task_1']
    order = [3, 0, 2, 1]
    permuted = {key: [values[i] for i in order] for key, values in meta.items()}
    actual, _ = scene_max_similarity(matrix[np.ix_(order, order)], permuted)
    np.testing.assert_array_equal(actual, pooled)


def test_camera_matched_negatives_and_equal_galleries_separate_similarity_from_ranking():
    vectors = np.zeros((8, 6), dtype=np.float32)
    for i in range(4):
        for camera in range(2):
            vectors[2 * i + camera, camera] = 4
            vectors[2 * i + camera, 2 + i] = 1
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    data = analyze_views(vectors, vectors, metadata())
    assert data['summary']['HR_same']['positive_cosine'] == pytest.approx(1)
    assert data['summary']['HR_cross']['positive_cosine'] == pytest.approx(1 / 17)
    assert data['summary']['HR_same']['negative_cosine'] == pytest.approx(16 / 17)
    assert data['summary']['HR_cross']['negative_cosine'] == pytest.approx(0)
    for row in data['summary'].values():
        assert row['margin'] == pytest.approx(1 / 17)
        assert row['scene_R@1'] == row['scene_MRR'] == 1
        assert row['pairs'] == row['retrieval_queries'] == 8
    assert {row['metrics']['gallery_size'] for row in data['retrieval_blocks']} == {4}
    assert {row['negative_count'] for row in data['observations']} == {3}


def test_collapsed_embeddings_have_high_cosine_but_zero_margin_and_chance_retrieval():
    vectors = np.ones((8, 1))
    data = analyze_views(vectors, vectors, metadata())
    for row in data['summary'].values():
        assert row['positive_cosine'] == row['negative_cosine'] == 1
        assert row['margin'] == 0
        assert row['scene_R@1'] == .25
        assert row['scene_MRR'] == pytest.approx(np.mean([1, .5, 1 / 3, .25]))


def test_singleton_camera_blocks_do_not_inflate_retrieval_and_missing_views_are_reported():
    meta = metadata(2)
    meta['camera_ids'] = ['a', 'b', 'c', 'd']
    data = analyze_views(np.eye(4), np.eye(4), meta)
    for row in data['summary'].values():
        assert row['scene_R@1'] is None and row['scene_MRR'] is None
        assert row['negative_pairs'] == row['retrieval_queries'] == 0
        assert row['margin'] is None
    incomplete = {key: values[:3] for key, values in metadata(2).items()}
    data = analyze_views(np.eye(3), np.eye(3), incomplete)
    assert data['complete_scenes'] == 1 and len(data['excluded_scenes']) == 1


def test_saved_analysis_and_global_redraw_use_only_cached_vectors_and_preserve_original_results(tmp_path):
    from bise.common.plots import index_plots, redraw_plots

    root = tmp_path / 'retrain/batch'
    output = root / 'video/E5/evaluations/cfg2'
    output.mkdir(parents=True)
    vectors = np.repeat(np.eye(4), 2, axis=0)
    for name in ('human_embeddings', 'robot_embeddings'):
        np.save(output / f'{name}.npy', vectors)
    np.save(output / 'similarity_matrix.npy', vectors @ vectors.T)
    (output / 'metadata.json').write_text(json.dumps(metadata()))
    (output / 'metrics.json').write_text('{"original": true}')
    original = {p: p.read_bytes() for p in output.iterdir()}
    for name in ('similarity_heatmap.png', 'task_scene_sorted_similarity_heatmap.png'):
        (output / name).write_bytes(b'legacy plot')
    export_views(output)
    assert all(path.read_bytes() == content for path, content in original.items())
    index = index_plots(root)
    assert len(index) == 3
    assert index['video/E5/evaluations/cfg2/camera_heatmap.png']['options']['max_items'] == 0
    assert index['video/E5/evaluations/cfg2/scene_heatmap.png']['options']['level'] == 'scene'
    assert 'sha256' not in (root / 'plots.json').read_text()
    (output / 'views.json').unlink()
    for path in output.glob('*.png'):
        path.unlink()
    redraw_plots(root)
    assert len(list(output.glob('*.png'))) == 3
    assert (output / 'views.json').exists()
    assert all(path.read_bytes() == content for path, content in original.items())


def test_invalid_vectors_or_duplicate_camera_are_not_silently_used():
    with pytest.raises(ValueError, match='normalized'):
        analyze_views(np.ones((4, 2)), np.ones((4, 2)), metadata(2))
    meta = metadata(2)
    meta['camera_ids'][1] = 'a'
    with pytest.raises(ValueError, match='Duplicate camera'):
        analyze_views(np.eye(4), np.eye(4), meta)
