import copy
import json

import numpy as np
import pytest

from bise.data.rh20t.wilor_roles import sample_indices, validate_roles
from bise.data.rh20t.wilor_trajectory import (
    canonicalize, fit_translation, fuse_cameras, select_observations, triangulate_joint,
)


def settings():
    from pathlib import Path
    return json.loads(Path('configs/hand/wilor.json').read_text())['trajectory']


def annotation():
    return {'task': 'cut a vegetable', 'main_hand': 'A',
            'hands': [{'id': 'A', 'role': 'main'}, {'id': 'B', 'role': 'support'},
                      {'id': 'C', 'role': 'idle'}],
            'cameras': [{'camera': 'c', 'segments': [
                {'track_id': i, 'hand_id': hand, 'start_s': 0, 'end_s': .2,
                 'confidence': 'high'} for i, hand in enumerate('ABC')]}],
            'frames': {'context': {'origin_timestamp_ms': 1000}}}


def test_role_selection_retains_support_excludes_idle_and_ambiguous_observations():
    points = np.full((6, 21, 2), 20.)
    data = {'timestamp_ms': np.array([1000, 1100]), 'frame_ptr': np.array([0, 3, 6]),
            'track_id': np.array([0, 1, 2, 0, 1, 2]), 'joints_image_px': points,
            'image_size_wh': np.array([100, 100]), 'scores': np.ones(6),
            'joints_root_m': np.zeros((6, 21, 3)),
            'association_ambiguous': np.array([False, False, False, True, False, False])}
    hands, rows, _ = select_observations(data, 'c', annotation(), settings())
    assert hands == ['A', 'B']
    assert rows.tolist() == [[0, -1], [1, 4]]
    assert len(sample_indices({**data, 'frame_index': np.array([0, 1])}, 1, 2)) <= 2


def test_schema_rejects_identity_collision_and_made_up_tracks():
    valid = annotation()
    context = {'cameras': {'c': {'tracks': [{'track_id': i, 'start_s': 0, 'end_s': .2}
                                          for i in range(3)], 'duration_s': .2}}}
    validate_roles(json.dumps(valid), context)
    invalid = copy.deepcopy(valid)
    invalid['cameras'][0]['segments'][1]['hand_id'] = 'A'
    with pytest.raises(ValueError, match='overlapping'):
        validate_roles(json.dumps(invalid), context)
    invalid = copy.deepcopy(valid)
    invalid['cameras'][0]['segments'][0]['track_id'] = 99
    with pytest.raises(ValueError, match='Unknown track'):
        validate_roles(json.dumps(invalid), context)


def test_depth_anchor_rejects_background_outliers_and_insufficient_support():
    local = np.random.default_rng(42).normal(0, .025, (21, 3))
    measured = local + [0.1, .2, .9]
    measured[10:16] += [0, 0, .6]
    translation, good = fit_translation(local, measured, np.ones(21, bool), settings())
    np.testing.assert_allclose(translation, [.1, .2, .9])
    assert not good[10:16].any()
    translation, _ = fit_translation(local, measured, np.arange(21) < 3, settings())
    assert translation is None


def test_triangulation_rejects_bad_time_geometry_and_preserves_metric_scale():
    k = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1]])
    a, b = np.eye(4), np.eye(4)
    b[0, 3] = .2
    xyz = np.array([.1, .05, 1.])
    def observation(t):
        camera = xyz-t[:3, 3]
        uv = k @ camera
        return uv[:2]/uv[2], k, t
    point, error = triangulate_joint([observation(a), observation(b)], settings())
    np.testing.assert_allclose(point, xyz, atol=1e-6)
    assert error < 1e-6
    bad = observation(b)
    bad = (bad[0]+[0, 30], *bad[1:])
    assert triangulate_joint([observation(a), bad], settings())[0] is None
    assert triangulate_joint([observation(a), observation(a)], settings())[0] is None


def test_fusion_does_not_mix_unverified_cameras_or_interpolate_missing_frames():
    base = np.zeros((1, 3, 21, 3), np.float32)
    base[..., 2] = 1
    for j in range(21):
        base[:, :, j, :2] = [.01*(j % 5), .01*(j//5)]
    base[:, :, 5, :2] = [.05, .01]
    base[:, :, 17, :2] = [-.05, .01]
    base[:, :, 9, :2] = [0, .08]
    k = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1]])
    def camera(offset):
        transform = np.eye(4)
        transform[0, 3] = offset
        shifted = base.copy()
        shifted[..., 0] += offset
        return {'hands': np.array(['A']), 'timestamp_ms': np.array([1000, 1100, 1200]),
                'valid_frame': np.array([[True, False, True]]), 'points_base_m': shifted,
                'joints_image_px': np.zeros((1, 3, 21, 2)), 'intrinsics': k,
                'T_base_from_camera': transform}
    other = camera(2)
    other['valid_frame'][:] = [[False, True, False]]
    result = fuse_cameras({'a': camera(0), 'b': other}, annotation(), settings())
    assert not (result['method'] == 2).any()
    assert not result['valid'][:, 1].any()
    assert np.isnan(result['points_base_m'][:, 1]).all()
    # A missing joint remains missing after initial-palm normalization.
    point = result['points_base_m'].copy()
    point[0, 2, 3] = np.nan
    normalized, _, _ = canonicalize(point, np.isfinite(point).all(axis=-1))
    assert np.isnan(normalized[0, 2, 3]).all()


def test_complete_fusion_path_recovers_consistent_views_and_rejects_time_mismatch():
    k = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1]])
    xyz = np.stack([np.arange(21)*.006, np.sin(np.arange(21))*.012, np.ones(21)], axis=1)
    xyz[5, :2], xyz[9, :2], xyz[17, :2] = [.05, .02], [0, .08], [-.05, .02]
    def camera(offset, dt=0):
        transform = np.eye(4)
        transform[0, 3] = offset
        pixel = (xyz-transform[:3, 3]) @ k.T
        pixel = pixel[:, :2]/pixel[:, 2:]
        return {'hands': np.array(['A']), 'timestamp_ms': np.arange(6)*100+1000+dt,
                'valid_frame': np.ones((1, 6), bool),
                'points_base_m': np.tile(xyz, (1, 6, 1, 1)),
                'joints_image_px': np.tile(pixel, (1, 6, 1, 1)),
                'intrinsics': k, 'T_base_from_camera': transform}
    result = fuse_cameras({'a': camera(0), 'b': camera(.2)}, annotation(), settings())
    assert (result['method'] == 2).all()
    np.testing.assert_allclose(result['points_base_m'][0, 0], xyz, atol=1e-6)
    result = fuse_cameras({'a': camera(0), 'b': camera(.2, dt=5000)}, annotation(), settings())
    assert not (result['method'] == 2).any()
