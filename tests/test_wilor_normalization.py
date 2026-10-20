import json
from pathlib import Path

import numpy as np

from bise.data.rh20t.wilor_compare import candidate_metrics
from bise.data.rh20t.wilor_normalization import add_candidates


def scene():
    settings = json.loads(Path('configs/hand/wilor.json').read_text())['comparison']
    shape = np.stack([np.linspace(0, .03, 21), np.linspace(0, .08, 21), np.zeros(21)], axis=1)
    shape[0], shape[5], shape[9], shape[17] = [0, 0, 0], [.04, .03, 0], [0, .08, 0], [-.04, .03, 0]
    points = np.tile(shape, (2, 10, 1, 1))
    points[..., 2] += 1
    points[:, :, :, 0] += np.arange(10)[None, :, None]*.002
    points[1, ..., 0] += .2
    valid = np.ones(points.shape[:-1], bool)
    times = 1000+np.arange(10)*40
    fused = {'hands': np.array(['A', 'B']), 'main_hand': np.asarray('A'),
             'points_base_m': points.copy(), 'valid': valid,
             'method': np.full(valid.shape, 2, np.uint8), 'timestamp_ms': times}
    cameras = {str(i): {'hands': np.array(['A', 'B']), 'points_base_m': points.copy(),
                       'valid_frame': valid.all(axis=-1).copy(), 'timestamp_ms': times}
               for i in range(2)}
    return fused, cameras, settings


def test_shared_frame_preserves_bimanual_geometry_and_is_identical_across_cameras():
    fused, cameras, settings = scene()
    before = fused['points_base_m'].copy()
    fused['valid'][1, 3] = False
    fused['points_base_m'][1, 3] = np.nan
    report = add_candidates(fused, cameras, settings)
    assert report['shared_anchor_source'] == 'multiview'
    for camera in cameras.values():
        np.testing.assert_allclose(camera['n1_shared_pose_in_base'], fused['n1_shared_pose_in_base'])
    np.testing.assert_allclose(fused['points_n1_shared'][1, 0]-fused['points_n1_shared'][0, 0],
                               before[1, 0]-before[0, 0], atol=1e-7)
    assert np.isnan(fused['points_n1_shared'][1, 3]).all()
    assert not fused['n1_shared_valid'][1, 3].any()
    np.testing.assert_array_equal(cameras['0']['points_base_m'], before)


def test_shared_candidate_does_not_hide_a_camera_translation_error_or_rescale_motion():
    fused, cameras, settings = scene()
    cameras['1']['points_base_m'][..., 0] += .3
    add_candidates(fused, cameras, settings)
    rows = np.stack([np.arange(10), np.arange(10)], axis=1)
    metrics = candidate_metrics(cameras['0'], cameras['1'], rows, 0)
    assert metrics['candidate_paired_frames'] == 10
    assert abs(metrics['shared_candidate_joint_rmse_m']-.3) < 1e-6
    assert metrics['view_candidate_joint_rmse_m'] < 1e-6
    normalized_step = np.linalg.norm(np.diff(fused['points_n1_shared'][0, :, 0], axis=0), axis=-1)
    np.testing.assert_allclose(normalized_step, .002, atol=1e-7)


def test_anchor_preference_fallback_and_unavailable_status_are_explicit():
    fused, cameras, settings = scene()
    fused['method'][:, :5] = 1
    report = add_candidates(fused, cameras, settings)
    assert report['shared_anchor_timestamp_ms'] == 1240
    fused, cameras, settings = scene()
    fused['method'][:] = 1
    report = add_candidates(fused, cameras, settings)
    assert report['shared_anchor_source'] == 'single_view'
    fused, cameras, settings = scene()
    fused['main_hand'] = np.asarray('')
    report = add_candidates(fused, cameras, settings)
    assert report['shared_status'] == 'no_stable_main_hand_anchor'
    assert np.isnan(fused['points_n1_shared']).all()
    assert cameras['0']['n1_view_valid'].all()  # The independent candidate remains available.


def test_view_candidate_does_not_silently_use_a_different_action_time():
    fused, cameras, settings = scene()
    fused['method'][:, :5] = 1
    cameras['1']['valid_frame'][:, :8] = False
    report = add_candidates(fused, cameras, settings)
    assert report['shared_anchor_timestamp_ms'] == 1240
    assert (cameras['1']['n1_view_anchor_timestamp_ms'] == -1).all()
    assert not cameras['1']['n1_view_valid'].any()
    assert cameras['1']['n1_shared_valid'][:, 8:].all()
