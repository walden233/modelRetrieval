import copy
import json
from pathlib import Path

import numpy as np

from bise.data.rh20t.wilor_identity import HAND_FIELDS, correct_identities, finalize_camera


def settings():
    return json.loads(Path('configs/hand/wilor.json').read_text())['trajectory']['identity']


def scene(count=3):
    cameras = {}
    n = 15
    points = np.zeros((2, n, 21, 3))
    points[..., 2] = 1
    points[0, ..., 0], points[1, ..., 0] = -.15, .15
    points[0, ..., 1], points[1, ..., 1] = .05, -.05
    points[..., 0] += np.arange(n)[None, :, None]*.002
    points[..., 1] += np.arange(21)[None, None, :]*.001
    k = np.array([[500., 0, 320], [0, 500, 180], [0, 0, 1]])
    for i in range(count):
        transform = np.eye(4)
        transform[0, 3] = -.2+i*.2
        local = points-transform[:3, 3]
        uv = local @ k.T
        uv = uv[..., :2]/uv[..., 2:]
        cameras[str(i)] = {
            'hands': np.array(['A', 'B']), 'roles': np.array(['main', 'support']),
            'timestamp_ms': 1000+np.arange(n)*40, 'frame_index': np.arange(n),
            'observation_index': np.arange(2*n).reshape(2, n),
            'role_confidence': np.ones((2, n)), 'valid_frame': np.ones((2, n), bool),
            'points_base_m': points.copy(), 'joints_image_px': uv,
            'joints_camera_m': local.copy(), 'joints_root_m': local-local[:, :, :1],
            'depth_anchors': np.full((2, n), 10), 'depth_mad_m': np.zeros((2, n)),
            'intrinsics': k, 'T_base_from_camera': transform}
    annotation = {'cameras': [{'camera': name, 'segments': [
        {'hand_id': 'A'}, {'hand_id': 'B'}]} for name in cameras]}
    return cameras, annotation


def swap(camera):
    for field in HAND_FIELDS:
        camera[field] = camera[field][::-1].copy()


def test_single_bad_view_is_corrected_against_two_references_without_changing_raw_labels():
    cameras, annotation = scene()
    expected = cameras['1']['points_base_m'].copy()
    swap(cameras['1'])
    original = copy.deepcopy(annotation)
    corrected, resolved, report = correct_identities(cameras, annotation, settings())
    assert report['status'] == 'corrected'
    assert report['swapped_cameras'] == ['1']
    assert report['after_score'] < report['before_score']*.5
    np.testing.assert_allclose(corrected['1']['points_base_m'], expected)
    assert list(corrected['1']['hands']) == ['A', 'B']
    assert resolved['cameras'][1]['segments'][0]['hand_id'] == 'B'
    assert annotation == original
    np.testing.assert_allclose(cameras['1']['points_base_m'], expected[::-1])


def test_two_views_need_a_semantic_role_anchor_not_just_geometric_correspondence():
    cameras, annotation = scene(2)
    swap(cameras['1'])
    _, _, report = correct_identities(cameras, annotation, settings())
    assert report['status'] == 'ambiguous_two_view_role_anchor'
    cameras['1']['role_confidence'][:] = .3
    _, _, report = correct_identities(cameras, annotation, settings())
    assert report['status'] == 'corrected' and report['swapped_cameras'] == ['1']


def test_low_confidence_can_only_be_recovered_at_individually_verified_times():
    cameras, annotation = scene()
    swap(cameras['1'])
    cameras['1']['role_confidence'][:] = .3
    for name in ('0', '2'):
        cameras[name]['valid_frame'][:, -1] = False
    corrected, _, report = correct_identities(cameras, annotation, settings())
    assert report['status'] == 'corrected'
    final = finalize_camera(corrected['1'], .7)
    assert final['valid_frame'][:, :-1].all()
    assert not final['valid_frame'][:, -1].any()
    assert (final['observation_index'][:, -1] == -1).all()
    assert np.isnan(final['points_base_m'][:, -1]).all()
    assert (final['role_confidence'] == .3).all()  # Never rewrite the VLM's original confidence.


def test_calibration_error_or_insufficient_overlap_cannot_be_fixed_by_swapping_labels():
    cameras, annotation = scene()
    swap(cameras['1'])
    cameras['1']['points_base_m'][..., 2] += .5
    _, _, report = correct_identities(cameras, annotation, settings())
    assert not report['swapped_cameras']
    cameras, annotation = scene()
    for data in cameras.values():
        data['valid_frame'][:, 3:] = False
    _, _, report = correct_identities(cameras, annotation, settings())
    assert report['status'] == 'insufficient_joint_overlap_or_calibration'


def test_unchanged_consistent_views_and_single_hand_case():
    cameras, annotation = scene()
    _, _, report = correct_identities(cameras, annotation, settings())
    assert not report['swapped_cameras']
    for data in cameras.values():
        data['hands'] = np.array(['A'])
    _, _, report = correct_identities(cameras, annotation, settings())
    assert report['status'] == 'not_two_physical_hands'
