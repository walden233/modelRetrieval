import json
from pathlib import Path

import numpy as np

from bise.data.rh20t.wilor_compare import common_anchor, pair_metrics, synchronized_rows


def settings():
    return json.loads(Path('configs/hand/wilor.json').read_text())['comparison']


def camera(rotation=None, translation=None, scale=1):
    local = np.stack([np.linspace(-.03, .04, 21), np.linspace(0, .1, 21), np.zeros(21)], axis=1)
    local[0], local[5], local[9], local[17] = [0, 0, 0], [.04, .04, 0], [0, .08, 0], [-.04, .04, 0]
    points = np.tile(local, (20, 1, 1))
    points[:, :, 0] += np.arange(20)[:, None]*.005
    points[:, :, 2] += 1
    points *= scale
    if rotation is not None:
        points = points @ rotation.T
    if translation is not None:
        points += translation
    return {'points_base_m': points[None], 'valid_frame': np.ones((1, 20), bool),
            'timestamp_ms': np.arange(20)*40+1000, 'frame_index': np.arange(20)}


def test_exact_n1_removes_one_global_rotation_translation_but_does_not_rescale():
    theta = .7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    a, b = camera(), camera(rotation, [.4, -.2, .3])
    cameras = {'a': a, 'b': b}
    anchor = common_anchor(cameras, ['a', 'b'], 0, settings())
    rows = synchronized_rows(cameras, ['a', 'b'], 0, 50)
    metrics, _ = pair_metrics(a, b, rows, 0, anchor, ['a', 'b'])
    assert metrics['base_joint_rmse_m'] > .1
    assert metrics['n1_joint_rmse_m'] < 1e-6
    assert metrics['n1_wrist_motion_rmse_m'] < 1e-6
    assert metrics['wrist_displacement_cosine'] > .99999
    b = camera(scale=2)
    cameras['b'] = b
    anchor = common_anchor(cameras, ['a', 'b'], 0, settings())
    metrics, _ = pair_metrics(a, b, rows, 0, anchor, ['a', 'b'])
    assert metrics['n1_joint_rmse_m'] > .01


def test_common_anchor_uses_shared_window_not_separate_first_observations():
    a, b = camera(), camera()
    a['valid_frame'][0, :2] = False
    b['valid_frame'][0, :5] = False
    cameras = {'a': a, 'b': b}
    anchor = common_anchor(cameras, ['a', 'b'], 0, settings())
    assert anchor['indices'] == {'a': 6, 'b': 6}
    rows = synchronized_rows(cameras, ['a', 'b'], 0, 50)
    assert len(set(rows[:, 0])) == len(rows)
    assert len(set(rows[:, 1])) == len(rows)
    b['timestamp_ms'] += 5000
    assert len(synchronized_rows(cameras, ['a', 'b'], 0, 50)) == 0
    assert common_anchor(cameras, ['a', 'b'], 0, settings()) is None


def test_unstable_palm_is_not_forced_to_be_an_anchor_and_static_cosine_is_undefined():
    a, b = camera(), camera()
    for data in (a, b):
        for i in range(20):
            angle = np.radians(80*i)
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            data['points_base_m'][0, i] = data['points_base_m'][0, i] @ rotation.T
    assert common_anchor({'a': a, 'b': b}, ['a', 'b'], 0, settings()) is None
    a, b = camera(), camera()
    a['points_base_m'][:] = a['points_base_m'][:, :1]
    b['points_base_m'][:] = b['points_base_m'][:, :1]
    cameras = {'a': a, 'b': b}
    anchor = common_anchor(cameras, ['a', 'b'], 0, settings())
    rows = synchronized_rows(cameras, ['a', 'b'], 0, 50)
    metrics, _ = pair_metrics(a, b, rows, 0, anchor, ['a', 'b'])
    assert metrics['wrist_displacement_cosine'] is None
