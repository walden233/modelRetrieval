import cv2
import numpy as np
import pytest

from bise.data.rh20t.wilor import HandTracker, extract_video
from bise.data.rh20t.wilor_quality import cross_view_quality, fundamental, track_quality


def test_tracks_follow_geometry_across_detection_order_and_split_long_gaps():
    tracker = HandTracker(max_gap_seconds=.5)
    boxes = np.array([[0, 0, 40, 40], [200, 0, 240, 40]], dtype=float)
    points = np.repeat(np.array([[[20., 20.]], [[220., 20.]]]), 21, axis=1)
    ids, _ = tracker.update(boxes, points, np.array([0, 1]), 1000)
    swapped, _ = tracker.update(boxes[::-1], points[::-1]+1, np.array([1, 0]), 1100)
    assert swapped.tolist() == ids[::-1].tolist()
    reacquired, _ = tracker.update(boxes, points, np.array([0, 1]), 2000)
    assert set(reacquired).isdisjoint(ids)


def test_cross_view_diagnostic_uses_calibration_and_source_timestamps():
    k = np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1.]])
    t1, t2 = np.eye(4), np.eye(4)
    t2[0, 3] = .2
    points = np.stack([np.linspace(-.1, .1, 21), np.linspace(-.05, .05, 21), np.ones(21)], axis=1)
    def project(t):
        xyz = points - t[:3, 3]
        uv = xyz @ k.T
        return uv[:, :2] / uv[:, 2:]
    def data(uv, time):
        return {'timestamp_ms': np.array([time]), 'frame_ptr': np.array([0, 1]),
                'joints_image_px': uv[None]}
    a, b = data(project(t1), 1000), data(project(t2), 1010)
    matrix = fundamental((k, t1), (k, t2))
    assert cross_view_quality(a, b, matrix)['median_epipolar_px'] < 1e-7
    b['joints_image_px'][..., 1] += 12
    assert cross_view_quality(a, b, matrix)['median_epipolar_px'] > 10
    b['timestamp_ms'] += 500
    assert cross_view_quality(a, b, matrix)['hand_observation_pairs'] == 0


@pytest.mark.parametrize('frame_batch_size', [1, 4])
def test_extraction_keeps_empty_frames_and_writes_only_separate_output(tmp_path, frame_batch_size):
    source, output = tmp_path/'source', tmp_path/'output'
    source.mkdir()
    output.mkdir()
    video = source/'cam_a_human.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 60, np.uint8))
    writer.release()
    np.save(source/'cam_a_human_timestamps.npy', {'color': [1000, 1070, 1140, 1210, 1280]})
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    class Backend:
        index = 0
        batches = []
        def predict_many(self, images):
            self.batches.append(len(images))
            return [self.predict(image) for image in images]
        def predict(self, image):
            self.index += 1
            if self.index in (1, 3, 5):
                return []
            return [{'boxes': np.array([10, 10, 40, 40]), 'right': 1, 'scores': .9,
                     'joints_image_px': np.full((21, 2), 20.),
                     'joints_root_m': np.arange(63).reshape(21, 3)*.001,
                     'joints_camera_estimate_m': np.ones((21, 3)), 'model_focal_px': 100.,
                     'mano_shape': np.zeros(10), 'mano_pose': np.repeat(np.eye(3)[None], 15, axis=0),
                     'mano_orientation': np.eye(3)[None]}]
    backend = Backend()
    extract_video(backend, source, 'a', output,
                  {'frame_stride': 1, 'frame_batch_size': frame_batch_size,
                   'batch_size': 4, 'tracking': {}})
    assert backend.batches == ([4, 1] if frame_batch_size == 4 else [1]*5)
    with np.load(output/'a.npz', allow_pickle=False) as d:
        assert d['frame_ptr'].tolist() == [0, 0, 1, 1, 2, 2]
        assert d['timestamp_ms'].tolist() == [1000, 1070, 1140, 1210, 1280]
        assert int(d['frame_batch_size']) == frame_batch_size
        assert int(d['hand_batch_size']) == 4
        quality = track_quality(d)
        assert len(quality) == 1
        assert quality[0]['median_bone_cv'] is None
        assert not quality[0]['sufficient_temporal_support']
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
