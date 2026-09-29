import numpy as np
import pytest

from bise.data.rh20t.hand_3d_extractor import (
    camera_matrix_from_calibration,
    decode_rh20t_depth_frame,
    nearest_timestamp_indices,
    palm_pose_from_landmarks,
    project_pixel_to_camera,
    sample_depth_median,
    track_hands_by_wrist,
    validate_rigid_transform,
    wrist_middle_span_quality,
)


def test_decode_rh20t_packed_depth_uses_low_and_high_byte_planes():
    low = np.asarray([[1, 2], [255, 0]], dtype=np.uint8)
    high = np.asarray([[0, 1], [2, 255]], dtype=np.uint8)
    gray = np.vstack((low, high))
    encoded_bgr = np.repeat(gray[:, :, None], 3, axis=2)

    decoded = decode_rh20t_depth_frame(encoded_bgr)

    np.testing.assert_array_equal(decoded, [[1, 258], [767, 65280]])


def test_intrinsics_scale_to_video_and_backproject_metric_point():
    calibration = np.asarray(
        [[1000, 0, 640, 0], [0, 900, 360, 0], [0, 0, 1, 0]], dtype=np.float64
    )
    intrinsics = camera_matrix_from_calibration(calibration, 640, 360)

    np.testing.assert_allclose(intrinsics, [[500, 0, 320], [0, 450, 180], [0, 0, 1]])
    np.testing.assert_allclose(
        project_pixel_to_camera(420, 80, 0.5, intrinsics), [0.1, -0.1111111, 0.5], atol=2e-8
    )


def test_depth_sampling_uses_local_median_and_rejects_sparse_patch():
    depth = np.zeros((7, 7), dtype=np.float32)
    depth[2:5, 2:5] = 0.8
    depth[3, 3] = 3.0

    median, count, mad = sample_depth_median(depth, 3, 3, radius=2)
    assert median == pytest.approx(0.8)
    assert count >= 3
    assert mad == pytest.approx(0.0)

    invalid, invalid_count, _ = sample_depth_median(np.zeros((3, 3), dtype=np.float32), 1, 1)
    assert np.isnan(invalid)
    assert invalid_count == 0


def test_nearest_timestamp_matching_preserves_missing_pairs():
    indices, deltas, valid = nearest_timestamp_indices(
        np.asarray([100, 180, 1000]), np.asarray([100, 180, 260])
    )
    np.testing.assert_array_equal(indices, [0, 1, -1])
    np.testing.assert_array_equal(deltas, [0, 0, -740])
    np.testing.assert_array_equal(valid, [True, True, False])


def test_rigid_transform_validation_rejects_reflection():
    transform = np.eye(4)
    np.testing.assert_array_equal(validate_rigid_transform(transform), transform.astype(np.float32))
    transform[0, 0] = -1
    with pytest.raises(ValueError, match="proper rotation"):
        validate_rigid_transform(transform)


def test_palm_pose_is_a_right_handed_frame_from_depth_landmarks():
    landmarks = np.full((21, 3), np.nan, dtype=np.float32)
    valid = np.zeros(21, dtype=bool)
    landmarks[0] = [0, 0, 0]
    landmarks[5] = [1, 0, 0]
    landmarks[9] = [0, 1, 0]
    landmarks[17] = [-1, 0, 0]
    valid[[0, 5, 9, 17]] = True

    pose = palm_pose_from_landmarks(landmarks, valid)

    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(pose[3], [0, 0, 0, 1])


def test_hand_tracks_keep_right_only_detection_on_its_own_id():
    count = np.asarray([1, 2, 2, 1], dtype=np.uint8)
    labels = np.asarray([["Right", ""], ["Left", "Right"], ["Right", "Right"], ["Right", ""]])
    points = np.full((4, 2, 21, 2), np.nan, dtype=np.float32)
    points[0, 0, 0] = [300, 100]
    points[1, 0, 0], points[1, 1, 0] = [100, 100], [302, 100]
    points[2, 0, 0], points[2, 1, 0] = [102, 100], [304, 100]
    points[3, 0, 0] = [306, 100]

    ids, ambiguous = track_hands_by_wrist(count, labels, points, np.asarray([640, 360]))

    np.testing.assert_array_equal(ids, [[1, -1], [0, 1], [0, 1], [1, -1]])
    assert ambiguous[2].all()
    assert not ambiguous[0, 0]


def test_hand_track_uses_nearby_wrist_when_handedness_flickers():
    count = np.asarray([1, 1], dtype=np.uint8)
    labels = np.asarray([["Left", ""], ["Right", ""]])
    points = np.full((2, 2, 21, 2), np.nan, dtype=np.float32)
    points[0, 0, 0] = [100, 100]
    points[1, 0, 0] = [101, 100]

    ids, ambiguous = track_hands_by_wrist(count, labels, points, np.asarray([640, 360]))

    np.testing.assert_array_equal(ids[:, 0], [0, 0])
    assert ambiguous[1, 0]


def test_span_quality_flags_large_depth_disagreement_without_filling_missing_hands():
    camera = np.full((2, 2, 21, 3), np.nan, dtype=np.float32)
    local = np.full_like(camera, np.nan)
    camera[:, 0, 0] = [0, 0, 0]
    camera[0, 0, 12], camera[1, 0, 12] = [0.1, 0, 0], [0.3, 0, 0]
    local[:, 0, 0] = [0, 0, 0]
    local[:, 0, 12] = [0.1, 0, 0]

    ratio, suspect = wrist_middle_span_quality(camera, local)

    np.testing.assert_allclose(ratio[:, 0], [1.0, 3.0])
    assert np.isnan(ratio[:, 1]).all()
    np.testing.assert_array_equal(suspect[:, 0], [False, True])
    assert not suspect[:, 1].any()
