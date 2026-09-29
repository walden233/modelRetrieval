import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from add_hand_base_coordinates import (
    aligned_base_from_camera, apply_transform, pose_matrix, reconcile_calibration_tcp_height,
)


def test_aligned_base_from_camera_inverts_official_cfg3_chain():
    in_hand = np.eye(4)
    in_hand[:3, 3] = [0.1, -0.2, 0.3]
    camera = np.eye(4)
    camera[:3, 3] = [-0.3, 0.4, 0.1]
    tcp = np.array([0.5, 0.2, 0.4, 1, 0, 0, 0], dtype=float)

    transform = aligned_base_from_camera(camera, in_hand, tcp)

    from add_hand_base_coordinates import ALIGN_BASE, TCP_CAMERA
    official_base_to_camera = camera @ np.linalg.inv(in_hand) @ TCP_CAMERA @ np.linalg.inv(pose_matrix(tcp)) @ ALIGN_BASE
    np.testing.assert_allclose(transform @ official_base_to_camera, np.eye(4), atol=1e-6)


def test_aligned_base_from_camera_uses_cfg2_rotated_tcp_camera():
    from add_hand_base_coordinates import CFG2_ALIGN_BASE, CFG2_TCP_CAMERA
    in_hand = np.eye(4)
    camera = np.eye(4)
    camera[:3, 3] = [0.2, -0.1, 0.3]
    tcp = np.array([0.5, 0.0, 0.17, 0, 0, 1, 0], dtype=float)

    transform = aligned_base_from_camera(camera, in_hand, tcp, cfg=2)

    official_base_to_camera = camera @ np.linalg.inv(in_hand) @ CFG2_TCP_CAMERA @ np.linalg.inv(pose_matrix(tcp)) @ CFG2_ALIGN_BASE
    np.testing.assert_allclose(transform @ official_base_to_camera, np.eye(4), atol=1e-6)


def test_apply_transform_keeps_missing_points_and_poses_invalid(tmp_path):
    points = np.full((1, 2, 21, 3), np.nan, dtype=np.float32)
    points[0, 0, 0] = [1, 2, 3]
    valid = np.zeros((1, 2, 21), dtype=bool)
    valid[0, 0, 0] = True
    poses = np.full((1, 2, 4, 4), np.nan, dtype=np.float32)
    poses[0, 0] = np.eye(4)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    camera = {"landmarks_camera_m": points, "landmarks_camera_valid": valid, "palm_pose_camera": poses}

    count = apply_transform(camera, transform, "test", tmp_path / "T_base_from_camera.json")

    assert count == 1
    np.testing.assert_allclose(camera["landmarks_base_m"][0, 0, 0], [1.1, 1.8, 3.3])
    assert np.isnan(camera["landmarks_base_m"][0, 1]).all()
    np.testing.assert_allclose(camera["palm_pose_base"][0, 0], transform)
    assert np.isnan(camera["palm_pose_base"][0, 1]).all()


def test_reconcile_gross_calibration_tcp_height_mismatch():
    calibration = np.array([0, 0, 0.45, 1, 0, 0, 0])

    corrected, delta = reconcile_calibration_tcp_height(calibration, 0.2)

    assert corrected[2] == 0.2
    assert delta == 0.25
    assert calibration[2] == 0.45
