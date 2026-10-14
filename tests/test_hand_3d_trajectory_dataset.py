import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bise.data.rh20t.collate import collate_trajectories, trajectory_context
from bise.data.rh20t.hand_3d_trajectory_dataset import (
    RH20THand3DTrajectoryDataset, build_hand_3d_trajectory_manifest,
)
from bise.data.rh20t.human_representation import hand_xyz_features
from bise.modalities.trajectory.factory import build_trajectory_dataset
from bise.modalities.trajectory.human_robustness import fixed_gallery_comparison


def _write_scene(root):
    scene = root / "task_0001" / "scene_1"
    scene.mkdir(parents=True)
    n = 6
    points = np.full((n, 2, 21, 3), np.nan, dtype=np.float32)
    valid = np.zeros((n, 2, 21), dtype=bool)
    tracks = np.full((n, 2), -1, dtype=np.int8)
    for frame, slot, track in [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1),
                               (2, 0, 1), (3, 0, 0), (4, 0, 1), (5, 0, 1)]:
        tracks[frame, slot] = track
        valid[frame, slot] = True
        points[frame, slot] = frame + 10 * track
    suspect = np.zeros((n, 2), dtype=bool)
    suspect[3, 0] = True
    palm_poses = np.full((n, 2, 4, 4), np.nan, dtype=np.float32)
    for frame, slot in np.argwhere(tracks >= 0):
        palm_poses[frame, slot] = np.eye(4, dtype=np.float32)
        palm_poses[frame, slot, :3, 3] = points[frame, slot, 0]
    camera = {"frame_index": np.arange(n, dtype=np.int32),
              "timestamp_ms": np.array([1000, 1100, 1200, 2000, 2100, 2200]),
              "hand_count": np.array([2, 2, 1, 1, 1, 1]),
              "hand_track_id": tracks,
              "hand_track_ambiguous": np.zeros((n, 2), dtype=bool),
              "wrist_middle_span_suspect": suspect,
              "landmarks_base_m": points, "landmarks_base_valid": valid,
              "palm_pose_base": palm_poses,
              "T_base_from_camera": np.eye(4), "extrinsics_source": "fixture"}
    np.save(scene / "human_hand_3d.npy", {"schema_version": 2, "landmark_count": 21,
                                         "max_hands": 2, "cameras": {"cam": camera}})
    robot = [{"timestamp": t, "tcp": np.array([i, 0, 0, 1, 0, 0, 0])}
             for i, t in enumerate((5000, 5100, 6200))]
    np.save(scene / "tcp_base.npy", {"cam": robot})
    return scene


def test_track_identity_missing_frame_and_real_time(tmp_path):
    _write_scene(tmp_path)
    dataset = RH20THand3DTrajectoryDataset(str(tmp_path), max_human_frames=6,
                                           max_robot_frames=3)
    entry = dataset.manifest["scenes"][0]["cameras"]["cam"]
    assert entry["selected_track_id"] == 1
    assert entry["observed_frames"] == 5
    item = dataset[0]
    assert item["human_observed"][0].tolist() == [True, True, True, False, True, True]
    assert item["human_frame_indices"][0].tolist() == list(range(6))
    assert item["human_times"][0].tolist() == pytest.approx([0, .1, .2, 1., 1.1, 1.2])
    np.testing.assert_allclose(item["human_poses"][0][0].numpy(), 10)
    np.testing.assert_allclose(item["human_poses"][0][1].numpy(), 11)
    np.testing.assert_allclose(item["human_poses"][0][3].numpy(), 0)
    batch = collate_trajectories([item])
    assert trajectory_context(batch, "human", "cpu")[0, 3, 1].item() == pytest.approx(np.log1p(.8))


def test_manifest_without_source_hashes_keeps_sampling_and_reads_current_data(tmp_path):
    scene = _write_scene(tmp_path)
    manifest = build_hand_3d_trajectory_manifest(str(tmp_path))
    assert not {"human_sha256", "robot_sha256"} & manifest["scenes"][0].keys()
    manifest["scenes"][0].update(human_sha256="obsolete", robot_sha256="obsolete")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    config = {"data_root": str(tmp_path), "use_6_keypoints": True,
              "data_params": {"human_source": "hand_3d_base", "preserve_timing": True,
                              "max_human_frames": 4, "max_robot_frames": 2},
              "model_params": {"temporal_features": True}}
    dataset = build_trajectory_dataset(config)
    item = dataset[0]
    assert item["human_poses"][0].shape == (4, 6, 3)
    assert item["human_frame_indices"][0][[0, -1]].tolist() == [0, 5]
    assert item["human_observed"][0].sum() >= 2
    (tmp_path / "old.json").write_text(json.dumps({**manifest, "human_source": "human_pose"}))
    with pytest.raises(ValueError, match="Unsupported hand_3d_base"):
        RH20THand3DTrajectoryDataset(str(tmp_path), manifest_path=str(tmp_path / "old.json"))

    data = np.load(scene / "human_hand_3d.npy", allow_pickle=True).item()
    data["cameras"]["cam"]["timestamp_ms"][1] += 1
    np.save(scene / "human_hand_3d.npy", data)
    frozen = RH20THand3DTrajectoryDataset(str(tmp_path), manifest_path=str(path))
    assert not {"human_sha256", "robot_sha256"} & frozen.manifest["scenes"][0].keys()
    assert frozen[0]["human_times"][0][1].item() == pytest.approx(.101)


def test_invalid_scene_excluded_without_sample_substitution(tmp_path):
    scene = _write_scene(tmp_path)
    data = np.load(scene / "human_hand_3d.npy", allow_pickle=True).item()
    data["cameras"]["cam"]["timestamp_ms"][2] = 900
    np.save(scene / "human_hand_3d.npy", data)
    manifest = build_hand_3d_trajectory_manifest(str(tmp_path))
    assert not manifest["scenes"]
    assert any("timestamps must increase" in x["reason"] for x in manifest["exclusions"])


def test_n1_n2_coordinate_invariance_and_missing_frames():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(5, 21, 3)).astype(np.float32)
    observed = np.array([False, True, True, False, True])
    points[~observed] = np.nan
    angle = np.deg2rad(27)
    palm = np.eye(4)
    palm[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    palm[:3, 3] = [0.2, -0.1, 0.3]
    changed = np.eye(4)
    changed[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    changed[:3, 3] = [0.4, -0.3, 0.2]
    translation = np.eye(4)
    translation[:3, 3] = changed[:3, 3]

    n0 = hand_xyz_features(points, observed, "base_xyz")
    n0_shifted = hand_xyz_features(points, observed, "base_xyz", global_transform=translation)
    assert not np.allclose(n0, n0_shifted)
    n1 = hand_xyz_features(points, observed, "wrist_relative")
    np.testing.assert_allclose(n1, hand_xyz_features(points, observed, "wrist_relative",
                                                   global_transform=translation), atol=1e-6)
    assert not np.allclose(n1, hand_xyz_features(points, observed, "wrist_relative",
                                                  global_transform=changed))
    n2 = hand_xyz_features(points, observed, "palm_relative", palm)
    np.testing.assert_allclose(n2, hand_xyz_features(points, observed, "palm_relative",
                                                   palm, changed), atol=1e-6)
    np.testing.assert_array_equal(n2[~observed], 0)


def test_n2_dataset_transform_keeps_robot_gallery_fixed(tmp_path):
    _write_scene(tmp_path)
    dataset = RH20THand3DTrajectoryDataset(str(tmp_path), human_representation="palm_relative")
    changed = np.eye(4)
    changed[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    changed[:3, 3] = [0.1, 0.2, 0.3]
    original, transformed = dataset[0], dataset.with_human_transform(changed)[0]
    np.testing.assert_allclose(original["human_poses"][0], transformed["human_poses"][0], atol=1e-6)
    np.testing.assert_array_equal(original["tcp_bases"][0], transformed["tcp_bases"][0])
    assert original["human_observed"][0].tolist() == transformed["human_observed"][0].tolist()


def test_fixed_gallery_comparison_reports_ranking_stability():
    baseline = {"human_embeddings": np.array([[1., 0.], [0., 1.]]),
                "robot_embeddings": np.array([[1., 0.], [0., 1.]]),
                "metadata": {"human_scene_indices": [0, 1], "robot_scene_indices": [0, 1],
                             "human_task_indices": [0, 1], "robot_task_indices": [0, 1]}}
    identical = fixed_gallery_comparison(baseline, baseline["human_embeddings"])
    assert identical["top1_same_candidate_rate"] == 1.0
    assert identical["max_abs_embedding_delta"] == 0.0
    changed = fixed_gallery_comparison(baseline, baseline["human_embeddings"][::-1])
    assert changed["top1_same_candidate_rate"] == 0.0
    assert changed["retrieval"]["scene"]["transformed"]["R@1"] == 0.0
