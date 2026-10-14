import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bise.data.rh20t.trajectory_dataset import RH20TTrajectoryDataset
from bise.data.rh20t.trajectory_manifest import build_trajectory_manifest
from bise.data.rh20t.collate import collate_trajectories, trajectory_context, trajectory_context_kwargs
from bise.modalities.trajectory.factory import split_trajectory_dataset, build_trajectory_model
from bise.modalities.trajectory.trainer import train_augmented_trajectory_epoch
from bise.modalities.trajectory.models import CrossModalTrajectoryModel


def write_scene(root, task, scene, missing=False):
    path = root / task / scene
    path.mkdir(parents=True)
    human = [{"frame_index": i, "hands_landmarks": [] if missing or i == 1
              else [np.ones((21, 3), dtype=np.float32) * (i + 1)]} for i in [0, 1, 21]]
    robot = [{"timestamp": 1000 + i, "tcp": np.array([i/1000, 0, 0, 1., 0, 0, 0])}
             for i in [0, 100, 5000]]
    np.save(path / "human_pose.npy", {"cam": human})
    np.save(path / "tcp_base.npy", {"cam": robot})
    return path


def test_invalid_scenes_removed_before_split_and_no_identity_fallback(tmp_path):
    write_scene(tmp_path, "task_0001", "scene_1", missing=True)
    write_scene(tmp_path, "task_0001", "scene_2")
    dataset = RH20TTrajectoryDataset(str(tmp_path), True)
    assert len(dataset) == 1
    assert dataset[0]["scene_id"] == "task_0001/scene_2"
    assert any(e["reason"] == "no valid paired camera" for e in dataset.manifest["exclusions"])
    (tmp_path / "task_0001/scene_2/human_pose.npy").unlink()
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_gap_context_distinguishes_missing_from_padding(tmp_path):
    write_scene(tmp_path, "task_0001", "scene_1")
    dataset = RH20TTrajectoryDataset(str(tmp_path), True, preserve_timing=True,
                                    max_human_frames=256, max_robot_frames=256)
    item = dataset[0]
    assert item["human_frame_indices"][0].tolist() == [0, 1, 21]
    assert item["human_observed"][0].tolist() == [True, False, True]
    batch = collate_trajectories([item])
    assert batch["human_mask"].tolist() == [[True, True, True]]
    context = trajectory_context(batch, "human", "cpu")
    assert context[0, 1, 2] == 0
    assert context[0, 2, 1] == pytest.approx(np.log1p(20))
    assert context[0, 1, 0] == pytest.approx(1/21)
    assert batch["robot_times"].tolist() == [[0., pytest.approx(.1), 5.]]


def test_manifest_cannot_be_reused_for_other_cfg(tmp_path):
    root = tmp_path / "cfg2"
    write_scene(root, "task_0001", "scene_1")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(build_trajectory_manifest(str(root))))
    with pytest.raises(ValueError, match="different data root"):
        RH20TTrajectoryDataset(str(tmp_path / "cfg3"), manifest_path=str(path))


def test_time_aware_forward_backward_and_missing_coordinates_ignored(tmp_path):
    torch.set_num_threads(2)
    write_scene(tmp_path, "task_0001", "scene_1")
    write_scene(tmp_path, "task_0002", "scene_1")
    dataset = RH20TTrajectoryDataset(str(tmp_path), True, preserve_timing=True,
                                    max_human_frames=256, max_robot_frames=256)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_trajectories)
    model = CrossModalTrajectoryModel(18, 9, 16, 4, 1, 32, 8, dropout=0,
                                      robot_representation="relative_rot6d", temporal_features=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    loss = train_augmented_trajectory_epoch(model, loader, optimizer, "cpu", .3, True)
    assert np.isfinite(loss).all()
    batch = next(iter(loader))
    model.eval()
    context = trajectory_context_kwargs(batch, "cpu")
    with torch.no_grad():
        a = model(batch["human_poses"], batch["human_mask"], batch["tcp_bases"], batch["tcp_mask"], **context)[0]
        batch["human_poses"][:, 1] = 999
        b = model(batch["human_poses"], batch["human_mask"], batch["tcp_bases"], batch["tcp_mask"], **context)[0]
    torch.testing.assert_close(a, b)
    assert model.human_encoder.time_proj.weight.grad is not None


def test_tool_mapping_requires_explicit_target_configuration():
    config = {"data_root": "/tmp/cfg3", "model_params": {},
              "reference_frames": {"cfg2": {"source": "calibration", "tcp_to_reference": np.eye(4).tolist()}}}
    with pytest.raises(ValueError, match="cfg3"):
        build_trajectory_model(config, "cpu")


def test_manifest_records_missing_source_file(tmp_path):
    path = write_scene(tmp_path, "task_0001", "scene_1")
    (path / "tcp_base.npy").unlink()
    manifest = build_trajectory_manifest(str(tmp_path))
    assert not manifest["scenes"]
    assert manifest["exclusions"][0]["scene_id"] == "task_0001/scene_1"


def test_single_sampling_keeps_time_and_observation_alignment(tmp_path):
    write_scene(tmp_path, "task_0001", "scene_1")
    dataset = RH20TTrajectoryDataset(str(tmp_path), True, preserve_timing=True,
                                    max_human_frames=2, max_robot_frames=2)
    item = dataset[0]
    batch = collate_trajectories([item])
    assert batch["human_frame_indices"].tolist() == [[0, 21]]
    assert batch["robot_times"].tolist() == [[0., 5.]]
    assert batch["tcp_bases"].shape[1] == 2
    assert batch["human_observed"].tolist() == [[True, True]]


def test_checkpoint_rejects_same_dimension_different_geometry(tmp_path):
    from bise.modalities.trajectory.provenance import validate_checkpoint_config
    saved = {"model_params": {"robot_input_dim": 9, "robot_representation": "rot6d"}}
    (tmp_path / "params.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="model_params"):
        validate_checkpoint_config({"model_params": {"robot_input_dim": 9,
                                    "robot_representation": "relative_rot6d"}},
                                   str(tmp_path / "best_model.pth"))


def test_evaluation_reuses_training_manifest_snapshot(tmp_path):
    from bise.modalities.trajectory.factory import build_trajectory_dataset
    root = tmp_path / "data"
    write_scene(root, "task_0001", "scene_1")
    run = tmp_path / "run"
    run.mkdir()
    (run / "data_manifest.json").write_text(json.dumps(build_trajectory_manifest(str(root))))
    # Newly added scenes must not silently enter an existing checkpoint's data inventory.
    write_scene(root, "task_0002", "scene_1")
    dataset = build_trajectory_dataset({"data_root": str(root), "model_params": {}},
                                       str(run / "best_model.pth"))
    assert len(dataset) == 1
    assert dataset[0]["scene_id"] == "task_0001/scene_1"
