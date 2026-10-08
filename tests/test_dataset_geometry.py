import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runs.trajectory.prepare_dataset_geometry import prepare_dataset_geometry


@pytest.fixture
def inputs(tmp_path):
    dataset = tmp_path / "dataset"
    raw = tmp_path / "raw"
    root = dataset / "RH20T_cfg2"
    scene = root / "task_0001/scene_1"
    scene.mkdir(parents=True)
    original = "task_0001_user_0002_scene_0001_cfg_0002"
    (scene / "T_base_from_camera.json").write_text(json.dumps({"raw_robot_scene": original}))
    tcp = [{"timestamp": 1000, "tcp": [1., 2., 3., 1., 0., 0., 0.]},
           {"timestamp": 1100, "tcp": [2., 3., 4., 1., 0., 0., 0.]}]
    np.save(scene / "tcp_base.npy", {"cam": tcp, "unused": []})
    feedback = raw / "RH20T_cfg2" / original / "transformed/gripper.npy"
    feedback.parent.mkdir(parents=True)
    (feedback.parent / "tcp_base.npy").write_bytes((scene / "tcp_base.npy").read_bytes())
    np.save(feedback, {"cam": {1000: {"gripper_info": [20], "gripper_command": [95]},
                               1100: {"gripper_info": [40], "gripper_command": [95]}}})
    profile = {"gripper": "test gripper", "max_width_m": .095, "width_scale_to_m": .001,
               "reference": "aligned_tcp_proxy", "tcp_to_grasp": np.eye(4).tolist(),
               "source": "test parameters"}
    parameters = tmp_path / "parameters.json"
    # Legacy scene paths and cached arrays must never be used to generate the new cache.
    parameters.write_text(json.dumps({"profiles": {"cfg2": profile}, "max_gap_ms": 250,
                                      "scenes": {"task_0001/scene_1": {"path": "missing.npy"}}}))
    return dataset, parameters, raw, scene, feedback, profile


def snapshot(directory):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def test_generate_from_current_tcp_and_actual_feedback_without_overwriting(inputs):
    dataset, parameters, raw, scene, feedback, profile = inputs
    before = snapshot(dataset) | snapshot(raw)
    result = prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert result["RH20T_cfg2"]["new"] == 1
    assert json.loads((scene.parents[1] / "geometry.json").read_text()) == {
        "profiles": {"cfg2": profile}
    }
    geometry = np.load(scene / "geometry.npy", allow_pickle=True).item()
    assert set(geometry) == {"cam"}
    assert geometry["cam"].dtype == np.float32
    np.testing.assert_allclose(geometry["cam"][:, :7],
                               [[1, 2, 3, 1, 0, 0, 0], [2, 3, 4, 1, 0, 0, 0]])
    np.testing.assert_allclose(geometry["cam"][:, 7], [20 / 95, 40 / 95])
    assert (scene / "gripper.npy").read_bytes() == feedback.read_bytes()
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    completed = snapshot(dataset)
    result = prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert result["RH20T_cfg2"]["reused"] == 1
    assert snapshot(dataset) == completed
    # A changed trajectory must not replace the existing derived cache.
    tcp = np.load(scene / "tcp_base.npy", allow_pickle=True).item()
    tcp["cam"][0]["tcp"][0] = 9
    np.save(scene / "tcp_base.npy", tcp)
    preserved = snapshot(dataset)
    with pytest.raises(ValueError, match="TCP hash matched 0"):
        prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert snapshot(dataset) == preserved


def test_check_only_and_missing_feedback_add_nothing(inputs):
    dataset, parameters, raw, scene, feedback, _ = inputs
    before = snapshot(dataset)
    result = prepare_dataset_geometry(dataset, parameters, raw, check=True, cfgs=("cfg2",))
    assert result["RH20T_cfg2"]["new"] == 1
    assert snapshot(dataset) == before
    feedback.unlink()
    with pytest.raises(FileNotFoundError, match="no files added"):
        prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert snapshot(dataset) == before


def test_existing_local_feedback_is_never_overwritten(inputs):
    dataset, parameters, raw, scene, feedback, _ = inputs
    (scene / "gripper.npy").write_bytes(b"existing local feedback")
    before = snapshot(dataset)
    with pytest.raises(ValueError, match="Existing output differs"):
        prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert snapshot(dataset) == before


def test_scene_and_hardware_are_matched_by_tcp_hash(inputs):
    dataset, parameters, raw, scene, feedback, profile = inputs
    # Misleading names/provenance must not override the actual TCP hash match.
    (scene / "T_base_from_camera.json").write_text('{"raw_robot_scene": "wrong_scene"}')
    renamed = dataset / "RH20T_cfg3"
    scene.parents[1].rename(renamed)
    result = prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg3",))
    assert result["RH20T_cfg3"]["scenes"] == 1
    assert json.loads((renamed / "geometry.json").read_text()) == {
        "profiles": {"cfg2": profile}
    }


def test_conflicting_parameter_file_is_never_overwritten(inputs):
    dataset, parameters, raw, scene, feedback, _ = inputs
    (scene.parents[1] / "geometry.json").write_text('{"custom": true}')
    before = snapshot(dataset)
    with pytest.raises(ValueError, match="Existing output differs"):
        prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert snapshot(dataset) == before


def test_ambiguous_raw_tcp_hash_fails_without_adding_files(inputs):
    dataset, parameters, raw, scene, feedback, _ = inputs
    duplicate = raw / "RH20T_cfg2/task_0009_user_0002_scene_0001_cfg_0002/transformed"
    duplicate.mkdir(parents=True)
    (duplicate / "tcp_base.npy").write_bytes((scene / "tcp_base.npy").read_bytes())
    before = snapshot(dataset)
    with pytest.raises(ValueError, match="TCP hash matched 2"):
        prepare_dataset_geometry(dataset, parameters, raw, cfgs=("cfg2",))
    assert snapshot(dataset) == before
