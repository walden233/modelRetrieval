import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.data.rh20t.collate import collate_trajectories
from bise.data.rh20t.grasp_geometry import (
    bind_geometry, calibrated_grasp_tcp, geometry_digest, geometry_plan,
    measured_widths, prepare_grasp_geometry, validate_profile,
)
from bise.modalities.trajectory.augmentations import augment_robot_tcp_rotation
from bise.modalities.trajectory.factory import build_trajectory_dataset, build_trajectory_model
from bise.modalities.trajectory.evaluator import evaluate_trajectory_retrieval
from bise.modalities.trajectory.grasp import human_grasp_features, robot_grasp_features
from bise.modalities.trajectory.trainer import train_augmented_trajectory_epoch
from test_hand_3d_trajectory_dataset import _write_scene

HAND_11 = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17, 20]


def config(name):
    return json.loads((ROOT / f"configs/retrain/s43/trajectory/{name}.json").read_text())


def hand():
    points = torch.randn(2, 5, 21, 3) * .005
    points[..., 0, :] = torch.tensor([0., 0., 0.])
    points[..., 9, :] = torch.tensor([0., .06, 0.])
    points[..., 5, :] = torch.tensor([.03, .04, 0.])
    points[..., 17, :] = torch.tensor([-.03, .04, 0.])
    points[..., 4, :] = torch.tensor([-.02, .08, .01])
    points[..., 8, :] = torch.tensor([.02, .08, .01])
    return points + torch.arange(5)[None, :, None, None] * .01


def test_s43_matrix_and_keypoint_only_ablation():
    configs = {name: config(name) for name in ("T1", "T2", "T3", "T4", "T5", "N1", "N2", "N3")}
    assert all(c["seed"] == 43 and not c["use_6_keypoints"] for c in configs.values())
    assert all(c["num_epochs"] > 0 and "max_train_batches" not in c
               for c in configs.values())
    for name, c in configs.items():
        assert c["model_params"]["use_gripper"]
        assert c["model_params"]["robot_input_dim"] == (15 if name in ("N2", "N3") else 10)
        expected_dim = 63 if name == "T5" else (42 if name in ("N2", "N3") else 33)
        assert c["model_params"]["human_input_dim"] == expected_dim
        assert c["model_params"].get("human_keypoint_indices") == (
            None if name == "T5" else HAND_11)
    assert configs["T4"]["split"]["unit"] == "task"
    assert configs["T3"]["mode"] == "augment"
    assert not configs["T3"]["model_params"]["shared_transformer"]
    assert configs["T3"]["model_params"]["dual_adapters"]
    assert all(c["model_params"]["dual_adapters"] for c in configs.values())
    assert all(c["model_params"]["shared_transformer"]
               for name, c in configs.items() if name != "T3")
    for name in ("N1", "N3"):
        assert configs[name]["data_params"]["human_representation"] == "palm_relative"
    assert configs["N2"]["data_params"]["human_representation"] == "base_xyz"
    for name in ("N2", "N3"):
        assert configs[name]["model_params"]["grasp_geometry"]
        assert configs[name]["model_params"]["shared_transformer"]
    t2, t5 = copy.deepcopy(configs["T2"]), copy.deepcopy(configs["T5"])
    for c in (t2, t5):
        c.pop("experiment_name")
        c["model_params"].pop("human_keypoint_indices", None)
        c["model_params"]["human_input_dim"] = 63
    assert t2 == t5


def test_shared_weights_have_distinct_stems_and_adapters_and_roundtrip():
    cfg = config("T2")
    cfg["model_params"].update(d_model=16, nhead=4, num_layers=1,
                               dim_feedforward=32, proj_dim=8, dropout=0)
    model = build_trajectory_model(cfg, "cpu").eval()
    assert model.human_encoder.input_proj is not model.robot_encoder.input_proj
    assert model.human_encoder.transformer_encoder is model.robot_encoder.transformer_encoder
    assert model.human_encoder.pos_encoder is model.robot_encoder.pos_encoder
    assert model.human_adapter is not model.robot_adapter
    restored = build_trajectory_model(cfg, "cpu").eval()
    restored.load_state_dict(model.state_dict())
    points = hand()
    mask = torch.ones(2, 5, dtype=torch.bool)
    context = torch.ones(2, 5, 3)
    with torch.no_grad():
        torch.testing.assert_close(model.forward_human(points, mask, context),
                                   restored.forward_human(points, mask, context))


def test_baseline_uses_measured_opening_and_preserves_legacy_pose_only_model():
    cfg = config("T2")
    cfg["model_params"].update(d_model=16, nhead=4, num_layers=1,
                               dim_feedforward=32, proj_dim=8, dropout=0)
    model = build_trajectory_model(cfg, "cpu").eval()
    tcp = torch.zeros(2, 5, 8)
    tcp[..., 3] = 1
    tcp[..., 7] = .4
    tcp.requires_grad_()
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, -1] = False
    captured = []
    hook = model.robot_encoder.input_proj.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0]))
    model.forward_robot(tcp, mask, torch.ones(2, 5, 3))[..., 0].sum().backward()
    hook.remove()
    assert captured[0].shape == (2, 5, 10)
    torch.testing.assert_close(captured[0][0, :, -1], tcp[0, :, 7])
    assert captured[0][1, -1].abs().sum() == 0
    assert tcp.grad[..., 7].abs().sum() > 0
    assert tcp.grad[1, -1].abs().sum() == 0
    with pytest.raises(ValueError, match="measured opening"):
        model.forward_robot(tcp[..., :7], mask, torch.ones(2, 5, 3))
    cfg["model_params"].pop("use_gripper")
    cfg["model_params"]["robot_input_dim"] = 9
    legacy = build_trajectory_model(cfg, "cpu").eval()
    restored = build_trajectory_model(cfg, "cpu").eval()
    restored.load_state_dict(legacy.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(legacy.forward_robot(tcp[..., :7], mask, torch.ones(2, 5, 3)),
                                   restored.forward_robot(tcp[..., :7], mask, torch.ones(2, 5, 3)))


def test_training_batch_limit_preserves_dataset_and_counts_actual_steps():
    from test_retrain_comparison import load_entry
    module = load_entry("trajectory/train_trajectory")
    dataset = list(range(24))
    loader = DataLoader(dataset, batch_size=8)
    limited = module.LimitedBatches(loader, 1)
    assert len(limited) == 1 and len(loader.dataset) == 24
    assert len(list(limited)) == 1 and limited.steps == 1
    assert len(list(limited)) == 1 and limited.steps == 2
    assert len(module.LimitedBatches(loader, 10)) == 3
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            module.LimitedBatches(loader, invalid)


@pytest.mark.parametrize("name", ["T1", "T2", "T3", "T4", "T5", "N1", "N2", "N3"])
def test_keypoints_are_selected_after_common_augmentation_source(name):
    cfg = config(name)
    cfg["model_params"].update(d_model=16, nhead=4, num_layers=1,
                               dim_feedforward=32, proj_dim=8, dropout=0)
    model = build_trajectory_model(cfg, "cpu")
    points = hand().requires_grad_()
    mask = torch.ones(2, 5, dtype=torch.bool)
    context = torch.ones(2, 5, 3)
    captured = []
    hook = model.human_encoder.input_proj.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0]))
    model.forward_human(points, mask, context)[..., 0].sum().backward()
    hook.remove()
    assert captured[0].shape == (2, 5, cfg["model_params"]["human_input_dim"])
    selected = list(range(21)) if name == "T5" else HAND_11
    assert (points.grad[..., selected, :].abs().sum(dim=(0, 1, 3)) > 0).all()
    omitted = sorted(set(range(21)) - set(selected))
    assert points.grad[..., omitted, :].abs().sum() == 0


@pytest.mark.parametrize("indices", [None, HAND_11])
def test_local_hand_shape_is_rigid_and_size_invariant_but_global_motion_is_kept(indices):
    points = hand()
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3] = False
    baseline = human_grasp_features(points, mask, keypoint_indices=indices)
    assert baseline.shape[-1] == (72 if indices is None else 42)
    if indices is not None:
        full = human_grasp_features(points, mask)
        torch.testing.assert_close(baseline[..., :9], full[..., :9])
        torch.testing.assert_close(baseline[..., 9:],
                                   full[..., 9:].reshape(2, 5, 21, 3)[..., indices, :].flatten(-2))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    changed = human_grasp_features((points @ rotation.T) * 1.3 + 2, mask,
                                  keypoint_indices=indices)
    torch.testing.assert_close(baseline[..., 9:], changed[..., 9:], atol=2e-5, rtol=2e-5)
    assert not torch.allclose(baseline[..., :9], changed[..., :9])
    assert baseline[1, 3].abs().sum() == 0


def test_actual_feedback_units_timestamps_and_gap_rejection():
    profile = {"max_width_m": .1, "width_scale_to_m": .001}
    series = {1000: {"gripper_info": [20, 0, 0], "gripper_command": [100, 0, 0]},
              1100: {"gripper_info": [40, 0, 0], "gripper_command": [100, 0, 0]}}
    np.testing.assert_allclose(measured_widths(series, [1000, 1050, 1100], profile), [.2, .3, .4])
    with pytest.raises(ValueError, match="timestamps"):
        measured_widths(series, [5000], profile)
    with pytest.raises(ValueError, match="interval"):
        measured_widths({1000: series[1000], 2000: series[1100]}, [1500], profile)
    np.testing.assert_allclose(
        measured_widths({1000: series[1000], 2000: series[1100]}, [1000, 2000], profile), [.2, .4])
    with pytest.raises(ValueError, match="calibration range"):
        measured_widths(series, [1050], {**profile, "max_width_m": .01})


def test_single_feedback_sample_requires_exact_tcp_timestamp():
    profile = {"max_width_m": .1, "width_scale_to_m": .001}
    series = {1000: {"gripper_info": [40, 0, 0]}}
    np.testing.assert_allclose(measured_widths(series, [1000], profile), [.4])
    with pytest.raises(ValueError, match="exact TCP timestamps"):
        measured_widths(series, [1000, 1001], profile)
    with pytest.raises(ValueError, match="finite"):
        measured_widths(series, [float("nan")], profile)


def test_moving_tool_offset_and_measured_jaw_points():
    transform = np.eye(4)
    transform[0, 3] = .1
    tcp = np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 2**-.5, 0, 0, 2**-.5]])
    prepared = calibrated_grasp_tcp(tcp, [.4, .8], {"tcp_to_grasp": transform.tolist()})
    np.testing.assert_allclose(prepared[:, :3], [[.1, 0, 0], [0, .1, 0]], atol=1e-6)
    mask = torch.ones(1, 2, dtype=torch.bool)
    features = robot_grasp_features(torch.tensor(prepared)[None], mask)
    torch.testing.assert_close(features[0, :, 9:],
                               torch.tensor([[-.2, 0, 0, .2, 0, 0], [-.4, 0, 0, .4, 0, 0]]))
    augmented = augment_robot_tcp_rotation(torch.tensor(prepared)[None])
    torch.testing.assert_close(augmented[..., 7], torch.tensor(prepared)[None, ..., 7])


def prepared_fixture(tmp_path):
    manifests = tmp_path / "manifests"
    raw = tmp_path / "raw"
    original = "task_0001_user_0001_scene_0001_cfg_0002"
    source = raw / original / "transformed"
    source.mkdir(parents=True)
    series = {t: {"gripper_info": [40, 0, 0], "gripper_command": [100, 0, 0]}
              for t in range(5000, 6300, 100)}
    np.save(source / "gripper.npy", {"cam": series})
    from bise.data.rh20t.hand_3d_trajectory_dataset import build_hand_3d_trajectory_manifest
    for cfg in ("cfg2", "cfg3"):
        root = tmp_path / f"RH20T_{cfg}"
        scene = _write_scene(root)
        source_identity = json.dumps({"raw_robot_scene": original})
        (scene / "T_base_from_camera.json").write_text(source_identity)
        directory = manifests / cfg
        directory.mkdir(parents=True)
        (directory / "data_manifest.json").write_text(
            json.dumps(build_hand_3d_trajectory_manifest(str(root))))
    geometry = {"raw_roots": {"cfg2": str(raw)},
                "profiles": {"cfg2": {"source": "measured fixture",
                                      "tcp_to_grasp": np.eye(4).tolist(),
                                      "max_width_m": .1, "width_scale_to_m": .001}}}
    return manifests, geometry


def test_prepared_trajectory_entry_trains_one_batch_without_changing_split(tmp_path, monkeypatch):
    from bise.common import evaluation, training
    from bise.common.fingerprints import write_json
    from test_retrain_comparison import load_entry
    manifests, settings = prepared_fixture(tmp_path)
    paths = prepare_grasp_geometry(settings, manifests)
    cfg = config("T2")
    cfg.update(data_root=str(tmp_path / "RH20T_cfg2"), device="cpu",
               num_epochs=1, max_train_batches=1)
    cfg["data_manifests"] = {f"RH20T_{c}": str(manifests / c / "data_manifest.json")
                             for c in ("cfg2", "cfg3")}
    cfg["data_params"]["geometry_manifests"] = {f"RH20T_{c}": p for c, p in paths.items()}
    cfg["model_params"].update(d_model=16, nhead=4, num_layers=1,
                               dim_feedforward=32, proj_dim=8, dropout=0)
    cfg["split"] = {"unit": "scene", "ratios": {"train": 1., "val": 0., "test": 0.}}
    # This single-scene fixture exercises training; protocol binding is covered separately.
    bind = lambda run, current: write_json(run / "protocols.json", {})
    monkeypatch.setattr(training, "bind_protocols", bind)
    monkeypatch.setattr(evaluation, "bind_protocols", bind)
    run = tmp_path / "runs/T2"
    training.prepare_training_run(cfg, "trajectory", run)
    training.validate_prepared_run(run)
    split = (run / "split_manifest.json").read_bytes()
    module = load_entry("trajectory/train_trajectory")
    monkeypatch.setattr(sys, "argv", ["train_trajectory", "--config", str(run / "params.json"),
                                     "--run-dir", str(run), "--prepared"])
    module.main()
    assert (run / "split_manifest.json").read_bytes() == split
    assert (run / "best_model.pth").is_file()
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "complete" and status["train_batches"] == 1


def test_preparation_uses_actual_source_cfg_and_freezes_calibration(tmp_path):
    manifests, settings = prepared_fixture(tmp_path)
    _, report = geometry_plan(settings, manifests)
    assert report["ready"]
    assert report["source_cfg_counts"]["cfg3"] == {"cfg2": 1}
    outputs = prepare_grasp_geometry(settings, manifests)
    assert outputs == prepare_grasp_geometry(settings, manifests)
    changed = copy.deepcopy(settings)
    changed["profiles"]["cfg2"]["max_width_m"] = .2
    with pytest.raises(ValueError, match="Calibration differs"):
        prepare_grasp_geometry(changed, manifests)


def test_aligned_tcp_proxy_preserves_motion_and_records_virtual_geometry(tmp_path):
    manifests, settings = prepared_fixture(tmp_path)
    profile = settings["profiles"]["cfg2"]
    profile["reference"] = "aligned_tcp_proxy"
    validate_profile(profile)
    paths = prepare_grasp_geometry(settings, manifests)
    frozen = json.loads(Path(paths["cfg2"]).read_text())
    assert frozen["references"] == {"cfg2": "aligned_tcp_proxy"}
    assert "virtual" in frozen["model"]
    cached = np.load(frozen["scenes"]["task_0001/scene_1"]["path"], allow_pickle=True).item()
    original = np.load(tmp_path / "RH20T_cfg2/task_0001/scene_1/tcp_base.npy",
                       allow_pickle=True).item()
    np.testing.assert_allclose(cached["cam"][:, :7], [frame["tcp"] for frame in original["cam"]])
    bad = copy.deepcopy(profile)
    bad["tcp_to_grasp"][0][3] = .01
    with pytest.raises(ValueError, match="identity"):
        validate_profile(bad)


@pytest.mark.parametrize("name", ["T2", "T5", "N1", "N2", "N3"])
@pytest.mark.parametrize("storage", ["legacy", "dataset"])
def test_geometry_dataset_joint_training_roundtrip_and_source_binding(tmp_path, name, storage):
    manifests, settings = prepared_fixture(tmp_path)
    paths = prepare_grasp_geometry(settings, manifests)
    if storage == "dataset":
        from bise.data.rh20t.hand_3d_trajectory_dataset import build_hand_3d_trajectory_manifest
        for c in ("cfg2", "cfg3"):
            root = tmp_path / f"RH20T_{c}"
            legacy = json.loads(Path(paths[c]).read_text())
            entry = legacy["scenes"]["task_0001/scene_1"]
            (root / "task_0001/scene_1/geometry.npy").write_bytes(Path(entry["path"]).read_bytes())
            parameters = root / "geometry.json"
            parameters.write_text(json.dumps({"profiles": settings["profiles"]}))
            paths[c] = str(parameters)
            (manifests / c / "data_manifest.json").write_text(
                json.dumps(build_hand_3d_trajectory_manifest(str(root))))
            inventory = json.loads((manifests / c / "data_manifest.json").read_text())
            assert "geometry_parameters_sha256" not in inventory
            assert all("geometry_sha256" not in row for row in inventory["scenes"])
            (manifests / c / "geometry.json").unlink()
            shutil.rmtree(manifests / c / "geometry")
        shutil.rmtree(tmp_path / "raw")
    cfg = config(name)
    cfg["data_root"] = str(tmp_path / "RH20T_cfg2")
    cfg["data_manifests"] = {f"RH20T_{c}": str(manifests / c / "data_manifest.json")
                             for c in ("cfg2", "cfg3")}
    cfg["data_params"]["geometry_manifests"] = {f"RH20T_{c}": p for c, p in paths.items()}
    cfg["model_params"].update(d_model=16, nhead=4, num_layers=1,
                               dim_feedforward=32, proj_dim=8, dropout=0)
    bind_geometry(cfg)
    dataset = build_trajectory_dataset(cfg)
    item = dataset[0]
    assert item["tcp_bases"][0].shape[-1] == 8
    assert item["human_poses"][0].shape[-2:] == (21, 3)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_trajectories)
    model = build_trajectory_model(cfg, "cpu")
    result = train_augmented_trajectory_epoch(model, loader,
                                              torch.optim.AdamW(model.parameters(), lr=2e-4),
                                              "cpu", .3, use_task_labels=True)
    assert np.isfinite(result).all()
    restored = build_trajectory_model(cfg, "cpu")
    restored.load_state_dict(model.state_dict())
    result = evaluate_trajectory_retrieval(restored, loader, "cpu")
    assert result["robot_embeddings"].shape == (1, 8)
    cfg3 = copy.deepcopy(cfg)
    cfg3["data_root"] = str(tmp_path / "RH20T_cfg3")
    target = build_trajectory_dataset(cfg3)
    evaluated = evaluate_trajectory_retrieval(
        restored, DataLoader(target, batch_size=1, collate_fn=collate_trajectories), "cpu")
    assert evaluated["metadata"]["robot_scene_ids"] == ["task_0001/scene_1"]
    transform = np.eye(4)
    transform[:3, 3] = [.1, -.2, .3]
    shifted = dataset.with_human_transform(transform)[0]
    if name in ("N1", "N3"):
        torch.testing.assert_close(item["human_poses"][0], shifted["human_poses"][0])
    else:
        assert not torch.allclose(item["human_poses"][0], shifted["human_poses"][0])
    torch.testing.assert_close(item["tcp_bases"][0], shifted["tcp_bases"][0])
    digest = geometry_digest(cfg, ["task_0001/scene_1"])
    if storage == "legacy":
        assert digest == cfg["geometry_hashes"]["RH20T_cfg2"]
    else:
        assert digest != cfg["geometry_hashes"]["RH20T_cfg2"]
        cache = tmp_path / "RH20T_cfg2/task_0001/scene_1/geometry.npy"
        original = cache.read_bytes()
        cached = np.load(cache, allow_pickle=True).item()
        cached["cam"][0, 7] = -1
        np.save(cache, cached)
        assert geometry_digest(cfg, ["task_0001/scene_1"]) != digest
        with pytest.raises(ValueError, match="Invalid calibrated grasp"):
            dataset[0]
        cache.write_bytes(original)
    changed = json.loads(Path(paths["cfg2"]).read_text())
    changed["profiles"]["cfg2"]["source"] = "edited"
    Path(paths["cfg2"]).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed since training"):
        build_trajectory_dataset(cfg)


def test_runtime_batch_binds_geometry_paths_and_one_epoch_t3(tmp_path):
    from test_retrain_comparison import load_entry
    module = load_entry("retrain")
    source = json.loads((ROOT / "configs/retrain/s43/run.json").read_text())
    source["manifest_dir"] = str(tmp_path / "manifests")
    registries, runtime = module.prepare_batch(
        source, tmp_path / "runs/s44", ["trajectory"], 43,
        {"trajectory": 1, "video": None}, device="cpu")
    assert list(registries["trajectory"]["groups"]) == ["T1", "T2", "T3", "T4",
                                                        "T5", "N1", "N2", "N3"]
    t3 = json.loads(Path(runtime["trajectory"]["T3"]["config"]).read_text())
    assert t3["mode"] == "augment" and t3["num_epochs"] == 1
    n3 = json.loads(Path(runtime["trajectory"]["N3"]["config"]).read_text())
    assert n3["data_params"]["geometry_manifests"]["RH20T_cfg3"] == str(
        ROOT / "dataset/RH20T_subset/RH20T_cfg3/geometry.json")
    assert "source_dir" not in runtime
    assert runtime["eval_manifests"]["cfg2"] == str(
        tmp_path / "manifests/s44/cfg2/metadata.json")
    assert n3["data_manifests"]["RH20T_cfg2"] == str(
        tmp_path / "manifests/s44/cfg2/data_manifest.json")


def test_selected_audit_excludes_unprepared_geometry_groups(tmp_path, monkeypatch):
    from test_retrain_comparison import load_entry
    module = load_entry("retrain")
    source = json.loads((ROOT / "configs/retrain/s43/run.json").read_text())
    source["output_root"] = str(tmp_path / "runs")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(source))
    def audit(current, files):
        assert list(current["trajectory"]) == ["T2"]
        assert current["video"] == {} and files
        return {}
    monkeypatch.setitem(sys.modules, "audit_splits", SimpleNamespace(audit=audit))
    monkeypatch.setattr(sys, "argv", ["retrain", "--config", str(path), "--modality",
                                     "trajectory", "--stage", "audit", "--only", "T2"])
    module.main()


@pytest.mark.parametrize("group,expected", [("T2", 1), ("N3", 1)])
def test_geometry_preparation_is_scoped_to_selected_groups(tmp_path, monkeypatch, group, expected):
    from test_retrain_comparison import load_entry
    from bise.data.rh20t import grasp_geometry, splits
    module = load_entry("retrain")
    source = json.loads((ROOT / "configs/retrain/s43/run.json").read_text())
    source["output_root"] = str(tmp_path / "runs")
    source["manifest_dir"] = str(tmp_path / "manifests")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(source))
    calls = []
    def prepare_protocols(*args):
        assert args[1] is None
        return {}
    monkeypatch.setattr(splits, "prepare_protocols", prepare_protocols)
    monkeypatch.setattr(grasp_geometry, "validate_dataset_geometry",
                        lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["retrain", "--config", str(path), "--modality",
                                     "trajectory", "--stage", "prepare", "--only", group])
    module.main()
    assert len(calls) == expected


def test_geometry_entry_creates_only_per_cfg_manifests(tmp_path, monkeypatch):
    from test_retrain_comparison import load_entry
    manifests, settings = prepared_fixture(tmp_path)
    path = tmp_path / "grippers.json"
    path.write_text(json.dumps(settings))
    module = load_entry("trajectory/prepare_geometry")
    monkeypatch.setattr(sys, "argv", ["prepare_geometry", "--config", str(path),
                                     "--manifests", str(manifests)])
    assert module.main() == 0
    assert not (manifests / "geometry.json").exists()
    assert all((manifests / cfg / "geometry.json").is_file() for cfg in ("cfg2", "cfg3"))
