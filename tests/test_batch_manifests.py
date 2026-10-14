"""Check the supplied batch metadata and aligned modality split format."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bise.common.evaluation import bind_protocols
from bise.common.fingerprints import file_hash, without_sha256, write_json
from bise.data.rh20t.splits import (
    load_protocol, load_split_map, prepare_protocols, protocol_hash, shared_scene_split,
)
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset
from bise.modalities.video.factory import build_split_manifest as video_manifest
from bise.modalities.video.factory import split_video_dataset
from test_retrain_comparison import ROOT, load_entry
from test_retrain_splits import frozen_protocol


def batch_protocol(tmp_path, cfg="cfg2", count=180):
    _, old, identities = frozen_protocol(tmp_path, cfg, count, with_files=True)
    directory = tmp_path / "manifests/s42" / cfg
    data_path = directory / "data_manifest.json"
    write_json(data_path, json.loads(Path(old["data_manifest"]).read_text()))
    mapping = {"train": list(reversed(identities[170:])) if cfg == "cfg2" else [],
               "val": list(reversed(old["val"])), "test": list(reversed(old["test"]))}
    paths = {modality: directory / f"split_manifest_{modality}.json"
             for modality in ("trajectory", "video")}
    write_json(paths["trajectory"], mapping)
    write_json(paths["video"], {name: [pair["sample_id"] for identity in scenes
                                      for pair in old["video_pairs"][identity]]
                               for name, scenes in mapping.items()})
    metadata = {key: old[key] for key in ("protocol", "cfg", "seed", "data_root")}
    metadata.update(schema_version=3, data_manifest=str(data_path),
                    data_sha256=file_hash(data_path),
                    split_manifests={key: str(path) for key, path in paths.items()})
    if cfg == "cfg3":
        metadata["scenes_per_task"] = 2
    path = directory / "metadata.json"
    write_json(path, metadata)
    return path, metadata, mapping


def test_cleanup_preserves_manifest_order_and_matching_run_bindings(tmp_path):
    path, metadata, _ = batch_protocol(tmp_path)
    data_path = Path(metadata["data_manifest"])
    data = json.loads(data_path.read_text())
    data["scenes"][0].update(human_sha256="old human digest", robot_sha256="old robot digest")
    data["scenes"][0]["cameras"]["0"]["sha256"] = "old nested digest"
    write_json(data_path, data)
    before = load_protocol(path)
    root = path.parent.parent.parent
    original = {source: source.read_bytes() for source in root.rglob("*.json")}
    run = tmp_path / "runs/T1"
    config = {"protocol": "retrain_v2", "eval_manifests": {"cfg2": str(path)}}
    bind_protocols(run, config)
    write_json(run / "data_manifest.json", data)
    snapshot = (run / "data_manifest.json").read_bytes()
    stale = tmp_path / "runs/stale/protocols.json"
    write_json(stale, {"cfg2": {"path": str(path), "sha256": "unrelated old protocol"}})
    stale_bytes = stale.read_bytes()
    cleaner = load_entry("clean_manifests")
    assert cleaner.clean_manifests(root, tmp_path / "runs", check=True) == {
        "manifests": 2, "protocol_bindings": 1, "check": True}
    assert all(source.read_bytes() == content for source, content in original.items())
    assert cleaner.clean_manifests(root, tmp_path / "runs") == {
        "manifests": 2, "protocol_bindings": 1, "check": False}
    assert json.loads(data_path.read_text()) == without_sha256(data)
    assert load_protocol(path) == without_sha256(before)
    bind_protocols(run, config)
    assert (run / "data_manifest.json").read_bytes() == snapshot
    assert stale.read_bytes() == stale_bytes
    assert cleaner.clean_manifests(root, tmp_path / "runs")["manifests"] == 0


def test_batch_splits_keep_modality_scene_and_camera_order(tmp_path):
    path, metadata, mapping = batch_protocol(tmp_path)
    payload = load_protocol(path)
    for source in metadata["split_manifests"].values():
        assert load_protocol(source) == payload
        assert protocol_hash(source) == protocol_hash(path)

    dataset = SimpleNamespace(scenes=[SimpleNamespace(scene_path=f"/data/{identity}")
                                     for identity in reversed(payload["eligible_scenes"])])
    trajectory = split_trajectory_dataset(dataset, {
        "shared_manifest_path": metadata["split_manifests"]["trajectory"]})
    assert build_split_manifest(trajectory) == mapping

    samples = [SimpleNamespace(sample_id=pair["sample_id"], scene_id=identity,
                                task_id=identity.split('/')[0])
               for identity, pairs in sorted(payload["video_pairs"].items()) for pair in pairs]
    video = split_video_dataset(SimpleNamespace(samples=samples), {
        "shared_manifest_path": metadata["split_manifests"]["video"]})
    assert video_manifest(video) == load_split_map(metadata["split_manifests"]["video"])


@pytest.mark.parametrize("change", ["scene_order", "camera_count", "duplicate_camera"])
def test_batch_rejects_misaligned_video_split(tmp_path, change):
    path, metadata, _ = batch_protocol(tmp_path)
    video_path = Path(metadata["split_manifests"]["video"])
    mapping = load_split_map(video_path)
    if change == "scene_order":
        mapping["test"][:4] = mapping["test"][2:4] + mapping["test"][:2]
    elif change == "camera_count":
        mapping["test"].pop()
    else:
        mapping["test"][1] = mapping["test"][0]
    write_json(video_path, mapping)
    with pytest.raises(ValueError):
        load_protocol(path)


def test_batch_protocol_binding_includes_both_modality_splits(tmp_path):
    _, metadata, _ = batch_protocol(tmp_path)
    config = {"protocol": "retrain_v2", "eval_manifests": {
        "cfg2": metadata["split_manifests"]["trajectory"]}}
    run = tmp_path / "run"
    bind_protocols(run, config)
    video = Path(metadata["split_manifests"]["video"])
    mapping = load_split_map(video)
    mapping["test"][:2] = reversed(mapping["test"][:2])
    write_json(video, mapping)
    with pytest.raises(ValueError, match="different frozen protocol"):
        bind_protocols(run, config)


def test_batch_fixed_and_random_validation_preserve_test(tmp_path):
    path, _, mapping = batch_protocol(tmp_path, count=240)
    payload = load_protocol(path)
    base = {"shared_manifest_path": str(path)}
    fixed = shared_scene_split(payload["eligible_scenes"], base)
    assert fixed == mapping
    random = shared_scene_split(payload["eligible_scenes"],
                                {**base, "use_fixed_val": False, "seed": 43})
    assert random["test"] == fixed["test"]
    assert random["val"] != fixed["val"] and len(random["val"]) == 50


def test_prepare_reuses_supplied_batch_files_without_rewriting(tmp_path):
    paths = [batch_protocol(tmp_path, cfg, count)[0]
             for cfg, count in (("cfg2", 180), ("cfg3", 2))]
    directory = paths[0].parent.parent
    before = {path: path.read_bytes() for path in directory.rglob("*.json")}
    result = prepare_protocols(tmp_path, tmp_path / "unused", directory)
    assert result == {cfg: str(path) for cfg, path in zip(("cfg2", "cfg3"), paths)}
    assert all(path.read_bytes() == value for path, value in before.items())
    with pytest.raises(ValueError, match="different seed"):
        prepare_protocols(tmp_path, tmp_path / "unused", directory, seed=43)


def test_new_batches_create_separate_metadata_and_aligned_splits(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for cfg, count in (("cfg2", 180), ("cfg3", 2)):
        _, payload, _ = frozen_protocol(tmp_path, cfg, count, with_files=True)
        data = json.loads(Path(payload["data_manifest"]).read_text())
        for entry in data["scenes"]:
            for filename, key in (("human_hand_3d.npy", "human_sha256"),
                                  ("tcp_base.npy", "robot_sha256")):
                file = Path(data["data_root"]) / entry["scene_id"] / filename
                file.write_bytes(b"fixture trajectory")
                entry[key] = file_hash(file)
        write_json(source / f"{cfg}_valid.json", data)
    from bise.data.rh20t import splits
    def no_video_hash(path):
        assert Path(path).suffix not in (".mp4", ".npy"), "Preparation must not hash source data"
        assert Path(path).name != "data_manifest.json", "Preparation must not hash inventories"
        return file_hash(path)
    monkeypatch.setattr(splits, "file_hash", no_video_hash)
    monkeypatch.setitem(sys.modules, "decord", SimpleNamespace(
        VideoReader=lambda *a, **kw: pytest.fail("Preparation must not decode videos")))
    first = prepare_protocols(tmp_path, source, tmp_path / "manifests/s42")
    second = prepare_protocols(tmp_path, source, tmp_path / "manifests/s43", seed=43)
    for cfg, path in first.items():
        metadata = json.loads(Path(path).read_text())
        assert metadata["schema_version"] == 3
        assert metadata == without_sha256(metadata)
        inventory = json.loads(Path(metadata["data_manifest"]).read_text())
        assert inventory == without_sha256(inventory)
        assert not {"train", "val", "test", "video_pairs"} & metadata.keys()
        assert set(metadata["split_manifests"]) == {"trajectory", "video"}
        assert set(load_split_map(metadata["split_manifests"]["trajectory"])) == {
            "train", "val", "test"}
        load_protocol(path)
    assert load_protocol(first["cfg2"])["test"] != load_protocol(second["cfg2"])["test"]
    assert load_protocol(first["cfg2"])["test_task_max"] == 98


@pytest.mark.parametrize("limit", [98, 50, None])
def test_preparation_freezes_cfg2_task_range_for_both_modalities(tmp_path, limit):
    from bise.data.rh20t.task_split import split_task_dataset
    source = tmp_path / "source"
    for cfg, count in (("cfg2", 432), ("cfg3", 2)):
        _, old, _ = frozen_protocol(tmp_path, cfg, count, with_files=True)
        data = json.loads(Path(old["data_manifest"]).read_text())
        for entry in data["scenes"]:
            for filename, key in (("human_hand_3d.npy", "human_sha256"),
                                  ("tcp_base.npy", "robot_sha256")):
                path = Path(data["data_root"]) / entry["scene_id"] / filename
                path.write_bytes(b"fixture trajectory")
                entry[key] = file_hash(path)
        write_json(source / f"{cfg}_valid.json", data)
    directory = tmp_path / "manifests/new"
    result = prepare_protocols(tmp_path, source, directory,
                               cfg2_test_limit=limit)
    protocol = load_protocol(result["cfg2"])
    assert protocol["test_task_max"] == limit
    assert len(protocol["test"]) == 120 and len(protocol["val"]) == 50
    assert sum(len(protocol[name]) for name in ("train", "val", "test")) == 432
    tasks = {int(scene.split('/')[0].removeprefix("task_")) for scene in protocol["test"]}
    if limit is not None:
        assert all(1 <= task <= limit for task in tasks)
    else:
        assert any(task > 98 for task in tasks)
    assert "test_task_max" not in load_protocol(result["cfg3"])
    video = load_split_map(protocol["split_manifests"]["video"])
    assert ["/".join(sample.split("::")[1:3]) for sample in video["test"][::2]] == (
        protocol["test"])

    # Task-held-out runs inherit the frozen preparation policy, without a per-run switch.
    available = protocol["eligible_scenes"]
    trajectory = SimpleNamespace(scenes=[SimpleNamespace(scene_path=f"/data/{scene}")
                                         for scene in available])
    videos = SimpleNamespace(samples=[SimpleNamespace(
        scene_id=scene, sample_id=f"rh20t::{scene.replace('/', '::')}::{camera}")
        for scene in available for camera in ("0", "1")])
    config = {"unit": "task", "seed": 42, "scene_counts": {"test": 120, "val": 50}}
    traj = build_split_manifest(split_task_dataset(trajectory, config, "trajectory",
                                                  result["cfg2"]))
    vid = video_manifest(split_task_dataset(videos, config, "video", result["cfg2"]))
    assert len(traj["test"]) == 120
    for name in traj:
        assert ["/".join(sample.split("::")[1:3]) for sample in vid[name][::2]] == traj[name]
    task_sets = {name: {scene.split('/')[0] for scene in scenes}
                 for name, scenes in traj.items()}
    assert not task_sets["test"] & (task_sets["train"] | task_sets["val"])
    assert not task_sets["train"] & task_sets["val"]
    if limit is not None:
        assert all(1 <= int(task.removeprefix("task_")) <= limit
                   for task in task_sets["test"])

    # Existing batches retain their original files even if the preparation option changes.
    before = {path: path.read_bytes() for path in directory.rglob("*.json")}
    prepare_protocols(tmp_path, source, directory, cfg2_test_limit=None if limit else 98)
    assert all(path.read_bytes() == value for path, value in before.items())
    if limit is None:
        metadata = json.loads(Path(result["cfg2"]).read_text())
        metadata["test_task_max"] = 98
        write_json(result["cfg2"], metadata)
        with pytest.raises(ValueError, match="outside the frozen task range"):
            load_protocol(result["cfg2"])


def test_video_pair_inventory_checks_real_files_and_keeps_camera_priority(tmp_path):
    from bise.data.rh20t.splits import _video_pairs
    identity = "task_0001/scene_1"
    scene = tmp_path / identity
    scene.mkdir(parents=True)
    for camera in ("2", "4"):
        for domain in ("human", "robot"):
            (scene / f"cam_{camera}_{domain}.mp4").touch()
    for domain in ("human", "robot"):
        (scene / f"cam_0_{domain}.mp4").mkdir()
        (scene / f"cam_1_{domain}.mp4").symlink_to(scene / "missing.mp4")
    (scene / "cam_3_human.mp4").touch()
    pairs = _video_pairs(tmp_path, {identity: {"cameras": {"4": {}}}})[identity]
    assert [p["camera_id"] for p in pairs] == ["4", "2"]
    assert all(set(pair) == {"camera_id", "sample_id", "human_path", "robot_path"}
               for pair in pairs)


def test_existing_batch_rejects_missing_frozen_video_without_substitution(tmp_path):
    path, metadata, mapping = batch_protocol(tmp_path)
    root = Path(metadata["data_root"])
    (root / mapping["test"][0] / "cam_0_human.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="Frozen video file missing"):
        load_protocol(path)


def test_new_batch_freezes_one_inventory_without_source_directory(tmp_path, monkeypatch):
    from bise.data.rh20t import hand_3d_trajectory_dataset
    inventories = {}
    for cfg, count in (("cfg2", 180), ("cfg3", 2)):
        _, payload, _ = frozen_protocol(tmp_path, cfg, count, with_files=True)
        data = json.loads(Path(payload["data_manifest"]).read_text())
        for entry in data["scenes"]:
            for filename, key in (("human_hand_3d.npy", "human_sha256"),
                                  ("tcp_base.npy", "robot_sha256")):
                file = Path(data["data_root"]) / entry["scene_id"] / filename
                file.write_bytes(b"fixture trajectory")
                entry[key] = file_hash(file)
        inventories[f"RH20T_{cfg}"] = data
    monkeypatch.setattr(hand_3d_trajectory_dataset, "build_hand_3d_trajectory_manifest",
                        lambda root: inventories[Path(root).name])
    output = tmp_path / "manifests/new"
    result = prepare_protocols(tmp_path, None, output, seed=43)
    expected = {f"{cfg}/{name}.json" for cfg in ("cfg2", "cfg3") for name in (
        "metadata", "data_manifest", "split_manifest_trajectory", "split_manifest_video")}
    assert {str(path.relative_to(output)) for path in output.rglob("*.json")} == expected
    assert not (output / "source").exists()
    before = {path: path.read_bytes() for path in output.rglob("*.json")}

    def unexpected_rescan(root):
        raise AssertionError(f"Frozen inventory must be reused: {root}")
    monkeypatch.setattr(hand_3d_trajectory_dataset, "build_hand_3d_trajectory_manifest",
                        unexpected_rescan)
    # Recover a partial preparation from the one authoritative inventory.
    (output / "cfg3/metadata.json").unlink()
    assert prepare_protocols(tmp_path, None, output, seed=43) == result
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(load_protocol(result["cfg2"])["test"]) == 120
    assert len(load_protocol(result["cfg3"])["test"]) == 2


def test_independent_audit_derives_manifests_from_batch(tmp_path):
    for cfg, count in (("cfg2", 180), ("cfg3", 2)):
        batch_protocol(tmp_path, cfg, count)
    module = load_entry("audit_splits")
    report = module.audit({"manifest_dir": str(tmp_path / "manifests"), "batch": "s42",
                           "trajectory": {}, "video": {}})
    assert report["protocols"]["cfg2"]["test"] == 120
    assert report["protocols"]["cfg3"]["test"] == 2


def test_file_audit_checks_video_existence_without_hashing(tmp_path, monkeypatch):
    from bise.data.rh20t import splits
    for cfg, count in (("cfg2", 180), ("cfg3", 2)):
        path, metadata, _ = batch_protocol(tmp_path, cfg, count)
        data_path = Path(metadata["data_manifest"])
        data = json.loads(data_path.read_text())
        for entry in data["scenes"]:
            for filename, key in (("human_hand_3d.npy", "human_sha256"),
                                  ("tcp_base.npy", "robot_sha256")):
                source = Path(data["data_root"]) / entry["scene_id"] / filename
                source.write_bytes(b"fixture trajectory")
                entry[key] = file_hash(source)
        write_json(data_path, data)
        metadata["data_sha256"] = file_hash(data_path)
        write_json(path, metadata)
    module = load_entry("audit_splits")
    def no_video_hash(path):
        assert Path(path).suffix not in (".mp4", ".npy"), "Audit must not hash source data"
        return file_hash(path)
    monkeypatch.setattr(splits, "file_hash", no_video_hash)
    monkeypatch.setattr(module, "file_hash", no_video_hash)
    report = module.audit({"manifest_dir": str(tmp_path / "manifests"), "batch": "s42",
                           "trajectory": {}, "video": {}}, files=True)
    assert report["protocols"]["cfg2"]["test"] == 120


def test_runtime_paths_follow_batch_and_preserve_task_configs(tmp_path):
    module = load_entry("retrain")
    config = json.loads((ROOT / "configs/retrain/s43/run.json").read_text())
    config["manifest_dir"] = str(tmp_path / "manifests")
    config["trajectory"] = {key: config["trajectory"][key] for key in ("T1", "T4")}
    config["video"] = {key: config["video"][key] for key in ("E1", "E6")}
    root = tmp_path / "runs/another"
    registries, runtime = module.prepare_batch(config, root, ["trajectory", "video"], 43,
                                              {"trajectory": None, "video": None})
    manifests = tmp_path / "manifests/another"
    assert runtime["manifest_dir"] == str(manifests)
    for modality, registry in registries.items():
        for key, entry in registry["groups"].items():
            current = json.loads(Path(entry["config"]).read_text())
            for cfg in ("cfg2", "cfg3"):
                assert current["eval_manifests"][cfg] == str(
                    manifests / cfg / f"split_manifest_{modality}.json")
            if key in ("T4", "E6"):
                source = json.loads((ROOT / config[modality][key]["config"]).read_text())
                assert current["split"] == {"unit": "task", "seed": source["split"]["seed"],
                                            "scene_counts": {"test": 120, "val": 50}}
            else:
                assert current["split"]["shared_manifest_path"] == current["eval_manifests"]["cfg2"]


def test_video_pipeline_accepts_batch_metadata_and_splits(tmp_path, monkeypatch):
    import test_retrain_video

    def protocol(directory, cfg="cfg2", count=180, with_files=False):
        _, metadata, mapping = batch_protocol(directory, cfg, count)
        return Path(metadata["split_manifests"]["video"]), metadata, mapping

    monkeypatch.setattr(test_retrain_video, "frozen_protocol", protocol)
    test_retrain_video.test_video_training_evaluation_and_frozen_raw_comparison(
        tmp_path, monkeypatch)
