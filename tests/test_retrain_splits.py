import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.common.fingerprints import file_hash, object_hash, write_json
from bise.common.evaluation import resolve_eval_split
from bise.data.rh20t.splits import PROTOCOL, load_protocol, shared_scene_split, validate_split
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset
from bise.modalities.video.factory import build_split_manifest as video_manifest
from bise.modalities.video.factory import split_video_dataset


def frozen_protocol(tmp_path, cfg="cfg2", count=180, with_files=False):
    root = tmp_path / f"RH20T_{cfg}"
    root.mkdir(exist_ok=True)
    identities = [f"task_{i // 4:04d}/scene_{i % 4 + 1}" for i in range(count)]
    entries, pairs = [], {}
    for identity in identities:
        task, name = identity.split("/")
        entries.append({"scene_id": identity, "task_id": task, "cameras": {"0": {}, "1": {}}})
        pairs[identity] = []
        for camera in ("0", "1"):
            pair = {"camera_id": camera, "sample_id": f"rh20t::{task}::{name}::{camera}"}
            for domain in ("human", "robot"):
                relative = f"{identity}/cam_{camera}_{domain}.mp4"
                pair[f"{domain}_path"] = relative
                pair[f"{domain}_sha256"] = "placeholder"
                if with_files:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture video")
                    pair[f"{domain}_sha256"] = file_hash(path)
            pairs[identity].append(pair)
    data = {"human_source": "hand_3d_base", "schema_version": 2,
            "data_root": str(root), "scenes": entries}
    data_path = tmp_path / f"{cfg}_data.json"
    write_json(data_path, data)
    val, test = (identities[120:170], identities[:120]) if cfg == "cfg2" else ([], identities)
    payload = {"protocol": PROTOCOL, "schema_version": 2, "cfg": cfg, "seed": 42,
               "data_root": str(root), "data_manifest": str(data_path),
               "data_sha256": file_hash(data_path), "eligible_scenes": identities,
               "video_pairs": pairs, "inventory_sha256": object_hash(pairs),
               "val": val, "test": test}
    path = tmp_path / f"{cfg}_split_manifest.json"
    write_json(path, payload)
    return path, payload, identities


def test_shared_fixed_val_is_default_and_test_ignores_training_seed(tmp_path):
    path, payload, identities = frozen_protocol(tmp_path)
    base = {"shared_manifest_path": str(path), "unit": "scene"}
    first = shared_scene_split(identities, {**base, "seed": 42})
    second = shared_scene_split(list(reversed(identities)), {**base, "seed": 43})
    assert first == second
    assert first["test"] == payload["test"]
    assert first["val"] == payload["val"]
    assert len(first["train"]) == 10


def test_random_val_preserves_test_and_is_reproducible(tmp_path):
    path, payload, identities = frozen_protocol(tmp_path, count=240)
    base = {"shared_manifest_path": str(path), "use_fixed_val": False, "val_count": 50}
    first = shared_scene_split(identities, {**base, "seed": 42})
    again = shared_scene_split(identities, {**base, "seed": 42})
    second = shared_scene_split(identities, {**base, "seed": 43})
    assert first == again
    assert first["test"] == second["test"] == payload["test"]
    assert first["val"] != second["val"]
    assert len(first["val"]) == len(second["val"]) == 50
    validate_split(first)


def test_shared_protocol_rejects_missing_scene_changed_inventory_and_task_unit(tmp_path):
    path, payload, identities = frozen_protocol(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        shared_scene_split(identities[1:], {"shared_manifest_path": str(path)})
    with pytest.raises(ValueError, match="unit=scene"):
        shared_scene_split(identities, {"shared_manifest_path": str(path), "unit": "task"})
    payload["video_pairs"][identities[0]].reverse()
    write_json(path, payload)
    with pytest.raises(ValueError, match="video inventory changed"):
        load_protocol(path)


def test_task_split_keeps_entire_tasks_in_one_partition():
    scenes = [SimpleNamespace(scene_path=f"/data/task_{task:04d}/scene_{scene}")
              for task in range(20) for scene in range(1, 5)]
    class Dataset:
        def __len__(self):
            return len(self.scenes)
    dataset = Dataset()
    dataset.scenes = scenes
    config = {"unit": "task", "seed": 42, "ratios": {"train": .8, "val": .1, "test": .1}}
    mapping = build_split_manifest(split_trajectory_dataset(dataset, config))
    counts = validate_split(mapping, "task")
    assert [counts[key]["tasks"] for key in ("train", "val", "test")] == [16, 2, 2]


def test_shared_video_expands_exact_two_cameras_in_manifest_order(tmp_path):
    path, payload, identities = frozen_protocol(tmp_path)
    samples = [SimpleNamespace(sample_id=p["sample_id"], scene_id=identity,
                               task_id=identity.split('/')[0])
               for identity, pairs in payload["video_pairs"].items() for p in pairs]
    class Dataset:
        def __len__(self):
            return len(self.samples)
    dataset = Dataset()
    dataset.samples = samples
    result = video_manifest(split_video_dataset(dataset, {"shared_manifest_path": str(path)}))
    assert len(result["test"]) == 240
    assert len(result["val"]) == 100
    assert result["test"] == [p["sample_id"] for identity in payload["test"]
                              for p in payload["video_pairs"][identity]]
    assert validate_split(result)["test"]["scenes"] == 120


def test_explicit_manifest_overrides_legacy_all_test_config(tmp_path):
    path = tmp_path / "split.json"
    write_json(path, {"test": ["task_0001/scene_1"]})
    config, resolved = resolve_eval_split({"split": {"all_as_test": True}},
                                          str(tmp_path / "model.pth"), str(path), False)
    assert config == {"manifest_path": str(path)}
    assert resolved == path

def test_frozen_video_order_and_decode_failure_never_substitute(tmp_path, monkeypatch):
    from bise.data.rh20t import video_dataset
    path, payload, identities = frozen_protocol(tmp_path, with_files=True)
    monkeypatch.setattr(video_dataset, "scan_task_scenes",
                        lambda root: pytest.fail("must use frozen pairs"))
    dataset = video_dataset.RH20TVideoDataset(payload["data_root"], object(),
                                            pair_manifest_path=str(path))
    assert [x.camera_id for x in dataset.samples[:2]] == ["0", "1"]
    monkeypatch.setattr(video_dataset, "sample_video_frames", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="Cannot decode paired video sample"):
        dataset[0]
    source = Path(dataset.samples[0].human_video_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during experiment"):
        dataset[0]


def test_current_configs_control_ablations_and_share_budgets():
    root = Path(__file__).resolve().parents[1] / "configs/retrain"
    for filename in (root / "s43/run.json",):
        registry = json.loads(filename.read_text())
        assert registry["prepare"] == {"cfg2_test_limit": 98}
        assert list(registry["trajectory"]) == ["T0", "T1", "T2", "T3", "T4", "T5",
                                                  "N1", "N2", "N3", "N4"]
        for name, entry in registry["trajectory"].items():
            config = json.loads((root.parents[1] / entry["config"]).read_text())
            assert config["num_epochs"] == 65
            assert config["loss_checkpoint_threshold"] == .74
            assert config["learning_rate"] == (1e-4 if name == "T0" else 2e-4)
            assert config["lr_schedule"] == {
                "name": "cosine", "hold_epochs": 8 if name == "T0" else 10,
                "min_lr": 1e-5 if name == "T0" else 2e-5,
            }
            assert config["mode"] == ("standard" if name in ("T0", "T1") else "augment")
            assert config["model_params"]["shared_transformer"] == (
                name in ("T0", "T2", "N3", "N4"))
            assert config["model_params"]["dual_adapters"]
            assert config["model_params"]["human_input_dim"] == (
                63 if name == "T5" else 42 if name in ("N2", "N3", "N4") else 33)
            assert config["split"]["unit"] == ("task" if name == "T4" else "scene")
            assert "cfg2_test_limit" not in config
            assert "cfg2_test_limit" not in config["split"]
        assert list(registry["video"]) == ["E0", "E1", "E2", "E3", "E4", "E5",
                                             "E6", "E7", "RawJ", "RawM"]
    for name in (f"E{i}" for i in range(8)):
        config = json.loads((root / f"s43/video/{name}.json").read_text())
        assert config["training"]["batch_size"] == config["training"]["eval_batch_size"] == 4
        assert config["loss"]["name"] == ("info_nce" if name in ("E0", "E2")
                                            else "multi_positive")
        assert config["loss"]["lambda_intra"] == (0 if name in ("E0", "E1") else .2)
        assert bool(config["dataset"].get("intra_train_augmentations")) == (
            name not in ("E0", "E1"))
        assert config["model"]["encoder_mode"] == ("robot_head" if name == "E4" else "dual_head")
        assert config["model"]["freeze_backbone"] == (name != "E5")
        assert config["model"]["unfreeze_last_n_blocks"] == (2 if name == "E5" else 0)
        assert config["model"]["backbone_type"] == ("videomae" if name == "E7" else "vjepa")
        assert config["split"]["unit"] == ("task" if name == "E6" else "scene")
