"""Exercise command-line seed precedence and stochastic video inputs."""

import json
import sys

import numpy as np
import pytest
import torch

from bise.common.fingerprints import write_json
from bise.modalities.video.frame_sampling import select_frame_indices
from bise.modalities.video.transforms import apply_video_transforms
from test_retrain_comparison import load_entry


@pytest.mark.parametrize("seed", [0, 57])
def test_pipeline_seed_overrides_training_task_split_and_video_dataset(tmp_path, seed):
    config = {"seed": 11, "manifest_dir": str(tmp_path / "manifests"),
              "dataset_root": str(tmp_path / "dataset"), "trajectory": {}, "video": {}}
    originals = {}
    for modality, name in (("trajectory", "T4"), ("video", "E6")):
        path = tmp_path / f"{name}.json"
        source = {"seed": 12, "split": {"unit": "task", "seed": 13}, "model_params": {}}
        if modality == "video":
            source["dataset"] = {"seed": 14}
        write_json(path, source)
        originals[path] = path.read_bytes()
        config[modality][name] = {"config": str(path)}
    root = tmp_path / "runs/new"
    _, runtime = load_entry("retrain").prepare_batch(
        config, root, ["trajectory", "video"], seed,
        {"trajectory": None, "video": None})
    assert runtime["seed"] == seed
    assert config["seed"] == 11
    for modality, name in (("trajectory", "T4"), ("video", "E6")):
        current = json.loads((root / modality / "configs" / f"{name}.json").read_text())
        assert current["seed"] == current["split"]["seed"] == seed
        if modality == "video":
            assert current["dataset"]["seed"] == seed
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("entry", ["retrain", "prepare_splits"])
def test_cli_seed_controls_shared_preparation_and_default_batch(tmp_path, monkeypatch, entry):
    from bise.data.rh20t import splits

    config = {"protocol": "retrain_v2", "seed": 11, "batch": "original",
              "manifest_dir": str(tmp_path / "manifests"),
              "output_root": str(tmp_path / "runs"), "dataset_root": str(tmp_path / "dataset"),
              "trajectory": {}, "video": {}}
    path = tmp_path / "run.json"
    write_json(path, config)
    seen = []
    monkeypatch.setattr(splits, "prepare_protocols", lambda *args: seen.append(args) or {})
    module = load_entry(entry)
    argv = [entry, "--config", str(path), "--seed", "0"]
    if entry == "retrain":
        argv += ["--stage", "prepare", "--modality", "video"]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert len(seen) == 1
    assert str(seen[0][2]) == str(tmp_path / "manifests/s0")
    assert seen[0][3] == 0
    assert json.loads(path.read_text()) == config


def test_unseeded_video_augmentations_follow_global_seed_and_produce_distinct_views():
    clip = torch.linspace(-1, 1, 192).reshape(4, 3, 4, 4)
    config = {"brightness_jitter": .1, "noise_std": .05, "temporal_roll_max": 1}
    torch.manual_seed(7)
    first = apply_video_transforms(clip, config)
    second = apply_video_transforms(clip, config)
    assert not torch.equal(first, second)
    torch.manual_seed(7)
    torch.testing.assert_close(first, apply_video_transforms(clip, config))
    torch.testing.assert_close(second, apply_video_transforms(clip, config))
    torch.manual_seed(8)
    assert not torch.equal(first, apply_video_transforms(clip, config))


def test_explicit_video_augmentation_seed_overrides_global_seed():
    clip = torch.zeros(4, 3, 4, 4)
    torch.manual_seed(7)
    first = apply_video_transforms(clip, {"noise_std": .05}, seed=57)
    torch.manual_seed(8)
    torch.testing.assert_close(first, apply_video_transforms(clip, {"noise_std": .05}, seed=57))


def test_random_video_sampling_follows_global_seed_or_explicit_override():
    np.random.seed(7)
    first = select_frame_indices(120, 8, "segment_random")
    second = select_frame_indices(120, 8, "segment_random")
    assert not np.array_equal(first, second)
    np.random.seed(7)
    np.testing.assert_array_equal(first, select_frame_indices(120, 8, "segment_random"))
    np.testing.assert_array_equal(second, select_frame_indices(120, 8, "segment_random"))
    explicit = select_frame_indices(120, 8, "segment_random", seed=57)
    np.random.seed(8)
    np.testing.assert_array_equal(explicit, select_frame_indices(120, 8, "segment_random", seed=57))
