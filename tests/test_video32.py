"""Verify grouped task batches, clean split preparation and actual VideoMAE fine-tuning."""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import Subset

from bise.data.rh20t.splits import validate_split
from bise.modalities.video.sampling import (
    ScenePairBatchSampler, TaskPairBatchSampler, build_training_sampler,
)
from test_video_semantic_training import metadata_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_scene_pairs_keep_both_cameras_and_distinct_tasks_covering_train_subset():
    source = metadata_dataset([2, 3, 2] + [1] * 10)
    subset = Subset(source, list(range(4, len(source.samples))))
    sampler = ScenePairBatchSampler(subset, batch_size=22, seed=44)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3
    assert {i for batch in batches for i in batch} == set(range(len(subset)))
    for batch in batches:
        assert len(batch) == len(set(batch)) == 22
        samples = [source.samples[subset.indices[i]] for i in batch]
        assert len({sample.task_id for sample in samples}) == 11
        for task in {sample.task_id for sample in samples}:
            pair = [sample for sample in samples if sample.task_id == task]
            assert len(pair) == 2 and len({sample.scene_id for sample in pair}) == 1
            assert {sample.camera_id for sample in pair} == {"a", "b"}
    assert list(ScenePairBatchSampler(subset, 22, 44)) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_scene_pairs_are_opt_in_and_reject_incomplete_views():
    source = metadata_dataset([2, 1, 1])
    assert build_training_sampler(source, {}, 44) is None
    assert build_training_sampler(source, {"sampler": "random"}, 44) is None
    settings = {"sampler": "scene_pairs", "batch_size": 4}
    assert isinstance(build_training_sampler(source, settings, 44), ScenePairBatchSampler)
    source.samples[0].camera_id = "b"
    with pytest.raises(ValueError, match="exactly two cameras"):
        build_training_sampler(source, settings, 44)
    source = metadata_dataset([2, 1, 1])
    with pytest.raises(ValueError, match="exactly two cameras"):
        ScenePairBatchSampler(Subset(source, [0, 1, 2, 4, 5, 6, 7]), 4)
    with pytest.raises(ValueError, match="batch_size=4"):
        build_training_sampler(source, {"sampler": "scene_views", "batch_size": 22}, 44)


def test_task_pairs_cover_train_samples_and_pad_without_repeating_task_within_batch():
    source = metadata_dataset([5, 1, 2, 1])
    sampler = TaskPairBatchSampler(source, batch_size=6, seed=44)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 5
    assert {i for batch in batches for i in batch} == set(range(len(source.samples)))
    for batch in batches:
        assert len(batch) == len(set(batch)) == 6
        assert sorted(Counter(source.samples[i].task_id for i in batch).values()) == [2, 2, 2]
    assert list(TaskPairBatchSampler(source, 6, 44)) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_task_pairs_prefer_other_scenes_and_only_use_subset_indices():
    source = metadata_dataset([3, 3, 3, 3])
    subset = Subset(source, list(range(12, 24)))
    sampler = TaskPairBatchSampler(subset, batch_size=4)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(12))
    for batch in batches:
        samples = [source.samples[subset.indices[i]] for i in batch]
        for task in {sample.task_id for sample in samples}:
            assert len({sample.scene_id for sample in samples if sample.task_id == task}) == 2
    single_scene = metadata_dataset([1] * 11)
    batches = list(TaskPairBatchSampler(single_scene, 22))
    assert len(batches) == 1 and set(batches[0]) == set(range(22))


def test_task_pairs_reject_invalid_budget_and_handle_odd_task_inventory():
    source = metadata_dataset([1, 1])
    with pytest.raises(ValueError, match="even"):
        TaskPairBatchSampler(source, 5)
    with pytest.raises(ValueError, match="distinct training tasks"):
        TaskPairBatchSampler(source, 22)
    source.samples.pop()
    with pytest.raises(ValueError, match="two distinct samples"):
        TaskPairBatchSampler(source, 4)
    source = metadata_dataset([2, 2])
    subset = Subset(source, [0, 1, 2, 4, 5, 6])
    batches = list(TaskPairBatchSampler(subset, 4))
    assert {i for batch in batches for i in batch} == set(range(6))
    assert all(len(set(batch)) == 4 for batch in batches)


def test_strict_split_checks_scene_and_task_overlap_without_changing_legacy_default():
    mapping = {"train": ["rh20t::task_0001::scene_1::a"],
               "val": [], "test": ["rh20t::task_0001::scene_1::b"]}
    validate_split(mapping)
    with pytest.raises(ValueError, match="Overlapping scenes"):
        validate_split(mapping, require_disjoint=True)
    mapping["test"] = ["rh20t::task_0001::scene_2::a"]
    validate_split(mapping, require_disjoint=True)
    with pytest.raises(ValueError, match="Overlapping tasks"):
        validate_split(mapping, "task", require_disjoint=True)


def test_video32_videomae_trains_last_two_blocks_and_evaluates_both_cfgs(tmp_path, monkeypatch):
    from transformers import VideoMAEConfig, VideoMAEModel
    from bise.data.rh20t import video_dataset
    from bise.common.training import prepare_training_run
    from bise.common.evaluation import effective_config, validate_evaluation
    from bise.modalities.video import factory
    from bise.modalities.video.models.videomae_adapter import VideoMAEAdapter
    from test_retrain_splits import frozen_protocol
    from test_retrain_video import entry, local_frames, local_processor

    protocol, payload, _ = frozen_protocol(tmp_path, with_files=True)
    cfg3, _, _ = frozen_protocol(tmp_path, cfg="cfg3", count=2, with_files=True)
    monkeypatch.syspath_prepend(str(ROOT / "runs/video"))
    monkeypatch.setattr(video_dataset, "sample_video_frames", local_frames)
    created = []

    def build_backbone(**kwargs):
        assert kwargs["backbone_type"] == "videomae"
        model = VideoMAEModel(VideoMAEConfig(
            image_size=4, patch_size=2, num_frames=2, tubelet_size=1,
            hidden_size=8, intermediate_size=16, num_hidden_layers=3,
            num_attention_heads=2,
        ))
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        created.append((model, initial))
        return local_processor, VideoMAEAdapter(model, hidden_size=8)

    monkeypatch.setattr(factory, "build_video_backbone", build_backbone)
    config = json.loads((ROOT / "configs/retrain/video32/E7.json").read_text())
    config["device"] = "cpu"
    config["dataset"].update(root_dir=payload["data_root"], num_frames=2,
                             pair_manifests={"RH20T_cfg2": str(protocol),
                                             "RH20T_cfg3": str(cfg3)})
    config["model"].update(feature_dim=4, dropout=0)
    config["training"].update(batch_size=4, eval_batch_size=64, num_epochs=1, amp=False)
    config["split"]["shared_manifest_path"] = str(protocol)
    config["eval_manifests"] = {"cfg2": str(protocol), "cfg3": str(cfg3)}
    run = tmp_path / "batch/video/E7"
    prepare_training_run(config, "video", run)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        train = entry("train_video")
        monkeypatch.setattr(sys, "argv", ["train_video", "--config", str(run / "params.json"),
                                         "--run-dir", str(run), "--prepared"])
        train.main()
        trained, initial = created[0]
        assert all(torch.equal(value, initial[key]) for key, value in trained.state_dict().items()
                   if key.startswith("encoder.layer.0."))
        for layer in (1, 2):
            assert any(not torch.equal(value, initial[key])
                       for key, value in trained.state_dict().items()
                       if key.startswith(f"encoder.layer.{layer}."))
        info = json.loads((run / "trainable.json").read_text())
        assert info["backbones"][0]["trainable_blocks"] == [2, 3]
        assert (run / "best_model.pth").exists() and not (run / "last_model.pth").exists()
        saved = json.loads((run / "params.json").read_text())
        evaluate = entry("evaluate_video")
        for cfg in ("cfg2", "cfg3"):
            current = effective_config(saved, "video", cfg)
            source = tmp_path / f"{cfg}.json"
            source.write_text(json.dumps(current))
            manifest = run / "split_manifest.json" if cfg == "cfg2" else cfg3
            output = run / "evaluations" / cfg
            monkeypatch.setattr(sys, "argv", [
                "evaluate_video", "--config", str(source),
                "--checkpoint", str(run / "best_model.pth"),
                "--split-manifest", str(manifest), "--output-dir", str(output)])
            evaluate.main()
            validate_evaluation(output, current, run / "best_model.pth", "video", manifest)
    finally:
        torch.set_num_threads(threads)
