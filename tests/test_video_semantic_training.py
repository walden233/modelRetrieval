"""Check train-only teachers, small-batch losses, and the three video entry points."""

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.modalities.video.sampling import SceneViewBatchSampler
from bise.modalities.video.semantic import SemanticSupervisor, prepare_semantic_supervision


def metadata_dataset(tasks):
    samples = []
    for task, count in enumerate(tasks):
        for scene in range(count):
            for camera in ("a", "b"):
                samples.append(SimpleNamespace(task_id=f"task_{task:04d}",
                                               scene_id=f"task_{task:04d}/scene_{scene}",
                                               camera_id=camera))
    return SimpleNamespace(samples=samples)


def test_sampler_uses_every_train_camera_once_and_changes_by_epoch():
    source = metadata_dataset([4, 3, 2, 1])
    sampler = SceneViewBatchSampler(source, seed=13)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 5
    assert sorted(i for batch in batches for i in batch) == list(range(20))
    assert list(SceneViewBatchSampler(source, 13)) == batches
    for batch in batches:
        samples = [source.samples[i] for i in batch]
        assert len({s.task_id for s in samples}) == 2
        assert len({s.scene_id for s in samples}) == 2
        for scene in {s.scene_id for s in samples}:
            assert {s.camera_id for s in samples if s.scene_id == scene} == {"a", "b"}
    sampler.set_epoch(1)
    assert list(sampler) != batches
    # Subset indices must stay local: no train/validation leakage through source indices.
    subset = Subset(source, list(range(8, 20)))
    assert sorted(i for b in SceneViewBatchSampler(subset) for i in b) == list(range(12))


def test_sampler_rejects_unusable_scene_groups():
    with pytest.raises(ValueError, match="even"):
        SceneViewBatchSampler(metadata_dataset([2, 1]))
    with pytest.raises(ValueError, match="different task"):
        SceneViewBatchSampler(metadata_dataset([3, 1]))
    source = metadata_dataset([1, 1])
    source.samples[0].camera_id = "b"
    with pytest.raises(ValueError, match="two cameras"):
        SceneViewBatchSampler(source)


def test_batch_distillation_matches_bidirectional_kl_and_masks_missing():
    teacher = SemanticSupervisor(["a", "b"], np.eye(2),
                                 {"mode": "batch", "text_temperature": .3,
                                  "video_temperature": .5})
    h = F.normalize(torch.randn(5, 3), dim=-1).requires_grad_()
    r = F.normalize(torch.randn(5, 3), dim=-1).requires_grad_()
    keys = ["a", "a", "b", "b", "missing"]
    loss, valid, entropy, queries = teacher(h, r, keys)
    text = teacher.vectors[torch.tensor([0, 0, 1, 1])]
    target = text @ text.T / .3
    logits = h[:4] @ r[:4].T / .5
    expected = sum(F.kl_div(F.log_softmax(p, -1), F.softmax(q, -1),
                            reduction="batchmean") for q, p in
                   ((target, logits), (target.T, logits.T))) / 2
    assert torch.allclose(loss, expected)
    assert valid == queries == 4 and entropy > 0
    loss.backward()
    assert h.grad[:4].abs().sum() > 0 and r.grad[:4].abs().sum() > 0
    assert h.grad[4].abs().sum() == r.grad[4].abs().sum() == 0
    loss, valid, _, queries = teacher(h, r, ["a"] * 4 + ["missing"])
    assert loss.item() == 0 and valid == 4 and queries == 0


def test_bank_distillation_uses_external_train_anchors_and_shared_projection():
    teacher = SemanticSupervisor(["a", "b", "c"], np.eye(3),
                                 {"mode": "bank", "warmup_epochs": 1, "weight": .2})
    h = F.normalize(torch.randn(2, 2), dim=-1).requires_grad_()
    r = F.normalize(torch.randn(2, 2), dim=-1).requires_grad_()
    projector = torch.nn.Linear(2, 3, bias=False)
    loss, valid, entropy, queries = teacher(h, r, ["a", "a"], projector)
    assert valid == queries == 2 and entropy > 0
    # Unlike batch-only relations, one current scene still has a full anchor distribution.
    loss.backward()
    assert all(x.grad is not None and x.grad.abs().sum() > 0
               for x in (h, r, projector.weight))
    assert teacher.vectors.grad is None
    assert teacher.loss_weight(.25) == pytest.approx(.05)
    assert teacher.loss_weight(2) == pytest.approx(.2)


def test_queued_relations_match_kl_and_backpropagate_only_through_current_videos():
    teacher = SemanticSupervisor(["a", "b", "c"], np.eye(3),
                                 {"mode": "queue", "queue_size": 3,
                                  "text_temperature": .3, "video_temperature": .5})
    old_h = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
    old_r = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
    loss, _, _, queries = teacher(old_h, old_r, ["a", "a"])
    assert loss.item() == 0 and queries == 0
    assert teacher.queue_indices.tolist() == [0, 0]
    h = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
    r = F.normalize(torch.randn(2, 4), dim=-1).requires_grad_()
    loss, valid, entropy, queries = teacher(h, r, ["b", "missing"])
    target = teacher.vectors[1:2] @ teacher.vectors[[1, 0, 0]].T / .3
    logits = (h[:1] @ torch.cat((r[:1], old_r.detach())).T / .5,
              r[:1] @ torch.cat((h[:1], old_h.detach())).T / .5)
    expected = sum(F.kl_div(F.log_softmax(p, -1), F.softmax(target, -1),
                            reduction="batchmean") for p in logits) / 2
    torch.testing.assert_close(loss, expected)
    assert valid == queries == 1 and entropy > 0 and teacher.candidate_count == 3
    loss.backward()
    assert h.grad[0].abs().sum() > 0 and r.grad[0].abs().sum() > 0
    assert h.grad[1].abs().sum() == r.grad[1].abs().sum() == 0
    assert old_h.grad is old_r.grad is None
    assert teacher.queue_human.grad_fn is teacher.queue_robot.grad_fn is None
    teacher(h, r, ["c", "missing"])
    assert teacher.queue_indices.tolist() == [0, 1, 2]
    before = teacher.queue_indices.clone()
    teacher(h, r, ["missing", "missing"])
    torch.testing.assert_close(teacher.queue_indices, before)
    assert teacher.state_dict() == {}
    teacher.reset_queue()
    assert not teacher.queue_indices.numel() and not teacher.queue_human.numel()
    loss, _, _, queries = teacher(h, r, ["b", "b"])
    assert loss.item() == 0 and queries == 0
    with pytest.raises(ValueError, match="queue_size"):
        SemanticSupervisor(["a", "b"], np.eye(2), {"mode": "queue", "queue_size": 0})


def test_local_views_use_independent_semantic_and_view_ablations():
    from bise.modalities.video.sampling import build_training_sampler

    configs = {name: json.loads((ROOT / f"configs/retrain/views/{name}.json").read_text())
               for name in ("V0", "V1", "V2")}
    dataset = metadata_dataset([3, 2, 2, 1, 1])
    for name, config in configs.items():
        assert config["training"]["batch_size"] == 8
        assert config["training"]["eval_batch_size"] == 8
        assert config["model"]["unfreeze_last_n_blocks"] == 1
        assert not config["model"].get("semantic_dim")
        sampler = build_training_sampler(dataset, config["training"], 44)
        if name == "V1":
            assert sampler is None
            assert config["loss"]["scene_positive_weight"] == .5
        else:
            assert config["loss"]["scene_positive_weight"] == 1
            for batch in sampler:
                assert len(batch) == len(set(batch)) == 8
                samples = [dataset.samples[i] for i in batch]
                assert len({s.task_id for s in samples}) == 4
                for task in {s.task_id for s in samples}:
                    pair = [s for s in samples if s.task_id == task]
                    assert len({s.scene_id for s in pair}) == 1
                    assert {s.camera_id for s in pair} == {"a", "b"}
    assert configs["V1"]["loss"]["semantic"] == configs["V2"]["loss"]["semantic"]
    assert configs["V1"]["loss"]["semantic"]["mode"] == "queue"
    assert configs["V1"]["loss"]["semantic"]["queue_size"] == 64
    assert "semantic" not in configs["V0"]["loss"]


def test_teacher_snapshot_filters_null_and_held_out_scenes(tmp_path):
    dataset = metadata_dataset([1, 1, 1])
    ids = [f"cfg2/{sample.scene_id}" for sample in dataset.samples[::2]]
    source = tmp_path / "all.npz"
    np.savez(source, scene_ids=ids + ["cfg2/task_test/scene_1"],
             vectors=np.eye(4), texts=["pick", "place", "null", "test description"],
             usable=[True, True, False, True])
    config = {"dataset": {"root_dir": "RH20T_cfg2"}, "model": {},
              "loss": {"semantic": {"mode": "batch", "teacher_path": str(source)}}}
    supervisor = prepare_semantic_supervision(config, dataset, tmp_path / "run")
    assert set(supervisor.indices) == set(ids[:2])
    settings = config["loss"]["semantic"]
    assert settings["coverage"]["missing_scenes"] == [ids[2]]
    with np.load(settings["teacher_path"], allow_pickle=False) as d:
        assert set(d["scene_ids"]) == set(ids[:2])
        assert "test description" not in d["texts"]
    # Loading the immutable per-run snapshot must not silently return to the global pool.
    source.unlink()
    again = prepare_semantic_supervision(config, dataset, tmp_path / "run")
    assert set(again.indices) == set(supervisor.indices)


def test_auxiliary_head_preserves_e5_initialization_and_retrieval():
    from bise.modalities.video.models.cross_domain_video_encoder import CrossDomainVideoEncoder
    from test_retrain_video import LocalBackbone

    models, states = [], []
    for semantic_dim in (0, 6):
        torch.manual_seed(44)
        model = CrossDomainVideoEncoder(LocalBackbone(), feature_dim=4, encoder_mode="dual_head",
                                        dropout=0, semantic_dim=semantic_dim)
        models.append(model)
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])
    first = models[0].state_dict()
    assert all(torch.equal(value, models[1].state_dict()[key]) for key, value in first.items())
    videos = torch.randn(2, 2, 3, 4, 4)
    assert torch.allclose(models[0].encode_human(videos), models[1].encode_human(videos))


@pytest.mark.parametrize("suite", ("views", "video32"))
def test_variants_prepare_train_and_evaluate_both_cfgs(tmp_path, monkeypatch, suite):
    from test_retrain_splits import frozen_protocol
    from test_retrain_video import entry, LocalBackbone, local_processor, local_frames
    from bise.data.rh20t import video_dataset
    from bise.modalities.video import factory
    from bise.common.training import prepare_training_run
    from bise.common.evaluation import effective_config, validate_evaluation

    scene_count = 218 if suite == "video32" else 186
    protocol, payload, identities = frozen_protocol(tmp_path, count=scene_count, with_files=True)
    cfg3, _, _ = frozen_protocol(tmp_path, cfg="cfg3", count=2, with_files=True)
    monkeypatch.syspath_prepend(str(ROOT / "runs/video"))
    monkeypatch.setattr(video_dataset, "sample_video_frames", local_frames)
    monkeypatch.setattr(factory, "build_video_backbone",
                        lambda **kwargs: (local_processor, LocalBackbone()))
    teacher_file = tmp_path / "teacher.npz"
    rng = np.random.default_rng(44)
    np.savez(teacher_file, scene_ids=[f"cfg2/{s}" for s in identities],
             texts=["perform operation"] * len(identities),
             vectors=rng.normal(size=(len(identities), 6)).astype(np.float32))
    train = entry("train_video")
    evaluate = entry("evaluate_video")
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        splits, teachers = [], []
        names = ("V0", "V1", "V2", "V3") if suite == "views" else ("V0", "V1", "V2")
        for name in names:
            config = json.loads((ROOT / f"configs/retrain/{suite}/{name}.json").read_text())
            config["device"] = "cpu"
            config["dataset"].update(root_dir=payload["data_root"], num_frames=2,
                                     pair_manifests={"RH20T_cfg2": str(protocol),
                                                     "RH20T_cfg3": str(cfg3)})
            config["model"].update(feature_dim=4, dropout=0, unfreeze_last_n_blocks=0)
            config["training"].update(num_epochs=1, eval_batch_size=64, amp=False)
            config["split"] = {"shared_manifest_path": str(protocol), "unit": "scene"}
            config["eval_manifests"] = {"cfg2": str(protocol), "cfg3": str(cfg3)}
            if name in ("V1", "V2"):
                config["loss"]["semantic"]["teacher_path"] = str(teacher_file)
            if config["model"].get("semantic_dim"):
                config["model"]["semantic_dim"] = 6
            run = tmp_path / "runs" / "video" / name
            prepare_training_run(config, "video", run)
            # Snapshots and all metadata exist before the model is built.
            if name in ("V1", "V2"):
                with np.load(run / "semantic_teachers.npz", allow_pickle=False) as data:
                    teachers.append(set(data["scene_ids"]))
                    assert len(teachers[-1]) == scene_count - 170
            monkeypatch.setattr(sys, "argv", ["train_video", "--config", str(run / "params.json"),
                                             "--run-dir", str(run), "--prepared"])
            train.main()
            splits.append((run / "split_manifest.json").read_bytes())
            assert json.loads((run / "status.json").read_text())["state"] == "complete"
            assert (run / "best_model.pth").is_file() and not (run / "last_model.pth").exists()
            curves = json.loads((run / "curves.json").read_text())
            if name in ("V1", "V2"):
                assert curves["train_loss_semantic"][0] >= -1e-6
                assert curves["semantic_coverage"] == [1.0]
                if suite == "views":
                    assert curves["semantic_candidates"][0] > 8
                    assert curves["semantic_weighted_loss"][0] > 0
                    assert curves["semantic_gradient_ratio"][0] > 0
            if name == "V3":
                assert not (run / "semantic_teachers.npz").exists()
                assert curves["train_loss_negative"][0] > 0
                assert curves["negative_weighted_loss"][0] == pytest.approx(
                    .1 * curves["train_loss_negative"][0])
            saved = json.loads((run / "params.json").read_text())
            for cfg in ("cfg2", "cfg3"):
                current = effective_config(deepcopy(saved), "video", cfg)
                source = tmp_path / f"{name}_{cfg}.json"
                source.write_text(json.dumps(current))
                output = run / "evaluations" / cfg
                manifest = run / "split_manifest.json" if cfg == "cfg2" else cfg3
                monkeypatch.setattr(sys, "argv", [
                    "evaluate_video", "--config", str(source),
                    "--checkpoint", str(run / "best_model.pth"),
                    "--split-manifest", str(manifest), "--output-dir", str(output)])
                evaluate.main()
                validate_evaluation(output, current, run / "best_model.pth", "video", manifest)
                for artifact in ("metrics.json", "views.json", "views.png",
                                 "camera_heatmap.png", "scene_heatmap.png"):
                    assert (output / artifact).is_file()
        assert all(split == splits[0] for split in splits)
        assert teachers[0] == teachers[1]
        if suite == "views":
            from bise.common.comparison import export_comparison
            registry = {"modality": "video", "batch": "test", "groups": {
                name: {"run_dir": str(tmp_path / "runs/video" / name), "label": name}
                for name in names}}
            summary = export_comparison(registry, ["cfg2", "cfg3"], tmp_path / "charts")
            assert all(set(summary[cfg]) == set(names) for cfg in ("cfg2", "cfg3"))
    finally:
        torch.set_num_threads(threads)
