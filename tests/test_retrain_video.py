"""Exercise video entry points with frozen pairs and a small local backbone."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.common.evaluation import effective_config, validate_evaluation
from bise.common.fingerprints import file_hash, write_json
from test_retrain_comparison import load_entry
from test_retrain_splits import frozen_protocol


class LocalBackbone(torch.nn.Module):
    output_dim = 4

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, self.output_dim)

    def encode_features(self, videos):
        return self.projection(videos.mean(dim=(1, 3, 4)))


def local_processor(clips, return_tensors):
    return {"pixel_values": torch.stack(clips)}


def local_frames(path, num_frames, **kwargs):
    # Both domains of a pair receive the same deterministic fixture frames.
    source = Path(path)
    seed = int(source.parent.parent.name.removeprefix("task_"))
    seed = seed * 10 + int(source.parent.name.removeprefix("scene_"))
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_frames, 3, 4, 4, generator=generator)


def entry(name):
    spec = importlib.util.spec_from_file_location(
        f"retrain_video_{name}", ROOT / "runs/video" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_training_evaluation_and_frozen_raw_comparison(tmp_path, monkeypatch):
    from bise.data.rh20t import video_dataset
    from bise.modalities.video import factory

    protocol, payload, _ = frozen_protocol(tmp_path, with_files=True)
    cfg3, cfg3_payload, _ = frozen_protocol(tmp_path, cfg="cfg3", count=2, with_files=True)
    monkeypatch.syspath_prepend(str(ROOT / "runs/video"))
    monkeypatch.setattr(video_dataset, "sample_video_frames", local_frames)
    monkeypatch.setattr(factory, "build_video_backbone",
                        lambda **kwargs: (local_processor, LocalBackbone()))
    config = json.loads((ROOT / "configs/retrain/video/E4.json").read_text())
    config["device"] = "cpu"
    config["dataset"].update(root_dir=payload["data_root"], num_frames=2,
                             pair_manifests={"RH20T_cfg2": str(protocol),
                                             "RH20T_cfg3": str(cfg3)})
    config["model"].update(feature_dim=4, dropout=0., unfreeze_last_n_blocks=0)
    config["training"].update(batch_size=10, eval_batch_size=32,
                              num_epochs=1, num_workers=0, amp=False)
    config["split"]["shared_manifest_path"] = str(protocol)
    config["eval_manifests"] = {"cfg2": str(protocol), "cfg3": str(cfg3)}
    source = tmp_path / "E4.json"
    write_json(source, config)
    run = tmp_path / "video/E4"
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        train = entry("train_video")
        monkeypatch.setattr(sys, "argv", ["train_video", "--config", str(source),
                                         "--run-dir", str(run)])
        train.main()
        assert json.loads((run / "status.json").read_text())["state"] == "complete"
        assert json.loads((run / "checkpoint_info.json").read_text())["validation_only"]
        assert (run / "best_model.pth").is_file()
        assert not (run / "last_model.pth").exists()
        trainable = json.loads((run / "trainable.json").read_text())
        assert trainable["encoder_mode"] == "dual_head"
        assert trainable["backbones"][0]["parameters"]["trainable"] == 0
        assert trainable["heads"]["human"]["trainable"] > 0
        assert trainable["heads"]["robot"]["trainable"] > 0

        evaluate = entry("evaluate_video")
        output = run / "evaluations/cfg2"
        monkeypatch.setattr(sys, "argv", [
            "evaluate_video", "--config", str(run / "params.json"),
            "--checkpoint", str(run / "best_model.pth"),
            "--split-manifest", str(run / "split_manifest.json"),
            "--output-dir", str(output),
        ])
        evaluate.main()
        saved = effective_config(json.loads((run / "params.json").read_text()), "video", "cfg2")
        info = validate_evaluation(output, saved, run / "best_model.pth", "video",
                                   run / "split_manifest.json")
        assert info["scenes"] == 120 and len(info["identities"]) == 240

        task_config = deepcopy(config)
        task_config["experiment_name"] = "E3"
        task_config["split"] = {"unit": "task", "seed": 42,
                                "scene_counts": {"test": 120, "val": 50}}
        task_config["loss"]["lambda_intra"] = 0.
        task_config["dataset"].pop("intra_train_augmentations")
        task_source = tmp_path / "E3.json"
        write_json(task_source, task_config)
        task_run = tmp_path / "video/E3"
        monkeypatch.setattr(sys, "argv", ["train_video", "--config", str(task_source),
                                         "--run-dir", str(task_run)])
        train.main()
        assert (task_run / "best_model.pth").is_file()
        assert not (task_run / "last_model.pth").exists()
        task_info = json.loads((task_run / "split_info.json").read_text())
        assert task_info["scene_counts"] == {"test": 120, "val": 50}
        # The fixture has four scenes per task, so 48 is the closest valid validation size.
        assert task_info["counts"]["val"]["scenes"] == 48
        assert not task_info["use_fixed_val"]
        monkeypatch.setattr(sys, "argv", [
            "evaluate_video", "--config", str(task_run / "params.json"),
            "--checkpoint", str(task_run / "best_model.pth"),
            "--output-dir", str(task_run / "evaluations/cfg2"),
        ])
        evaluate.main()

        single_config = deepcopy(config)
        single_config["experiment_name"] = "E6s"
        single_config["model"]["encoder_mode"] = "robot_head"
        single_config["split"] = {"unit": "scene", "seed": 43,
                                  "ratios": {"train": .8, "val": .1, "test": .1}}
        single_source = tmp_path / "E6s.json"
        write_json(single_source, single_config)
        single_run = tmp_path / "video/E6s"
        monkeypatch.setattr(sys, "argv", ["train_video", "--config", str(single_source),
                                         "--run-dir", str(single_run), "--split-manifest",
                                         str(run / "split_manifest.json")])
        train.main()
        assert json.loads((single_run / "split_manifest.json").read_text()) == json.loads(
            (run / "split_manifest.json").read_text()
        )
        assert (single_run / "best_model.pth").is_file()
        assert not (single_run / "last_model.pth").exists()
        single_info = json.loads((single_run / "trainable.json").read_text())
        assert single_info["heads"]["human"]["trainable"] == 0
        assert single_info["heads"]["robot"]["trainable"] > 0
        assert single_info["backbones"][0]["parameters"]["trainable"] == 0

        raw = entry("evaluate_raw_video_backbone")
        monkeypatch.setattr(raw, "build_video_backbone",
                            lambda **kwargs: (local_processor, LocalBackbone()))
        config.update(experiment_name="RawJ", raw_backbone=True)
        raw_source = tmp_path / "RawJ.json"
        write_json(raw_source, config)
        raw_run = tmp_path / "video/RawJ"
        monkeypatch.setattr(sys, "argv", [
            "evaluate_raw_video_backbone", "--config", str(raw_source),
            "--output-dir", str(raw_run),
            "--final-subdir", "evaluations/cfg2",
        ])
        raw.main()
        raw_split = json.loads((raw_run / "split_manifest.json").read_text())
        assert len(raw_split["test"]) == 240 and len(raw_split["val"]) == 100
        checkpoint_hash = file_hash(raw_run / "backbone.pth")
        embedding_hash = file_hash(raw_run / "evaluations/cfg2/human_embeddings.npy")
        raw.main()
        assert file_hash(raw_run / "backbone.pth") == checkpoint_hash
        assert file_hash(raw_run / "evaluations/cfg2/human_embeddings.npy") == embedding_hash

        registry = {"protocol": "retrain_v2", "modality": "video", "batch": "local", "groups": {
            "E4": {"run_dir": str(run)},
            "E3": {"run_dir": str(task_run)},
            "RawJ": {"run_dir": str(raw_run), "raw": True},
        }}
        runs_path = tmp_path / "video/runs.json"
        write_json(runs_path, registry)
        compare = load_entry("compare")
        commands = []
        def dispatch(command, log):
            commands.append(command)
            monkeypatch.setattr(sys, "argv", [str(part) for part in command[1:]])
            if str(command[1]).endswith("evaluate_raw_video_backbone.py"):
                raw.main()
            else:
                evaluate.main()
        monkeypatch.setattr(compare, "run_logged", dispatch)
        compare.compare(runs_path, skip_existing=True)
        # Default video comparison now evaluates cfg3 for every group, reusing cfg2.
        assert len(commands) == 3
        assert all(str(command[command.index("--data-root") + 1])
                   == cfg3_payload["data_root"] for command in commands)
        assert all(Path(command[command.index("--split-manifest") + 1]) == cfg3
                   for command in commands)
        charts = tmp_path / "video/charts"
        data = json.loads((charts / "cfg2.json").read_text())
        assert [item["queries"] for item in data["experiments"]] == [240, 240, 240]
        assert data["experiments"][1]["partition"] == "task"
        assert not (charts / "cfg2_task.png").exists()
        assert (charts / "cfg2.png").is_file() and (charts / "cfg2.csv").is_file()
        cfg3_data = json.loads((charts / "cfg3.json").read_text())
        assert [item["queries"] for item in cfg3_data["experiments"]] == [4, 4, 4]
        assert (charts / "cfg3.png").is_file() and (charts / "cfg3.csv").is_file()
    finally:
        torch.set_num_threads(threads)
