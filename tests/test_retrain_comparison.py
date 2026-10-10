import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs"))

from bise.common.comparison import eval_manifest, export_comparison
from bise.common.evaluation import bind_protocols, effective_config, evaluation_inputs
from bise.common.evaluation import stamp_evaluation, validate_evaluation
from bise.common.fingerprints import write_json
from test_retrain_splits import frozen_protocol


def load_entry(name):
    spec = importlib.util.spec_from_file_location(f"retrain_{name}", ROOT / "runs" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_fixture(run, config, protocol, score=.2):
    config = effective_config(config, "trajectory", "cfg2")
    checkpoint = run / "best_model.pth"
    inputs = evaluation_inputs(config, checkpoint, "trajectory", protocol)
    output = run / "evaluations/cfg2"
    output.mkdir(parents=True, exist_ok=True)
    identities = inputs["identities"]
    metadata = {}
    for domain in ("human", "robot"):
        metadata[f"{domain}_scene_ids"] = [x[0] for x in identities]
        metadata[f"{domain}_camera_ids"] = [x[1] for x in identities]
    metrics = {direction: {level: {"R@1": score, "R@5": .3, "R@10": .4,
                                  "MRR": .25, "NDCG@10": .2, "Mean Percentage Rank": .5,
                                  "MPR-all": 50.0}
                           for level in ("task", "scene")}
               for direction in ("human_to_robot", "robot_to_human")}
    write_json(output / "metadata.json", metadata)
    write_json(output / "metrics.json", metrics)
    write_json(output / "cases.json", [])
    values = np.random.default_rng(42).normal(size=(len(identities), 4)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    np.save(output / "human_embeddings.npy", values)
    np.save(output / "robot_embeddings.npy", values)
    np.save(output / "similarity_matrix.npy", values @ values.T)
    stamp_evaluation(output, config, checkpoint, "trajectory", protocol)
    return output, config


def batch_fixture(tmp_path, keys=("T1", "N1")):
    protocol, payload, identities = frozen_protocol(tmp_path)
    groups = {}
    batch = tmp_path / "batch/trajectory"
    for key in keys:
        run = batch / key
        run.mkdir(parents=True)
        config = {"protocol": "retrain_v2", "data_root": payload["data_root"],
                  "eval_manifests": {"cfg2": str(protocol)},
                  "split": {"unit": "scene"}, "model_params": {"human_input_dim": 18},
                  "data_params": {"human_representation":
                                  "wrist_relative" if key == "N1" else "base_xyz"}}
        write_json(run / "params.json", config)
        write_json(run / "split_manifest.json", {"train": identities[170:],
                                                 "val": payload["val"], "test": payload["test"]})
        write_json(run / "data_manifest.json",
                   json.loads(Path(payload["data_manifest"]).read_text()))
        write_json(run / "curves.json", {"train_loss": [1., .5], "val_mrr": [.1, .2]})
        (run / "best_model.pth").write_bytes(f"checkpoint {key}".encode())
        bind_protocols(run, config)
        evaluation_fixture(run, config, run / "split_manifest.json")
        groups[key] = {"run_dir": str(run), "config": str(run / "params.json"), "label": key}
    registry = {"protocol": "retrain_v2", "modality": "trajectory",
                "batch": "batch", "groups": groups}
    runs = batch / "runs.json"
    write_json(runs, registry)
    return runs, registry, protocol


def test_cache_rejects_checkpoint_change(tmp_path):
    runs, registry, protocol = batch_fixture(tmp_path)
    run = Path(registry["groups"]["T1"]["run_dir"])
    config = effective_config(json.loads((run / "params.json").read_text()), "trajectory", "cfg2")
    validate_evaluation(run / "evaluations/cfg2", config, run / "best_model.pth",
                        "trajectory", run / "split_manifest.json")
    (run / "best_model.pth").write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="Stale evaluation"):
        validate_evaluation(run / "evaluations/cfg2", config, run / "best_model.pth",
                            "trajectory", run / "split_manifest.json")
    assert not (runs.parent / "charts").exists()


def test_cache_rejects_wrong_order_and_modified_metrics(tmp_path):
    runs, registry, protocol = batch_fixture(tmp_path)
    run = Path(registry["groups"]["T1"]["run_dir"])
    config = json.loads((run / "params.json").read_text())
    path = run / "evaluations/cfg2/metadata.json"
    data = json.loads(path.read_text())
    data["human_camera_ids"][:2] = reversed(data["human_camera_ids"][:2])
    write_json(path, data)
    with pytest.raises(ValueError, match="inventory/order"):
        validate_evaluation(run / "evaluations/cfg2", config, run / "best_model.pth",
                            "trajectory", run / "split_manifest.json")
    evaluation_fixture(run, json.loads((run / "params.json").read_text()),
                       run / "split_manifest.json")
    path = run / "evaluations/cfg2/metrics.json"
    data = json.loads(path.read_text())
    data["human_to_robot"]["task"]["R@1"] = .9
    write_json(path, data)
    with pytest.raises(ValueError, match="artifacts changed"):
        validate_evaluation(run / "evaluations/cfg2", config, run / "best_model.pth",
                            "trajectory", run / "split_manifest.json")


def test_only_selected_run_is_evaluated_and_all_runs_are_merged(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    runs, registry, protocol = batch_fixture(tmp_path)
    module = load_entry("compare")
    commands = []
    untouched = Path(registry["groups"]["N1"]["run_dir"]) / "evaluations/cfg2/metrics.json"
    before = untouched.read_bytes()
    def evaluate(command, log):
        commands.append(command)
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        run = checkpoint.parent
        manifest = Path(command[command.index("--split-manifest") + 1])
        assert manifest == run / "split_manifest.json"
        evaluation_fixture(run, json.loads((run / "params.json").read_text()), manifest, score=.6)
    monkeypatch.setattr(module, "run_logged", evaluate)
    plotted_metrics = []
    original_bar = Axes.bar
    def bar(axis, *args, **kwargs):
        plotted_metrics.append(kwargs.get("label"))
        return original_bar(axis, *args, **kwargs)
    monkeypatch.setattr(Axes, "bar", bar)
    module.compare(runs, only=["T1"], cfgs=["cfg2"])
    assert len(commands) == 1
    assert untouched.read_bytes() == before
    result = json.loads((runs.parent / "charts/cfg2.json").read_text())
    assert {x["group"] for x in result["experiments"]} == {"T1", "N1"}
    assert result["experiments"][0]["metrics"]["human_to_robot"]["task"]["R@1"] == .6
    mpr_rows = [row for row in result["rows"] if row["metric"] == "MPR-all"]
    assert len(mpr_rows) == 8
    assert all(row["value"] == 50.0 for row in mpr_rows)
    assert plotted_metrics == ["R@1", "R@5", "R@10", "MRR", "NDCG@10"] * 2
    assert (runs.parent / "charts/cfg2.png").is_file()
    with (runs.parent / "charts/cfg2.csv").open(newline="") as handle:
        csv_mpr_rows = [row for row in csv.DictReader(handle) if row["metric"] == "MPR-all"]
    assert len(csv_mpr_rows) == 8


def test_task_heldout_results_join_main_chart_and_remove_old_chart(tmp_path, monkeypatch):
    from bise.common import comparison
    runs, registry, protocol = batch_fixture(tmp_path, keys=("T4", "N1"))
    run = Path(registry["groups"]["T4"]["run_dir"])
    config = json.loads((run / "params.json").read_text())
    config["split"]["unit"] = "task"
    write_json(run / "params.json", config)
    mapping = json.loads((run / "split_manifest.json").read_text())
    mapping["test"] = mapping["test"][:80]
    mapping["val"] = mapping["val"][:48]
    mapping["train"] = mapping["train"][2:]
    write_json(run / "split_manifest.json", mapping)
    evaluation_fixture(run, config, run / "split_manifest.json")
    charts = runs.parent / "charts"
    charts.mkdir()
    (charts / "cfg2_task.png").write_bytes(b"old separate chart")
    plotted = []
    original_plot = comparison._plot
    def plot(experiments, path, title):
        plotted.append([x["group"] for x in experiments])
        original_plot(experiments, path, title)
    monkeypatch.setattr(comparison, "_plot", plot)
    export_comparison(registry, ["cfg2"], charts)
    assert plotted == [["T4", "N1"]]
    assert not (charts / "cfg2_task.png").exists()
    assert (runs.parent / "charts/cfg2.png").exists()
    data = json.loads((runs.parent / "charts/cfg2.json").read_text())
    assert [x["partition"] for x in data["experiments"]] == ["task", "shared"]
    assert [x["test_scenes"] for x in data["experiments"]] == [80, 120]
    assert data["experiments"][0]["split_manifest"] == str(run / "split_manifest.json")


@pytest.mark.parametrize("other", ["missing_info", "missing_metrics", "stale"])
def test_re_evaluation_overwrites_selected_and_merges_available_metrics(tmp_path, monkeypatch, other):
    runs, registry, _ = batch_fixture(tmp_path, keys=("T1", "T2"))
    module = load_entry("compare")
    selected = Path(registry["groups"]["T2"]["run_dir"])
    unselected = Path(registry["groups"]["T1"]["run_dir"])
    # A malformed old record must not prevent replacing the evaluation.
    (selected / "evaluations/cfg2/run_info.json").write_text("old invalid JSON")
    if other == "missing_info":
        (unselected / "evaluations/cfg2/run_info.json").unlink()
    elif other == "missing_metrics":
        (unselected / "evaluations/cfg2/metrics.json").unlink()
        (unselected / "curves.json").unlink()
    else:
        (unselected / "best_model.pth").write_bytes(b"changed checkpoint")
    before = {path: path.read_bytes() for path in unselected.rglob("*") if path.is_file()}
    commands = []
    def evaluate(command, log):
        commands.append(command)
        run = Path(command[command.index("--checkpoint") + 1]).parent
        assert run == selected
        write_json(run / "evaluations/cfg2/run_info.json", {})
        evaluation_fixture(run, json.loads((run / "params.json").read_text()),
                           run / "split_manifest.json", score=.6)
    monkeypatch.setattr(module, "run_logged", evaluate)
    module.compare(runs, only=["T2"], cfgs=["cfg2"], skip_existing=True)
    assert len(commands) == 1
    result = json.loads((runs.parent / "charts/cfg2.json").read_text())
    expected = ["T2"] if other == "missing_metrics" else ["T1", "T2"]
    assert [row["group"] for row in result["experiments"]] == expected
    assert result["experiments"][-1]["metrics"]["human_to_robot"]["task"]["R@1"] == .6
    assert (runs.parent / "charts/cfg2.png").is_file()
    assert all(path.read_bytes() == value for path, value in before.items())
    dependencies = json.loads((runs.parent.parent / "plots.json").read_text())
    assert "run_info.json" not in json.dumps(dependencies)


def test_all_selected_evaluations_are_recomputed_even_with_skip_existing(tmp_path, monkeypatch):
    runs, registry, _ = batch_fixture(tmp_path)
    module = load_entry("compare")
    commands = []
    def evaluate(command, log):
        commands.append(command)
        run = Path(command[command.index("--checkpoint") + 1]).parent
        evaluation_fixture(run, json.loads((run / "params.json").read_text()),
                           run / "split_manifest.json", score=.7)
    monkeypatch.setattr(module, "run_logged", evaluate)
    module.compare(runs, cfgs=["cfg2"], skip_existing=True)
    assert len(commands) == 2
    result = json.loads((runs.parent / "charts/cfg2.json").read_text())
    assert all(row["metrics"]["human_to_robot"]["task"]["R@1"] == .7
               for row in result["experiments"])


def test_selected_evaluation_cannot_be_silently_skipped_when_outputs_are_missing(tmp_path,
                                                                              monkeypatch):
    runs, registry, _ = batch_fixture(tmp_path)
    run = Path(registry["groups"]["T1"]["run_dir"])
    shutil.rmtree(run / "evaluations")
    module = load_entry("compare")
    monkeypatch.setattr(module, "run_logged", lambda *args: None)
    with pytest.raises(RuntimeError, match="Selected evaluation trajectory T1 cfg2 did not produce"):
        module.compare(runs, only=["T1"], cfgs=["cfg2"])
    assert not (runs.parent / "charts").exists()


@pytest.mark.parametrize("skip_existing", [False, True])
def test_retrain_eval_cli_generates_missing_selected_run_for_both_cfgs(tmp_path, monkeypatch,
                                                                    capsys, skip_existing):
    runs, registry, _ = batch_fixture(tmp_path, keys=("T1", "T2"))
    selected = Path(registry["groups"]["T1"]["run_dir"])
    cfg3, _, _ = frozen_protocol(tmp_path, cfg="cfg3", count=2)
    params = json.loads((selected / "params.json").read_text())
    params["eval_manifests"]["cfg3"] = str(cfg3)
    write_json(selected / "params.json", params)
    shutil.rmtree(selected / "evaluations")
    metrics = json.loads((Path(registry["groups"]["T2"]["run_dir"])
                          / "evaluations/cfg2/metrics.json").read_text())
    source = tmp_path / "run.json"
    write_json(source, {"protocol": "retrain_v2", "seed": 42, "batch": "batch",
                        "output_root": str(tmp_path), "dataset_root": str(tmp_path),
                        "manifest_dir": str(tmp_path / "manifests"),
                        "trajectory": {key: {"config": entry["config"]}
                                       for key, entry in registry["groups"].items()}, "video": {}})
    module = load_entry("retrain")
    compare = load_entry("compare")
    monkeypatch.setitem(sys.modules, "compare", compare)
    commands = []
    def evaluate(command, log):
        commands.append(command)
        output = Path(command[command.index("--output-dir") + 1])
        assert output.parent.parent == selected
        manifest = Path(command[command.index("--split-manifest") + 1])
        scenes = json.loads(manifest.read_text())["test"]
        metadata = {f"{domain}_scene_ids": scenes for domain in ("human", "robot")}
        metadata.update({f"{domain}_camera_ids": ["0"] * len(scenes)
                         for domain in ("human", "robot")})
        write_json(output / "metrics.json", metrics)
        write_json(output / "metadata.json", metadata)
        write_json(output / "cases.json", [])
        write_json(output / "data_manifest.json", {})
        values = np.ones((len(scenes), 4), dtype=np.float32) / 2
        np.save(output / "human_embeddings.npy", values)
        np.save(output / "robot_embeddings.npy", values)
        np.save(output / "similarity_matrix.npy", values @ values.T)
    monkeypatch.setattr(compare, "run_logged", evaluate)
    argv = ["retrain", "--config", str(source), "--batch", "batch", "--modality", "trajectory",
            "--stage", "eval", "--only", "T1"]
    if skip_existing:
        argv.append("--skip-existing")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert len(commands) == 2
    for cfg in ("cfg2", "cfg3"):
        assert (selected / "evaluations" / cfg / "metrics.json").is_file()
        chart = json.loads((runs.parent / "charts" / f"{cfg}.json").read_text())
        assert "T1" in {row["group"] for row in chart["experiments"]}
    assert "Skip missing evaluation trajectory T1" not in capsys.readouterr().out


def test_cfg2_uses_run_manifest_and_cfg3_uses_matching_domain(tmp_path):
    protocol, payload, _ = frozen_protocol(tmp_path)
    cfg3, _, _ = frozen_protocol(tmp_path, cfg="cfg3", count=2)
    run = tmp_path / "run"
    for unit in ("scene", "task"):
        config = {"split": {"unit": unit},
                  "eval_manifests": {"cfg2": str(protocol), "cfg3": str(cfg3)}}
        assert eval_manifest(config, run, "cfg2") == run / "split_manifest.json"
        assert eval_manifest(config, run, "cfg3") == cfg3


def test_two_stage_epoch_budget_and_dry_run_without_files(tmp_path):
    module = load_entry("retrain")
    config = {"mode": "two_stage", "pretrain_epochs": 10, "finetune_epochs": 60}
    result = module.epoch_budget(config, "trajectory", 2)
    assert result["pretrain_epochs"] == result["finetune_epochs"] == 1
    assert module.epoch_budget(config, "trajectory", 70) == config
    with pytest.raises(ValueError, match="at least two"):
        module.epoch_budget(config, "trajectory", 1)
    run_config = json.loads((ROOT / "configs/retrain/run.json").read_text())
    run_config["output_root"] = str(tmp_path / "outputs")
    path = tmp_path / "run.json"
    write_json(path, run_config)
    subprocess.run([sys.executable, str(ROOT / "runs/retrain.py"),
                    "--config", str(path), "--dry-run"],
                   check=True, capture_output=True)
    assert not (tmp_path / "outputs").exists()
