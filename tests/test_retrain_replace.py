"""Replacing selected runs preserves unrelated experiments and removes stale charts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs"))

from bise.common.fingerprints import write_json
from test_retrain_comparison import load_entry


def prepare_metadata(config, modality, run, split_manifest=None):
    mapping = (json.loads(split_manifest.read_text())
               if split_manifest is not None and split_manifest.is_file()
               else {"train": [], "val": [], "test": []})
    write_json(run / "split_manifest.json", mapping)
    config["split"]["manifest_path"] = str(run / "split_manifest.json")
    write_json(run / "params.json", config)
    for name in ("data_manifest", "split_info", "protocols"):
        write_json(run / f"{name}.json", {})
    write_json(run / "status.json", {"state": "prepared"})


def test_replacement_pipeline_archives_selected_runs_and_reuses_other_groups(tmp_path,
                                                                          monkeypatch):
    module = load_entry("retrain")
    config = json.loads((ROOT / "configs/retrain/run.json").read_text())
    config["output_root"] = str(tmp_path / "retrain")
    config["trajectory"] = {key: config["trajectory"][key] for key in ("T1", "T4")}
    config["video"] = {key: config["video"][key] for key in ("E1", "E6")}
    for modality in ("trajectory", "video"):
        for key, entry in config[modality].items():
            source = json.loads((ROOT / entry["config"]).read_text())
            source["device"] = "cpu"
            if key in ("T4", "E6"):
                source["split"] = {"unit": "task", "seed": 42,
                                   "ratios": {"train": .8, "val": .1, "test": .1}}
            path = tmp_path / "source" / f"{key}.json"
            write_json(path, source)
            entry["config"] = str(path)
    root = tmp_path / "retrain/s42"
    module.prepare_batch(config, root, ["trajectory", "video"], 42,
                         {"trajectory": None, "video": None})
    plots, untouched = {}, {}
    for modality in ("trajectory", "video"):
        for key, entry in config[modality].items():
            run = root / modality / key
            saved = json.loads(Path(entry["config"]).read_text())
            saved["split"]["manifest_path"] = str(run / "split_manifest.json")
            write_json(run / "params.json", saved)
            write_json(run / "status.json", {"state": "complete"})
            write_json(run / "curves.json", {"train_loss": [1.]})
            (run / "best_model.pth").write_bytes(f"old {key}".encode())
            (run / "curves.png").write_bytes(b"old curve")
            relative = f"{modality}/{key}/curves.png"
            plots[relative] = {"kind": "run_curves", "inputs": [relative.replace("png", "json")]}
            write_json(run / "evaluations/cfg2/metrics.json", {"score": .5})
            log = root / modality / "logs" / f"{key}_train.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_bytes(f"old log {key}".encode())
            if key not in ("T4", "E6"):
                untouched.update({path: path.read_bytes() for path in run.rglob("*")
                                  if path.is_file()})
                untouched[log] = log.read_bytes()
        charts = root / modality / "charts"
        write_json(charts / "cfg2.json", {"old": True})
        (charts / "cfg2.png").write_bytes(b"old comparison")
        write_json(root / modality / "summary.json", {"old": True})
        plots[f"{modality}/charts/cfg2.png"] = {"kind": "comparison", "inputs": []}
    write_json(root / "plots.json", plots)
    for modality, key in (("trajectory", "T4"), ("video", "E6")):
        source = Path(config[modality][key]["config"])
        current = json.loads(source.read_text())
        current["split"] = {"unit": "task", "seed": 42,
                            "scene_counts": {"test": 120, "val": 50}}
        write_json(source, current)
    config_path = tmp_path / "run.json"
    write_json(config_path, config)

    from bise.data.rh20t import splits
    monkeypatch.setattr(splits, "prepare_protocols", lambda *args: None)
    monkeypatch.setattr("bise.data.rh20t.grasp_geometry.validate_dataset_geometry",
                        lambda *args: None)
    monkeypatch.setitem(sys.modules, "audit_splits",
                        SimpleNamespace(audit=lambda *args, **kwargs: {}))
    monkeypatch.setattr("bise.common.training.prepare_training_run", prepare_metadata)
    trained, compared = [], []
    def train(command, log):
        run = Path(command[command.index("--run-dir") + 1])
        source = Path(command[command.index("--config") + 1])
        assert "--prepared" in command
        for modality, key in (("trajectory", "T4"), ("video", "E6")):
            prepared = root / modality / key
            assert (prepared / "params.json").is_file()
            assert (prepared / "split_manifest.json").is_file()
        assert json.loads((run / "status.json").read_text())["state"] == "prepared"
        params = json.loads(source.read_text())
        assert params["split"]["scene_counts"] == {"test": 120, "val": 50}
        write_json(run / "params.json", params)
        write_json(run / "status.json", {"state": "complete"})
        (run / "best_model.pth").write_bytes(b"new checkpoint")
        trained.append(run.name)
    def compare(runs, only, charts_only, skip_existing):
        assert only in (["T4"], ["E6"])
        assert not charts_only and not skip_existing
        compared.extend(only)
    monkeypatch.setattr(module, "run_logged", train)
    monkeypatch.setitem(sys.modules, "compare", SimpleNamespace(compare=compare))
    monkeypatch.setattr(sys, "argv", ["retrain", "--config", str(config_path),
                                     "--batch", "s42", "--only", "T4", "E6", "--replace"])
    module.main()
    assert trained == compared == ["T4", "E6"]
    archives = list((root.parent / "archive/s42").iterdir())
    assert len(archives) == 1
    archive = archives[0]
    for modality, key in (("trajectory", "T4"), ("video", "E6")):
        assert (archive / modality / key / "best_model.pth").read_bytes() == f"old {key}".encode()
        assert (archive / modality / "charts/cfg2.png").read_bytes() == b"old comparison"
        assert (archive / modality / "logs" / f"{key}_train.log").is_file()
        assert (archive / modality / "summary.json").is_file()
        assert not (root / modality / "charts").exists()
    assert json.loads((root / "plots.json").read_text()) == {
        path: entry for path, entry in plots.items() if path in (
            "trajectory/T1/curves.png", "video/E1/curves.png")
    }
    assert all(path.read_bytes() == content for path, content in untouched.items())
    assert all(step["state"] == "complete"
               for step in json.loads((root / "pipeline.json").read_text())["steps"])


def test_existing_batch_accepts_updated_configs_and_registry(tmp_path):
    module = load_entry("retrain")
    config = json.loads((ROOT / "configs/retrain/run.json").read_text())
    config["trajectory"] = {key: config["trajectory"][key] for key in ("T1", "T4")}
    root = tmp_path / "s42"
    budgets = {"trajectory": None, "video": None}
    module.prepare_batch(config, root, ["trajectory"], 42, budgets)
    destination = root / "trajectory/configs/T1.json"
    saved = json.loads(destination.read_text())
    saved["learning_rate"] *= 2
    write_json(destination, saved)
    run = root / "trajectory/T1"
    write_json(run / "params.json", saved)
    mapping = {"train": ["task_0001/scene_1"], "val": [], "test": ["task_0002/scene_1"]}
    write_json(run / "split_manifest.json", mapping)
    before = (run / "params.json").read_bytes()
    config["trajectory"]["T1"]["label"] = "updated"
    module.prepare_batch(config, root, ["trajectory"], 42, budgets)
    updated = json.loads(destination.read_text())
    assert updated["learning_rate"] == saved["learning_rate"] / 2
    assert updated["split"]["manifest_path"] == str(run / "split_manifest.json")
    registry = json.loads((root / "trajectory/runs.json").read_text())
    assert registry["groups"]["T1"]["label"] == "updated"
    assert (run / "params.json").read_bytes() == before


@pytest.mark.parametrize("modality,key,skip,best,state,params", [
    ("trajectory", "T1", True, False, "complete", True),
    ("video", "E1", True, False, "complete", True),
    ("trajectory", "T1", True, True, "training", True),
    ("video", "E1", True, True, "training", True),
    ("trajectory", "T1", True, True, "complete", False),
    ("video", "E1", True, True, "complete", True),
    ("trajectory", "T1", False, True, "complete", True),
])
def test_same_batch_retraining_reuses_saved_split(tmp_path, monkeypatch, modality, key,
                                                skip, best, state, params):
    module = load_entry("retrain")
    config = json.loads((ROOT / "configs/retrain/run.json").read_text())
    config["output_root"] = str(tmp_path / "retrain")
    entry = config[modality][key]
    source = json.loads((ROOT / entry["config"]).read_text())
    source["device"] = "cpu"
    source_path = tmp_path / "source.json"
    write_json(source_path, source)
    config["trajectory"] = {}
    config["video"] = {}
    config[modality][key] = {**entry, "config": str(source_path)}
    root = tmp_path / "retrain/s42"
    module.prepare_batch(config, root, [modality], 42,
                         {"trajectory": None, "video": None})
    run = root / modality / key
    mapping = {"train": ["task_0001/scene_1"], "val": ["task_0002/scene_1"],
               "test": ["task_0003/scene_1"]}
    write_json(run / "split_manifest.json", mapping)
    write_json(run / "status.json", {"state": state})
    if params:
        write_json(run / "params.json", source)
    if best:
        (run / "best_model.pth").write_bytes(b"old checkpoint")
    write_json(run / "evaluations/cfg2/metrics.json", {"old": True})
    write_json(root / modality / "charts/cfg2.json", {"old": True})
    write_json(root / "plots.json", {
        f"{modality}/{key}/curves.png": {"kind": "run_curves", "inputs": []},
        f"{modality}/charts/cfg2.png": {"kind": "comparison", "inputs": []},
    })
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*")
              if path.is_file()}
    if modality == "trajectory":
        source["learning_rate"] *= 2
    else:
        source["optimization"]["learning_rate"] *= 2
    write_json(source_path, source)
    config_path = tmp_path / "run.json"
    write_json(config_path, config)
    commands = []

    def train(command, log):
        commands.append(command)
        assert "--prepared" in command
        preserved = run / "split_manifest.json"
        assert json.loads(preserved.read_text()) == mapping
        current = json.loads(Path(command[command.index("--config") + 1]).read_text())
        assert current["split"]["manifest_path"] == str(run / "split_manifest.json")
        write_json(run / "params.json", current)
        write_json(run / "split_manifest.json", mapping)
        write_json(run / "status.json", {"state": "complete"})
        (run / "best_model.pth").write_bytes(b"new checkpoint")

    monkeypatch.setattr(module, "run_logged", train)
    monkeypatch.setattr("bise.common.training.prepare_training_run", prepare_metadata)
    arguments = ["retrain", "--config", str(config_path), "--batch", "s42",
                 "--stage", "train", "--modality", modality]
    if skip:
        arguments.append("--skip-existing")
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    if skip and best and state == "complete" and params:
        assert commands == []
        assert all((run / path).read_bytes() == value for path, value in before.items())
        assert not (root.parent / "archive").exists()
    else:
        assert len(commands) == 1
        backups = list((root.parent / "archive/s42").iterdir())
        assert len(backups) == 1
        archived = backups[0] / modality / key
        assert all((archived / path).read_bytes() == value for path, value in before.items())
        assert (run / "best_model.pth").read_bytes() == b"new checkpoint"
        assert not (run / "evaluations").exists()
        assert not (root / modality / "charts").exists()
        assert json.loads((root / "plots.json").read_text()) == {}


@pytest.mark.parametrize("arguments", [["--replace"],
                                       ["--replace", "--only", "T4", "--stage", "eval"],
                                       ["--replace", "--only", "T4", "--skip-existing"],
                                       ["--replace", "--only", "RawJ"]])
def test_replace_rejects_ambiguous_scope_and_raw_training(arguments, monkeypatch):
    module = load_entry("retrain")
    monkeypatch.setattr(sys, "argv", ["retrain", *arguments])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("fail", [False, True])
def test_all_metadata_precedes_training_and_failure_stops_training(tmp_path, monkeypatch, fail):
    module = load_entry("retrain")
    config = json.loads((ROOT / "configs/retrain/run.json").read_text())
    config["output_root"] = str(tmp_path / "runs")
    config["trajectory"] = {key: config["trajectory"][key] for key in ("T1", "T4")}
    config["video"] = {key: config["video"][key] for key in ("E1", "RawJ")}
    for modality in ("trajectory", "video"):
        for key, entry in config[modality].items():
            current = json.loads((ROOT / entry["config"]).read_text())
            current["device"] = "cpu"
            path = tmp_path / f"{key}.json"
            write_json(path, current)
            entry["config"] = str(path)
    source = tmp_path / "run.json"
    write_json(source, config)
    prepared, trained = [], []
    def prepare(current, modality, run, split_manifest):
        prepared.append(run.name)
        if fail and run.name == "T4":
            raise RuntimeError("Cannot prepare T4 split")
        prepare_metadata(current, modality, run, split_manifest)
    root = tmp_path / "runs/test"
    def train(command, log):
        assert prepared == ["T1", "T4", "E1", "RawJ"]
        for modality in ("trajectory", "video"):
            for key in config[modality]:
                for name in ("params", "split_manifest", "data_manifest",
                             "split_info", "protocols", "status"):
                    assert (root / modality / key / f"{name}.json").is_file()
        assert json.loads((root / "video/RawJ/params.json").read_text())["raw_backbone"]
        run = Path(command[command.index("--run-dir") + 1])
        trained.append(run.name)
        write_json(run / "status.json", {"state": "complete"})
    monkeypatch.setattr("bise.common.training.prepare_training_run", prepare)
    monkeypatch.setattr(module, "run_logged", train)
    monkeypatch.setattr(sys, "argv", ["retrain", "--config", str(source), "--batch", "test",
                                     "--stage", "train"])
    if fail:
        with pytest.raises(RuntimeError, match="Cannot prepare T4 split"):
            module.main()
        assert prepared == ["T1", "T4"] and trained == []
    else:
        module.main()
        assert trained == ["T1", "T4", "E1"]
    assert json.loads((root / "pipeline.json").read_text())["steps"][-1]["state"] == (
        "failed" if fail else "complete")
