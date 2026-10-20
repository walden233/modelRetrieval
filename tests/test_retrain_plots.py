import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.common.comparison import METRICS, export_curves
from bise.common.fingerprints import file_hash, write_json
from bise.common.plots import index_plots, record_plot, redraw_plots
from bise.common.run_artifacts import save_run_artifacts
from bise.modalities.video.figures import (
    keep_first_camera_per_scene, plot_similarity_heatmap, plot_sorted_similarity_heatmap,
)


def plot_batch(tmp_path):
    root = tmp_path / "retrain" / "s42"
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    metadata = {
        "sample_ids": ["b/1", "b/2", "a/1", "a/2"],
        "scene_ids": ["b", "b", "a", "a"],
        "task_ids": ["task2", "task2", "task1", "task1"],
        "camera_ids": ["1", "2", "1", "2"],
    }
    history = {"train_loss": [1., None, .3], "val_mrr": [None, .2, .4]}
    metrics = {"human_to_robot": {level: {metric: .5 for metric in METRICS}
                                  for level in ("task", "scene")}}
    for modality, group in (("trajectory", "T4"), ("video", "E3")):
        directory = root / modality
        run = directory / group
        registry = {"modality": modality, "batch": "s42", "groups": {
            group: {"run_dir": str(run), "label": f"{group} label"},
        }}
        write_json(directory / "runs.json", registry)
        write_json(run / "curves.json", history)
        write_json(directory / "charts" / "curves.json", {group: history})
        for cfg in ("cfg2", "cfg3"):
            evaluation = run / "evaluations" / cfg
            write_json(evaluation / "metrics.json", metrics)
            write_json(evaluation / "run_info.json", {"scenes": 2})
            stored = metadata if modality == "video" else {
                f"human_{key}": value for key, value in metadata.items() if key != "sample_ids"
            }
            write_json(evaluation / "metadata.json", stored)
            np.save(evaluation / "similarity_matrix.npy", matrix)
            write_json(directory / "charts" / f"{cfg}.json", {
                "cfg": cfg, "modality": modality, "batch": "s42", "experiments": [{
                    "group": group, "label": f"{group} label", "test_scenes": 2,
                    "partition": "task" if cfg == "cfg2" else "shared", "metrics": metrics,
                }],
            })
    return root, matrix, metadata


def test_comparison_mpr_panel_uses_percent_and_keeps_missing_values(tmp_path, monkeypatch):
    from bise.common.comparison import _plot

    experiments = [{"label": "V0", "partition": "shared", "test_scenes": 120,
                    "metrics": {"human_to_robot": {
                        "task": {**dict.fromkeys(METRICS, .5), "MPR-all": 9.5},
                        "scene": {**dict.fromkeys(METRICS, .5), "MPR-all": 8.5}}}},
                   {"label": "Legacy", "partition": "shared", "test_scenes": 120,
                    "metrics": {"human_to_robot": {
                        level: dict.fromkeys(METRICS, .5) for level in ("task", "scene")}}}]

    def savefig(figure, path, **kwargs):
        assert len(figure.axes) == 2
        assert figure.axes[0].get_ylim() == (0., 1.)
        assert figure.axes[0].get_title() == "Task retrieval"
        axis = figure.axes[1]
        assert axis.get_ylim() == (0., 100.)
        assert "lower is better" in axis.get_title()
        np.testing.assert_allclose([bar.get_height() for bar in axis.patches],
                                   [9.5, np.nan], equal_nan=True)
        assert all("Scene" not in axis.get_title() for axis in figure.axes)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    _plot(experiments, tmp_path / "cfg2.png", "Video comparison")


def test_index_records_every_source_before_drawing_and_recovers_missing_pngs(tmp_path,
                                                                           monkeypatch):
    root, matrix, metadata = plot_batch(tmp_path)
    before = {path: file_hash(path) for path in root.rglob("*") if path.is_file()}
    payload = index_plots(root)
    assert len(payload) == 16
    assert not list(root.rglob("*.png"))
    comparison = payload["video/charts/cfg2.png"]
    assert comparison["inputs"] == ["video/charts/cfg2.json"]
    assert set(comparison["upstream"]) == {
        "video/runs.json", "video/E3/evaluations/cfg2/metrics.json",
        "video/E3/evaluations/cfg2/metadata.json",
    }
    assert payload["video/E3/curves.png"] == {
        "kind": "run_curves", "inputs": ["video/E3/curves.json"],
    }
    assert "sha256" not in (root / "plots.json").read_text()
    for entry in payload.values():
        for dependency in entry["inputs"] + entry.get("upstream", []):
            assert not Path(dependency).is_absolute()
            assert (root / dependency).is_file()

    observed = []
    original_savefig = matplotlib.figure.Figure.savefig
    def savefig(figure, path, **kwargs):
        manifest = json.loads((root / "plots.json").read_text())
        assert len(manifest) == 16
        assert path.relative_to(root).as_posix() in manifest
        assert "sha256" not in json.dumps(manifest)
        kwargs["dpi"] = 20
        original_savefig(figure, path, **kwargs)
        observed.append(path)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)

    from bise.modalities.video import figures
    original_heatmap = figures.plot_similarity_heatmap
    def heatmap(values, path, **kwargs):
        np.testing.assert_array_equal(values, matrix[np.ix_([0, 2], [0, 2])])
        original_heatmap(values, path, **kwargs)
    monkeypatch.setattr(figures, "plot_similarity_heatmap", heatmap)
    redraw_plots(root)
    assert len(observed) == 16
    assert len(list(root.rglob("*.png"))) == 16
    assert all(file_hash(path) == digest for path, digest in before.items())
    assert not list(root.rglob("*.pth"))
    assert not list(root.rglob("params.json"))

    (root / "video/charts/cfg2.png").unlink()
    assert "video/charts/cfg2.png" in index_plots(root)


def test_training_and_evaluation_register_dependencies_before_savefig(tmp_path, monkeypatch):
    root, matrix, metadata = plot_batch(tmp_path)
    observed = []
    def savefig(figure, path, **kwargs):
        manifest = json.loads((root / "plots.json").read_text())
        relative = path.relative_to(root).as_posix()
        entry = manifest[relative]
        assert all((root / p).is_file() for p in entry["inputs"])
        observed.append((relative, entry))
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    run = root / "video/E3"
    save_run_artifacts(run, {}, {"train_loss": [1, .5], "val_mrr": [.2, .4]}, None)
    evaluation = run / "evaluations/cfg2"
    filtered, filtered_meta = keep_first_camera_per_scene(matrix, metadata)
    plot_similarity_heatmap(filtered, evaluation / "similarity_heatmap.png", max_items=1)
    plot_sorted_similarity_heatmap(filtered, filtered_meta,
                                   evaluation / "task_scene_sorted_similarity_heatmap.png")
    registry = json.loads((root / "video/runs.json").read_text())
    export_curves(registry, root / "video/charts")
    assert [entry["kind"] for _, entry in observed] == ["run_curves", "heatmap", "heatmap",
                                                       "curves"]
    assert observed[1][1]["options"] == {
        "label": "Video", "max_items": 1, "sort": False, "first_camera": True,
    }
    assert index_plots(root)[observed[1][0]]["options"]["max_items"] == 1


@pytest.mark.parametrize("operation", ["index", "record"])
def test_legacy_manifest_migrates_without_hashes_and_keeps_options(tmp_path, operation):
    root, _, _ = plot_batch(tmp_path)
    relative = "video/E3/evaluations/cfg2/similarity_heatmap.png"
    inputs = ["video/E3/evaluations/cfg2/similarity_matrix.npy",
              "video/E3/evaluations/cfg2/metadata.json"]
    options = {"label": "Video", "max_items": 2, "sort": False, "first_camera": True}
    manifest = root / "plots.json"
    write_json(manifest, {"version": 1, "batch": "s42", "plots": [{
        "path": relative, "kind": "heatmap",
        "inputs": [{"path": path, "sha256": "old digest"} for path in inputs],
        "upstream": [], "options": options,
    }]})
    if operation == "index":
        index_plots(root)
    else:
        record_plot(root / "video/E3/curves.png", "run_curves", [root / "video/E3/curves.json"])
    payload = json.loads(manifest.read_text())
    assert payload[relative] == {"kind": "heatmap", "inputs": inputs, "options": options}
    assert "sha256" not in manifest.read_text()


def test_index_rejects_missing_data_and_unrecognized_png_before_overwrite(tmp_path):
    root, _, _ = plot_batch(tmp_path)
    index_plots(root)
    manifest = root / "plots.json"
    before = manifest.read_bytes()
    (root / "extra.png").write_bytes(b"unknown figure")
    with pytest.raises(ValueError, match="extra.png"):
        redraw_plots(root)
    assert manifest.read_bytes() == before
    (root / "extra.png").unlink()
    (root / "video/E3/evaluations/cfg2/metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        redraw_plots(root)
    assert manifest.read_bytes() == before
    assert not list(root.rglob("*.png"))


def test_index_cli_and_curves_preserve_epoch_numbers(tmp_path, monkeypatch):
    root, _, _ = plot_batch(tmp_path)
    result = subprocess.run([sys.executable, str(ROOT / "runs/redraw.py"),
                             "--root", str(root), "--index-only"],
                            check=True, capture_output=True, text=True)
    assert "Indexed 16 plots" in result.stdout
    assert (root / "plots.json").is_file()
    assert not list(root.rglob("*.png"))
    points = []
    from matplotlib.axes import Axes
    from bise.common.comparison import plot_curves
    original_plot = Axes.plot
    def plot(axis, x, y, **kwargs):
        points.append((list(x), list(y)))
        return original_plot(axis, x, y, **kwargs)
    monkeypatch.setattr(Axes, "plot", plot)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *args, **kwargs: None)
    directory = root / "trajectory/charts"
    series = json.loads((directory / "curves.json").read_text())
    plot_curves(series, {"T4": "T4 label"}, directory / "curves.png", "trajectory / s42")
    assert points == [([1, 3], [1., .3]), ([2, 3], [.2, .4])]


@pytest.mark.parametrize("modality,group", [("trajectory", "T4"), ("video", "E3")])
@pytest.mark.parametrize("missing", ["directory", "cfg2", "cfg3"])
def test_redraw_rebuilds_missing_chart_data_from_saved_runs(tmp_path, monkeypatch,
                                                           modality, group, missing):
    root, _, _ = plot_batch(tmp_path)
    directory = root / modality
    run = directory / group
    write_json(run / "params.json", {"split": {"unit": "task"}})
    for cfg in ("cfg2", "cfg3"):
        source = run / "evaluations" / cfg / "metrics.json"
        metrics = json.loads(source.read_text())
        for values in metrics["human_to_robot"].values():
            values["MPR-all"] = 42.0
        write_json(source, metrics)
    before = {path: file_hash(path) for path in run.rglob("*") if path.is_file()}
    charts = directory / "charts"
    if missing == "directory":
        shutil.rmtree(charts)
    else:
        (charts / f"{missing}.json").unlink()
        (charts / f"{missing}.png").write_bytes(b"old image without aggregate JSON")

    from bise.common import evaluation
    def evaluate(*args, **kwargs):
        raise AssertionError("Redraw must use saved results without evaluating models")
    monkeypatch.setattr(evaluation, "validate_evaluation", evaluate)
    def savefig(figure, path, **kwargs):
        manifest = json.loads((root / "plots.json").read_text())
        assert path.relative_to(root).as_posix() in manifest
        assert len(manifest) == 16
        path.write_bytes(b"redrawn image")
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)

    payload = redraw_plots(root)
    cfgs = ("cfg2", "cfg3") if missing == "directory" else (missing,)
    for cfg in cfgs:
        restored = json.loads((charts / f"{cfg}.json").read_text())
        assert [experiment["group"] for experiment in restored["experiments"]] == [group]
        assert restored["experiments"][0]["test_scenes"] == 2
        assert restored["experiments"][0]["partition"] == ("task" if cfg == "cfg2" else "shared")
        mpr = [row["value"] for row in restored["rows"] if row["metric"] == "MPR-all"]
        assert mpr == [42., 42.]
        assert (charts / f"{cfg}.csv").is_file() and (charts / f"{cfg}.png").is_file()
        assert payload[f"{modality}/charts/{cfg}.png"]["inputs"] == [
            f"{modality}/charts/{cfg}.json"]
    assert (charts / "curves.png").is_file()
    assert all(file_hash(path) == digest for path, digest in before.items())
    assert "sha256" not in (root / "plots.json").read_text()


def test_rebuild_plot_index_without_any_evaluation_run_info(tmp_path):
    root, _, _ = plot_batch(tmp_path)
    for source in root.rglob("run_info.json"):
        source.unlink()
    for modality in ("trajectory", "video"):
        shutil.rmtree(root / modality / "charts")
    payload = index_plots(root)
    assert len(payload) == 16
    assert "run_info.json" not in json.dumps(payload)
    for modality in ("trajectory", "video"):
        assert (root / modality / "charts/cfg2.json").is_file()
        assert (root / modality / "charts/cfg3.json").is_file()


def test_redraw_prefers_existing_comparison_json_over_changed_run_metrics(tmp_path, monkeypatch):
    root, _, _ = plot_batch(tmp_path)
    chart = root / "trajectory/charts/cfg2.json"
    original = chart.read_bytes()
    metrics_path = root / "trajectory/T4/evaluations/cfg2/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["human_to_robot"]["task"]["R@1"] = .9
    write_json(metrics_path, metrics)
    (root / "trajectory/charts/cfg3.json").unlink()
    plotted = {}
    from bise.common import comparison
    def plot(experiments, path, title):
        plotted[path.relative_to(root).as_posix()] = experiments
        path.write_bytes(b"restored image")
    monkeypatch.setattr(comparison, "_plot", plot)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda figure, path, **kwargs: path.write_bytes(b"restored image"))
    redraw_plots(root)
    assert chart.read_bytes() == original
    task_metrics = plotted["trajectory/charts/cfg2.png"][0]["metrics"]["human_to_robot"]["task"]
    assert task_metrics["R@1"] == .5
    assert json.loads(metrics_path.read_text())["human_to_robot"]["task"]["R@1"] == .9


def test_rebuild_uses_available_evaluations_and_preserves_registry_order(tmp_path, monkeypatch,
                                                                      capsys):
    root, _, _ = plot_batch(tmp_path)
    directory = root / "trajectory"
    registry = json.loads((directory / "runs.json").read_text())
    copied = directory / "T2"
    shutil.copytree(directory / "T4", copied)
    write_json(copied / "params.json", {"split": {"unit": "scene"}})
    registry["groups"]["T2"] = {"run_dir": str(copied), "label": "T2 label"}
    registry["groups"]["N3"] = {"run_dir": str(directory / "N3"), "label": "N3"}
    write_json(directory / "runs.json", registry)
    (directory / "charts/cfg2.json").unlink()
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig",
                        lambda figure, path, **kwargs: path.write_bytes(b"restored image"))
    redraw_plots(root)
    data = json.loads((directory / "charts/cfg2.json").read_text())
    assert [experiment["group"] for experiment in data["experiments"]] == ["T4", "T2"]
    assert "Skip missing evaluation trajectory N3 cfg2" in capsys.readouterr().out
