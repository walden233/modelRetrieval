import json
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
        "video/E3/evaluations/cfg2/run_info.json",
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
