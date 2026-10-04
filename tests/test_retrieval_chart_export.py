import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "runs" / "retrieval" / "export_retrieval_system_charts.py"
    spec = importlib.util.spec_from_file_location("export_retrieval_system_charts_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_includes_optimized_fusion_series_and_weights(tmp_path, monkeypatch):
    module = _load_module()
    metrics = {key: 0.5 for key in module.METRIC_KEYS}
    uniform = tmp_path / "uniform"
    optimized = tmp_path / "optimized"
    for directory in (uniform, optimized):
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps({"task": metrics}))
    weights = {"video": 0.4, "trajectory": 0.2, "semantic_text": 0.4}
    (optimized / "config.json").write_text(json.dumps({"fusion": {"weights": weights}}))
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({
        "Video+Trajectory+Semantic": str(uniform),
        "Video+Trajectory+Semantic (Optimized)": str(optimized),
    }))
    output = tmp_path / "charts"
    monkeypatch.setattr(sys, "argv", [
        "export", "--runs-json", str(runs), "--output-dir", str(output),
        "--level", "task", "--dpi", "50",
    ])
    module.main()

    payload = json.loads((output / "system_metrics_comparison_data.json").read_text())
    assert len(payload["series"]) == 2
    assert payload["series"][-1]["label"].endswith("(Optimized)")
    assert payload["series"][-1]["fusion_weights"] == weights
    assert (output / "system_metrics_comparison.png").is_file()


def test_selected_weight_label_prints_weights_in_chart_legend():
    module = _load_module()
    label = module.plot_label(
        "Video+Trajectory+Semantic (cfg3-selected)",
        {"video": 0.5, "trajectory": 0.2, "semantic_text": 0.3},
    )
    assert label.endswith("V=0.50, T=0.20, S=0.30")
