import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap

bootstrap()


DEFAULT_LABELS = {
    "T1": "T1_Baseline",
    "T2": "T2_Augment",
    "T3": "T3_TwoStage",
    "T4": "T4_TaskHeld",
    "T5": "T5_21Keypoints",
}
METRIC_KEYS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")
METRIC_LABELS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")
PALETTE = ("#245A78", "#D67D2A", "#4F8F6B", "#9A4C4C", "#6C5B7B", "#2D9CDB")
MARKERS = ("o", "s", "^", "D", "P", "X")


def parse_args():
    parser = argparse.ArgumentParser(description="Export paper-ready final trajectory experiment charts.")
    parser.add_argument("--runs-json", required=True, help="JSON file mapping T1-T5 to run directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for output PNG and JSON files.")
    parser.add_argument("--direction", default="human_to_robot", choices=["human_to_robot", "robot_to_human"])
    parser.add_argument("--level", default="task", choices=["task", "scene"])
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument("--metrics-subdir", default="final_test", help="Run subdirectory containing metrics.json.")
    parser.add_argument("--metrics-filename", default="trajectory_metrics_comparison.png", help="Output filename for the metrics chart.")
    parser.add_argument("--curves-filename", default="trajectory_curves_comparison.png", help="Output filename for the curves chart.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(Path(args.runs_json))
    experiments = [
        _load_experiment(key, run_path, label, args.direction, args.level, args.metrics_subdir)
        for key, run_path, label in runs
    ]

    _set_paper_style()
    _save_plot_data(
        experiments,
        output_dir,
        direction=args.direction,
        level=args.level,
        curves_filename=args.curves_filename,
        metrics_filename=args.metrics_filename,
        metrics_subdir=args.metrics_subdir,
    )
    _plot_curves(experiments, output_dir / args.curves_filename, dpi=args.dpi)
    _plot_metrics(experiments, output_dir / args.metrics_filename, dpi=args.dpi)


def _load_runs(path: Path) -> list[tuple[str, Path, str | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runs json must be an object, for example: {'T1': '<run_path>', ...}")

    runs = []
    for index in range(1, 6):
        key = f"T{index}"
        raw_value = payload.get(key, payload.get(key.lower()))
        if raw_value is None:
            raise KeyError(f"Missing run path for {key} in {path}")
        runs.append(_parse_run_entry(key, raw_value))
    return runs


def _parse_run_entry(key: str, value: Any) -> tuple[str, Path, str | None]:
    run_path = _extract_run_path(value)
    if not run_path.exists():
        raise FileNotFoundError(f"{key} run path not found: {run_path}")
    return key, run_path, _extract_label(value)


def _extract_run_path(value: Any) -> Path:
    if isinstance(value, str):
        return Path(value)
    if isinstance(value, dict):
        for field in ("run_path", "path", "run_dir"):
            if field in value:
                return Path(value[field])
    raise ValueError(f"Invalid run mapping value: {value!r}")


def _extract_label(value: Any) -> str | None:
    if isinstance(value, dict):
        label = value.get("label")
        return str(label) if label else None
    return None


def _load_experiment(
    key: str,
    run_path: Path,
    label_override: str | None,
    direction: str,
    level: str,
    metrics_subdir: str,
) -> dict[str, Any]:
    curves_path = run_path / "curves.json"
    metrics_path = run_path / metrics_subdir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{key} metrics.json not found: {metrics_path}")

    params = _load_optional_json(run_path / "params.json")
    label = label_override or _label_for_experiment(key, params)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))[direction][level]
    return {
        "key": key,
        "label": label,
        "run_path": run_path,
        "curves": json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.exists() else None,
        "metrics": metrics,
    }


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _label_for_experiment(key: str, params: dict[str, Any]) -> str:
    mode = params.get("mode")
    split = params.get("split", {})
    use_6_keypoints = params.get("use_6_keypoints")

    if key == "T2":
        return "T2_Augment"
    if key == "T3":
        return "T3_TwoStage"
    if key == "T4":
        unit = split.get("unit")
        return "T4_TaskHeld" if unit == "task" else "T4_Generalization"
    if key == "T5":
        return "T5_21Keypoints" if use_6_keypoints is False else "T5_KeypointAblation"
    if key == "T1":
        return "T1_Baseline" if mode == "standard" else DEFAULT_LABELS[key]
    return DEFAULT_LABELS.get(key, key)


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )


def _plot_curves(experiments: list[dict[str, Any]], output_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    curve_specs = (
        ("train_loss", "Train Loss", "Loss"),
        ("val_mrr", "Validation MRR", "MRR"),
    )

    for axis, (curve_key, title, ylabel) in zip(axes, curve_specs):
        for index, experiment in enumerate(experiments):
            if experiment["curves"] is None:
                continue
            values = _clean_curve(experiment["curves"].get(curve_key, []))
            if not values:
                continue
            epochs = np.arange(1, len(values) + 1)
            axis.plot(
                epochs,
                values,
                color=PALETTE[index % len(PALETTE)],
                marker=MARKERS[index % len(MARKERS)],
                markersize=5.5,
                linewidth=2.2,
                label=experiment["label"],
            )
        axis.set_title(title, fontsize=14, fontweight="bold", pad=10)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85)
        axis.margins(x=0.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.25, wspace=0.24)
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 0.03), fontsize=10)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _clean_curve(values: list[Any]) -> list[float]:
    clean_values = []
    for value in values:
        if value is None:
            continue
        clean_values.append(float(value))
    return clean_values


def _save_plot_data(
    experiments: list[dict[str, Any]],
    output_dir: Path,
    direction: str,
    level: str,
    curves_filename: str,
    metrics_filename: str,
    metrics_subdir: str,
) -> None:
    curves_payload = {
        "figure": curves_filename,
        "series": [
            {
                "key": experiment["key"],
                "label": experiment["label"],
                "run_path": str(experiment["run_path"]),
                "train_loss": _clean_curve(experiment["curves"].get("train_loss", [])),
                "val_mrr": _clean_curve(experiment["curves"].get("val_mrr", [])),
            }
            for experiment in experiments
            if experiment["curves"] is not None
        ],
    }
    metrics_payload = {
        "figure": metrics_filename,
        "metrics_subdir": metrics_subdir,
        "direction": direction,
        "level": level,
        "metric_keys": list(METRIC_KEYS),
        "metric_labels": list(METRIC_LABELS),
        "series": [
            {
                "key": experiment["key"],
                "label": experiment["label"],
                "run_path": str(experiment["run_path"]),
                "metrics": {
                    metric_key: float(experiment["metrics"].get(metric_key, 0.0) or 0.0)
                    for metric_key in METRIC_KEYS
                },
            }
            for experiment in experiments
        ],
    }
    curves_data_name = Path(curves_filename).with_suffix(".json").name.replace(".json", "_data.json")
    metrics_data_name = Path(metrics_filename).with_suffix(".json").name.replace(".json", "_data.json")
    (output_dir / curves_data_name).write_text(
        json.dumps(curves_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / metrics_data_name).write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _plot_metrics(experiments: list[dict[str, Any]], output_path: Path, dpi: int) -> None:
    x = np.arange(len(METRIC_KEYS))
    group_width = 0.78
    bar_width = group_width / len(experiments)

    fig, ax = plt.subplots(figsize=(10.6, 5.6))
    for index, experiment in enumerate(experiments):
        values = [float(experiment["metrics"].get(metric_key, 0.0) or 0.0) for metric_key in METRIC_KEYS]
        offset = (index - (len(experiments) - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.92,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.7,
            label=experiment["label"],
        )
        _annotate_bars(ax, bars)

    ax.set_title("Task-Level Trajectory Retrieval Metrics", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0.0, _metric_ylim(experiments))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.26)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(3, len(experiments)), fontsize=10)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.008,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.5,
            color="#222222",
        )


def _metric_ylim(experiments: list[dict[str, Any]]) -> float:
    max_value = 0.0
    for experiment in experiments:
        max_value = max(max_value, *(float(experiment["metrics"].get(metric_key, 0.0) or 0.0) for metric_key in METRIC_KEYS))
    return min(1.0, max(0.15, max_value + 0.12))


if __name__ == "__main__":
    main()
