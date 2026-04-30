import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import bootstrap

bootstrap()

from bise.modalities.video.figures import keep_first_camera_per_scene, plot_similarity_heatmap, plot_sorted_similarity_heatmap


DIRECTION_LABELS = {
    "human_to_robot": "H2R",
    "robot_to_human": "R2H",
}
TASK_METRICS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")


def parse_args():
    parser = argparse.ArgumentParser(description="Build charts and summary tables for video experiments.")
    parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Evaluation output directory produced by evaluate_video.py. Repeat for comparisons.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Training run directory containing curves.json. Repeat to compare training curves.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures and CSV files.")
    parser.add_argument("--max-heatmap-items", type=int, default=120, help="Maximum matrix size to render in heatmaps.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dirs = [_parse_named_path(item) for item in args.eval_dir]
    run_dirs = [_parse_named_path(item) for item in args.run_dir]

    metric_rows = []
    case_rows = []
    for name, eval_dir in eval_dirs:
        metrics = _load_json(eval_dir / "metrics.json")
        metric_rows.append(_flatten_metrics(name, metrics))

        similarity_path = eval_dir / "similarity_matrix.npy"
        if similarity_path.exists():
            matrix = np.load(similarity_path)
            metadata_path = eval_dir / "metadata.json"
            if metadata_path.exists():
                metadata = _load_json(metadata_path)
                matrix, metadata = keep_first_camera_per_scene(matrix, metadata)
                plot_similarity_heatmap(matrix, output_dir / f"{_safe_name(name)}_similarity_heatmap.png", args.max_heatmap_items)
                plot_sorted_similarity_heatmap(
                    matrix,
                    metadata,
                    output_dir / f"{_safe_name(name)}_task_scene_sorted_similarity_heatmap.png",
                    args.max_heatmap_items,
                )
            else:
                plot_similarity_heatmap(matrix, output_dir / f"{_safe_name(name)}_similarity_heatmap.png", args.max_heatmap_items)

        _plot_single_eval_metrics(name, metrics, output_dir / f"{_safe_name(name)}_task_metrics.png")

        cases_path = eval_dir / "cases.json"
        if cases_path.exists():
            case_rows.extend(_summarize_cases(name, _load_json(cases_path)))

    if metric_rows:
        _write_csv(output_dir / "video_metrics_summary.csv", metric_rows)
        _plot_metric_comparison(metric_rows, output_dir / "video_metrics_comparison.png")

    if case_rows:
        _write_csv(output_dir / "video_cases_summary.csv", case_rows)

    if run_dirs:
        curves = [(name, _load_json(run_dir / "curves.json")) for name, run_dir in run_dirs if (run_dir / "curves.json").exists()]
        if curves:
            _plot_curve_comparison(curves, output_dir / "video_curves_comparison.png")


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got: {value}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise ValueError(f"Experiment name is empty in: {value}")
    if not path.exists():
        raise FileNotFoundError(f"Path not found for {name}: {path}")
    return name, path


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_metrics(name: str, metrics: dict) -> dict:
    row = {"experiment": name}
    for direction, direction_label in DIRECTION_LABELS.items():
        task_metrics = metrics.get(direction, {}).get("task", {})
        scene_metrics = metrics.get(direction, {}).get("scene", {})
        for metric_name in TASK_METRICS:
            row[f"{direction_label}_task_{metric_name}"] = task_metrics.get(metric_name)
        row[f"{direction_label}_scene_R@1"] = scene_metrics.get("R@1")
        row[f"{direction_label}_task_valid_queries"] = task_metrics.get("valid_queries")
    return row


def _summarize_cases(name: str, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        retrieved = case.get("retrieved") or []
        top1 = retrieved[0] if retrieved else {}
        rows.append(
            {
                "experiment": name,
                "query_index": case.get("query_index"),
                "query_label": case.get("query_label"),
                "query_path": case.get("query_path"),
                "top1_label": top1.get("label"),
                "top1_path": top1.get("path"),
                "top1_score": top1.get("score"),
                "top1_is_positive": top1.get("is_positive"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "experiment" in fieldnames:
        fieldnames.remove("experiment")
        fieldnames.insert(0, "experiment")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_single_eval_metrics(name: str, metrics: dict, output_path: Path) -> None:
    labels = []
    values = []
    for direction, direction_label in DIRECTION_LABELS.items():
        task_metrics = metrics.get(direction, {}).get("task", {})
        for metric_name in ("R@1", "R@5", "R@10", "MRR"):
            labels.append(f"{direction_label} {metric_name}")
            values.append(task_metrics.get(metric_name, 0.0) or 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="#2f6f73")
    ax.set_title(f"{name} Task Retrieval Metrics")
    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) * 1.15))
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_comparison(rows: list[dict], output_path: Path) -> None:
    metrics = ("H2R_task_R@1", "H2R_task_R@10", "H2R_task_MRR", "R2H_task_R@1", "R2H_task_R@10", "R2H_task_MRR")
    labels = [row["experiment"] for row in rows]
    x = np.arange(len(labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 5.5))
    for offset, metric_name in enumerate(metrics):
        values = [row.get(metric_name, 0.0) or 0.0 for row in rows]
        ax.bar(x + (offset - (len(metrics) - 1) / 2) * width, values, width, label=metric_name)

    ax.set_title("Video Experiment Metric Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncols=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_curve_comparison(curves: list[tuple[str, dict]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, history in curves:
        _plot_curve(axes[0], name, history.get("train_loss", []))
        _plot_curve(axes[1], name, history.get("val_mrr", []))

    axes[0].set_title("Train Loss")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation MRR")
    axes[1].set_ylabel("MRR")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(ax, name: str, values: list[float | None]) -> None:
    points = [(index + 1, value) for index, value in enumerate(values) if value is not None]
    if not points:
        return
    epochs, clean_values = zip(*points)
    ax.plot(epochs, clean_values, marker="o", linewidth=2, label=name)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "experiment"


if __name__ == "__main__":
    main()
