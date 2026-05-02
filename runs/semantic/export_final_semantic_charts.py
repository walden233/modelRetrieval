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
    "text_only": "Text Description",
    "label_only": "Structured Labels",
    "text_plus_label": "Text + Labels",
}
METRIC_KEYS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")
METRIC_LABELS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")
PALETTE = ("#2A6FBB", "#D95F02", "#1B9E77")


def parse_args():
    parser = argparse.ArgumentParser(description="Export paper-ready semantic retrieval comparison chart.")
    parser.add_argument("--eval-dir", help="Directory containing metrics.json from evaluate_semantic_retrieval.py.")
    parser.add_argument("--metrics", help="Explicit metrics.json path. Overrides --eval-dir.")
    parser.add_argument("--output-dir", required=True, help="Directory for output PNG and JSON data.")
    parser.add_argument(
        "--direction",
        default="human_to_robot",
        help="Retrieval direction in metrics.json. Default: human_to_robot",
    )
    parser.add_argument(
        "--level",
        default="pair",
        choices=["pair", "task", "scene", "sample"],
        help="Positive level to plot. Default: pair",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument("--filename", default="semantic_metrics_comparison.png", help="Output chart filename.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = _resolve_metrics_path(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    experiments = _load_experiments(metrics, args.direction, args.level, metrics_path)

    _set_paper_style()
    _save_plot_data(
        experiments=experiments,
        output_dir=output_dir,
        filename=args.filename,
        metrics_path=metrics_path,
        direction=args.direction,
        level=args.level,
    )
    _plot_metrics(experiments, output_dir / args.filename, args.dpi, args.level)


def _resolve_metrics_path(args) -> Path:
    if args.metrics:
        metrics_path = Path(args.metrics)
    elif args.eval_dir:
        metrics_path = Path(args.eval_dir) / "metrics.json"
    else:
        raise ValueError("Either --metrics or --eval-dir is required.")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    return metrics_path


def _load_experiments(metrics: dict[str, Any], direction: str, level: str, metrics_path: Path) -> list[dict[str, Any]]:
    try:
        level_metrics = metrics[direction][level]
    except KeyError as exc:
        raise KeyError(f"metrics.json missing path [{direction}][{level}] in {metrics_path}") from exc

    experiments = []
    for mode_key in ("text_only", "label_only", "text_plus_label"):
        if mode_key not in level_metrics:
            raise KeyError(f"metrics.json missing semantic mode {mode_key!r} at [{direction}][{level}]")
        experiments.append(
            {
                "key": mode_key,
                "label": DEFAULT_LABELS[mode_key],
                "metrics": level_metrics[mode_key],
            }
        )
    return experiments


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


def _save_plot_data(
    experiments: list[dict[str, Any]],
    output_dir: Path,
    filename: str,
    metrics_path: Path,
    direction: str,
    level: str,
) -> None:
    payload = {
        "figure": filename,
        "metrics_path": str(metrics_path),
        "direction": direction,
        "level": level,
        "metric_keys": list(METRIC_KEYS),
        "metric_labels": list(METRIC_LABELS),
        "series": [
            {
                "key": experiment["key"],
                "label": experiment["label"],
                "metrics": {
                    metric_key: _safe_float(experiment["metrics"].get(metric_key, 0.0))
                    for metric_key in METRIC_KEYS
                },
            }
            for experiment in experiments
        ],
    }
    data_name = Path(filename).with_suffix(".json").name.replace(".json", "_data.json")
    (output_dir / data_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _plot_metrics(experiments: list[dict[str, Any]], output_path: Path, dpi: int, level: str) -> None:
    x = np.arange(len(METRIC_KEYS))
    group_width = 0.68
    bar_width = group_width / len(experiments)

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for index, experiment in enumerate(experiments):
        values = [_safe_float(experiment["metrics"].get(metric_key, 0.0)) for metric_key in METRIC_KEYS]
        offset = (index - (len(experiments) - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.9,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.8,
            label=experiment["label"],
        )
        _annotate_bars(ax, bars)

    ax.set_title(f"{level.title()}-Level Semantic Retrieval Metrics", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS)
    ax.set_ylim(0.0, _metric_ylim(experiments))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.22)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10)
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
            fontsize=8,
            color="#222222",
        )


def _metric_ylim(experiments: list[dict[str, Any]]) -> float:
    max_value = 0.0
    for experiment in experiments:
        max_value = max(
            max_value,
            *(_safe_float(experiment["metrics"].get(metric_key, 0.0)) for metric_key in METRIC_KEYS),
        )
    return min(1.0, max(0.15, max_value + 0.12))


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


if __name__ == "__main__":
    main()
