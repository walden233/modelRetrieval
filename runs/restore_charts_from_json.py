import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METRICS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")
PALETTE = (
    "#2A6FBB",
    "#D95F02",
    "#1B9E77",
    "#7570B3",
    "#C43C39",
    "#6A994E",
    "#8E6C8A",
    "#A6761D",
    "#4D4D4D",
)
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "*", "h")


def parse_args():
    parser = argparse.ArgumentParser(description="Restore PNG charts from saved final chart JSON data.")
    parser.add_argument("--root", default="artifacts", help="Root directory to scan recursively. Default: artifacts")
    parser.add_argument("--input", nargs="*", help="Explicit chart data JSON file(s). Overrides --root scanning.")
    parser.add_argument("--output-dir", help="Optional output directory. Default: same directory as each JSON.")
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    return parser.parse_args()


def main():
    args = parse_args()
    json_paths = [Path(path) for path in args.input] if args.input else _discover_chart_jsons(Path(args.root))
    if not json_paths:
        raise FileNotFoundError("No chart data JSON files found.")

    restored = []
    skipped = []
    for json_path in json_paths:
        output_path = _output_path(json_path, args.output_dir)
        if output_path.exists() and not args.overwrite:
            skipped.append(str(output_path))
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        _set_paper_style()
        if _is_curves_payload(payload):
            _plot_curves(payload, output_path, args.dpi)
        elif _is_metrics_payload(payload):
            _plot_metrics(payload, output_path, args.dpi)
        else:
            raise ValueError(f"Unsupported chart data schema: {json_path}")
        restored.append(str(output_path))

    print(json.dumps({"restored": restored, "skipped": skipped}, indent=2, ensure_ascii=False))


def _discover_chart_jsons(root: Path) -> list[Path]:
    patterns = ("*_data.json",)
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(set(paths))


def _output_path(json_path: Path, output_dir: str | None) -> Path:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    figure = payload.get("figure")
    if figure:
        filename = Path(str(figure)).name
    else:
        filename = json_path.name.replace("_data.json", ".png")
    return (Path(output_dir) / filename) if output_dir else (json_path.parent / filename)


def _is_curves_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("series"), list) and any(
        "train_loss" in item or "val_mrr" in item for item in payload["series"]
    )


def _is_metrics_payload(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("series"), list) and all(
        isinstance(item, dict) and "metrics" in item for item in payload["series"]
    )


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


def _plot_curves(payload: dict[str, Any], output_path: Path, dpi: int) -> None:
    series = payload.get("series", [])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    curve_specs = (
        ("train_loss", "Train Loss", "Loss"),
        ("val_mrr", "Validation MRR", "MRR"),
    )
    for axis, (curve_key, title, ylabel) in zip(axes, curve_specs):
        for index, item in enumerate(series):
            values = _clean_values(item.get(curve_key, []))
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
                label=str(item.get("label", item.get("key", f"Run {index + 1}"))),
            )
        axis.set_title(title, fontsize=14, fontweight="bold", pad=16)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85)
        axis.margins(x=0.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.25, wspace=0.24)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=_legend_columns(len(labels)), bbox_to_anchor=(0.5, 0.03), fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(payload: dict[str, Any], output_path: Path, dpi: int) -> None:
    series = payload.get("series", [])
    metric_keys = tuple(payload.get("metric_keys") or DEFAULT_METRICS)
    metric_labels = tuple(payload.get("metric_labels") or metric_keys)
    x = np.arange(len(metric_keys))
    group_width = 0.78 if len(series) > 3 else 0.68
    bar_width = group_width / max(len(series), 1)

    fig_width = max(9.4, 8.4 + 0.35 * len(series))
    fig, ax = plt.subplots(figsize=(fig_width, 5.6))
    for index, item in enumerate(series):
        metrics = item.get("metrics", {})
        values = [_safe_float(metrics.get(metric_key, 0.0)) for metric_key in metric_keys]
        offset = (index - (len(series) - 1) / 2.0) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width * 0.9,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.75,
            label=str(item.get("label", item.get("key", f"Run {index + 1}"))),
        )
        _annotate_bars(ax, bars)

    ax.set_title(_metrics_title(payload), fontsize=14, fontweight="bold", pad=16)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, _metric_ylim(series, metric_keys))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=_legend_columns(len(series)), fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _metrics_title(payload: dict[str, Any]) -> str:
    figure = str(payload.get("figure", "")).lower()
    level = str(payload.get("level", "")).strip()
    if "semantic" in figure:
        return f"{level.title()}-Level Semantic Retrieval Metrics" if level else "Semantic Retrieval Metrics"
    if "trajectory" in figure:
        return "Trajectory Retrieval Metrics"
    if "video" in figure:
        return "Video Retrieval Metrics"
    if "system" in figure or "eval_dir" in json.dumps(payload.get("series", [])[:1]):
        return "Retrieval System Metrics"
    return "Retrieval Metrics"


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


def _legend_columns(item_count: int) -> int:
    # 图例最多两行：7 个标签用 4 列形成 4+3，6 个用 3 列形成 3+3，5 个用 3 列形成 3+2。
    return max(1, int(np.ceil(item_count / 2)))


def _metric_ylim(series: list[dict[str, Any]], metric_keys: tuple[str, ...]) -> float:
    max_value = 0.0
    for item in series:
        metrics = item.get("metrics", {})
        max_value = max(max_value, *(_safe_float(metrics.get(metric_key, 0.0)) for metric_key in metric_keys))
    return min(1.0, max(0.15, max_value + 0.12))


def _clean_values(values: list[Any]) -> list[float]:
    return [_safe_float(value) for value in values if value is not None]


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
