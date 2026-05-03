from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


METRIC_KEYS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")


def plot_system_metrics(series: list[dict], output_path: str | Path, title: str = "Retrieval System Metrics", dpi: int = 400):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.frameon": False,
        }
    )
    x_positions = range(len(METRIC_KEYS))
    group_width = 0.76
    bar_width = group_width / max(len(series), 1)
    colors = ("#2A6FBB", "#D95F02", "#1B9E77", "#7570B3", "#C43C39", "#6A994E")
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for index, item in enumerate(series):
        values = [float(item["metrics"].get(metric, 0.0) or 0.0) for metric in METRIC_KEYS]
        offset = (index - (len(series) - 1) / 2.0) * bar_width
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_width * 0.9,
            color=colors[index % len(colors)],
            edgecolor="white",
            linewidth=0.7,
            label=item["label"],
        )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Score")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(METRIC_KEYS)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.85)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.24)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(3, max(1, len(series))), fontsize=10)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
