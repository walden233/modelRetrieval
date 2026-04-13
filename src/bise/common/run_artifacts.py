import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch

from .paths import ensure_directory


def _json_default(value: Any) -> str:
    if isinstance(value, (torch.device, datetime.datetime, Path)):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _plot_series(ax, values, title: str, ylabel: str, color: str | None = None) -> None:
    points = [(index + 1, value) for index, value in enumerate(values) if value is not None]
    if not points:
        ax.set_visible(False)
        return

    epochs, clean_values = zip(*points)
    ax.plot(epochs, clean_values, color=color)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True)


def save_run_artifacts(
    run_dir: str | Path,
    config: Dict[str, Any],
    history: Dict[str, Any],
    best_result: Dict[str, Any] | None,
) -> None:
    output_dir = ensure_directory(run_dir)

    with (output_dir / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=_json_default)

    if best_result is not None:
        with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(best_result, handle, indent=2)

    if not history:
        return

    plot_specs = [
        ("train_loss", "Train Loss", "Loss", None),
        ("val_mean_p_rank", "Validation Mean Percentage Rank", "Mean Percentage Rank", "orange"),
        ("train_loss_inter", "Inter-modal Loss", "Loss", None),
        ("train_loss_intra", "Intra-modal Loss", "Loss", None),
    ]
    active_specs = [spec for spec in plot_specs if any(value is not None for value in history.get(spec[0], []))]

    if not active_specs:
        return

    num_plots = len(active_specs)
    num_cols = 2 if num_plots > 1 else 1
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows))
    axes = axes if isinstance(axes, (list, tuple)) else axes
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    else:
        axes = list(axes)
    flattened_axes = []
    for axis in axes:
        if hasattr(axis, "flatten"):
            flattened_axes.extend(axis.flatten().tolist())
        else:
            flattened_axes.append(axis)

    for ax, (key, title, ylabel, color) in zip(flattened_axes, active_specs):
        _plot_series(ax, history.get(key, []), title, ylabel, color=color)

    for ax in flattened_axes[len(active_specs):]:
        ax.set_visible(False)

    fig.suptitle(output_dir.name)
    fig.tight_layout()
    fig.savefig(output_dir / "curves.png")
    plt.close(fig)
