"""Compare validated evaluations using each run's recorded cfg2 split."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bise.common.evaluation import effective_config, validate_evaluation
from bise.common.fingerprints import write_json
from bise.common.paths import resolve_path
from bise.common.plots import comparison_sources, curve_sources, record_plot
from bise.data.rh20t.splits import (
    PROTOCOL, load_protocol, load_split_map, shared_scene_split, validate_split,
)


METRICS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")


def _write_comparison(registry: dict, cfg: str, experiments: list, output: Path) -> None:
    rows = []
    for experiment in experiments:
        for direction, levels in experiment["metrics"].items():
            for level, metrics in levels.items():
                for metric, value in metrics.items():
                    rows.append({"cfg": cfg, "group": experiment["group"],
                                 "partition": experiment["partition"], "direction": direction,
                                 "level": level, "metric": metric, "value": value,
                                 "queries": experiment["queries"],
                                 "candidates": experiment["candidates"],
                                 "test_scenes": experiment["test_scenes"],
                                 "test_tasks": experiment["test_tasks"],
                                 "split_manifest": experiment["split_manifest"],
                                 "run_dir": experiment["run_dir"]})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{cfg}.json", {"protocol": PROTOCOL, "cfg": cfg,
                                       "modality": registry["modality"],
                                       "batch": registry.get("batch"),
                                       "experiments": experiments, "rows": rows})
    with (output / f"{cfg}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(path: str | Path) -> dict:
    registry = json.loads(resolve_path(path).read_text())
    if (registry.get("protocol") != PROTOCOL
            or registry.get("modality") not in ("trajectory", "video")):
        raise ValueError("Expected a retrain trajectory/video runs.json registry")
    if not registry.get("groups"):
        raise ValueError("Run registry is empty")
    return registry


def eval_manifest(config: dict, run: Path, cfg: str) -> Path:
    if cfg == "cfg2":
        return run / "split_manifest.json"
    return resolve_path(config["eval_manifests"][cfg])


def prepare_raw_split(config: dict, run: Path) -> None:
    """Record a cfg2 split for raw baselines, which have no training stage."""
    path = run / "split_manifest.json"
    if path.exists():
        return
    protocol = load_protocol(config["eval_manifests"]["cfg2"])
    split = dict(config["split"])
    split["shared_manifest_path"] = config["eval_manifests"]["cfg2"]
    split.setdefault("seed", config.get("seed", 42))
    mapping = shared_scene_split(list(protocol["video_pairs"]), split)
    samples = {
        name: [pair["sample_id"] for scene in scenes
               for pair in protocol["video_pairs"][scene]]
        for name, scenes in mapping.items()
    }
    write_json(path, samples)


def _plot(experiments: list, path: Path, title: str) -> None:
    if path.with_suffix(".json").is_file():
        source = path.with_suffix(".json")
        record_plot(path, "comparison", [source], {"title": title}, comparison_sources(source))
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(experiments) * 1.6), 5.2))
    positions = np.arange(len(experiments))
    width = .8 / len(METRICS)
    has_task_split = any(x["partition"] == "task" for x in experiments)
    labels = [f"{x['label']}{'*' if x['partition'] == 'task' else ''}\n"
              f"{x['test_scenes']} scenes" for x in experiments]
    for axis, level in zip(axes, ("task", "scene")):
        for index, metric in enumerate(METRICS):
            values = [x["metrics"]["human_to_robot"][level][metric] for x in experiments]
            axis.bar(positions + (index - 2) * width, values, width, label=metric)
        axis.set_xticks(positions, labels, rotation=20)
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1)
        axis.set_title(f"{level.capitalize()} retrieval")
        axis.grid(axis="y", alpha=.2)
    fig.suptitle(title)
    fig.legend(*axes[1].get_legend_handles_labels(), ncol=len(METRICS),
               loc="lower center", bbox_to_anchor=(.5, .04 if has_task_split else .01))
    if has_task_split:
        fig.text(.5, .01, "* Task split; cfg2 uses each run's recorded test set.",
                 ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .16 if has_task_split else .12, 1, .95))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_comparison(registry: dict, cfgs: list[str], output: Path) -> dict:
    # Validate every input before replacing any previously published chart.
    gathered = {}
    for cfg in cfgs:
        experiments, reference = [], None
        for key, entry in registry["groups"].items():
            run = resolve_path(entry["run_dir"])
            config = json.loads((run / "params.json").read_text())
            config = effective_config(config, registry["modality"], cfg)
            manifest = eval_manifest(config, run, cfg)
            directory = run / "evaluations" / cfg
            checkpoint = run / ("backbone.pth" if entry.get("raw") else "best_model.pth")
            print(f"Validate {registry['modality']} {key} {cfg}", flush=True)
            info = validate_evaluation(directory, config, checkpoint,
                                       registry["modality"], manifest)
            partition = ("task" if cfg == "cfg2" and config["split"].get("unit") == "task"
                         else "shared")
            signature = (info["protocol_sha256"], info["data_sha256"], info["evaluator_sha256"],
                         info["sampling"])
            # cfg2 permits per-run test sets; cfg3 still uses one frozen inventory.
            if cfg == "cfg3":
                signature += (info["identities"],)
            if reference is not None and reference != signature:
                raise ValueError(f"Incompatible evaluation data/settings for {cfg}")
            reference = signature
            metrics = json.loads((directory / "metrics.json").read_text())
            for direction in ("human_to_robot", "robot_to_human"):
                for level in ("task", "scene"):
                    if any(not np.isfinite(value) for value in metrics[direction][level].values()):
                        raise ValueError(f"Non-finite metrics: {directory}")
                    for metric in (*METRICS, "MPR-all"):
                        if metric not in metrics[direction][level]:
                            raise ValueError(f"Missing {metric}: {directory}")
            counts = {} if entry.get("raw") else validate_split(
                load_split_map(run / "split_manifest.json"), config["split"].get("unit", "scene"))
            experiments.append({"group": key, "label": entry.get("label", key),
                                "run_dir": str(run), "partition": partition,
                                "queries": len(info["identities"]),
                                "candidates": len(info["identities"]),
                                "test_scenes": info["scenes"], "test_tasks": info["tasks"],
                                "split_manifest": str(manifest),
                                "split_counts": counts, "sampling": info["sampling"],
                                "checkpoint_selection": config.get("checkpoint_selection"),
                                "metrics": metrics,
                                "evaluation": info})
        gathered[cfg] = experiments
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cfg, experiments in gathered.items():
        _write_comparison(registry, cfg, experiments, output)
        test_label = "per-run test" if cfg == "cfg2" else "shared test"
        _plot(experiments, output / f"{cfg}.png",
              f"{registry['modality']} {registry.get('batch', 'retrain')} {cfg} / "
              f"{test_label} / human to robot")
        (output / f"{cfg}_task.png").unlink(missing_ok=True)
        summary[cfg] = {
            x["group"]: {"queries": x["queries"], "partition": x["partition"],
                         "test_scenes": x["test_scenes"], "test_tasks": x["test_tasks"],
                         "split_manifest": x["split_manifest"],
                         "human_to_robot": x["metrics"]["human_to_robot"]}
            for x in experiments
        }
    return summary


def export_curves(registry: dict, output: Path) -> None:
    series = {}
    labels = {}
    for key, entry in registry["groups"].items():
        path = resolve_path(entry["run_dir"]) / "curves.json"
        if entry.get("raw"):
            continue
        values = json.loads(path.read_text())
        series[key] = values
        labels[key] = entry.get("label", key)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "curves.json", series)
    plot_curves(series, labels, output / "curves.png",
                f"{registry['modality']} / {registry.get('batch', 'retrain')}")


def plot_curves(series: dict, labels: dict, path: Path, title: str) -> None:
    record_plot(path, "curves", [path.with_suffix(".json")],
                {"labels": labels, "title": title}, curve_sources(path, list(series)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, values in series.items():
        for axis, metric in zip(axes, ("train_loss", "val_mrr")):
            points = [(i + 1, v) for i, v in enumerate(values.get(metric, [])) if v is not None]
            if points:
                axis.plot(*zip(*points), label=labels[key])
            axis.set_title("Training loss" if metric == "train_loss"
                           else "Validation MRR (per-run split)")
            axis.set_xlabel("Epoch")
            axis.grid(alpha=.2)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
