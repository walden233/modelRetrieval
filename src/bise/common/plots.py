"""Record retrain plot dependencies and redraw from saved JSON/NPY data."""

import json
from pathlib import Path

import numpy as np

from bise.common.fingerprints import write_json
from bise.common.paths import resolve_path


def _batch_root(path: Path) -> Path | None:
    for parent in path.resolve().parents:
        if (parent.parent.name == "retrain"
                or (parent / "trajectory" / "runs.json").is_file()
                or (parent / "video" / "runs.json").is_file()):
            return parent
    return None


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _entry(root: Path, kind: str, inputs: list[Path],
           options: dict, upstream: list[Path]) -> dict:
    def dependencies(paths: list[Path]) -> list[str]:
        result = []
        for source in dict.fromkeys(paths):
            if not source.is_file():
                raise FileNotFoundError(source)
            result.append(_relative(root, source))
        return result
    entry = {"kind": kind, "inputs": dependencies(inputs)}
    if upstream:
        entry["upstream"] = dependencies(upstream)
    if options:
        entry["options"] = options
    return entry


def _read_recipes(manifest: Path) -> dict:
    if not manifest.exists():
        return {}
    payload = json.loads(manifest.read_text())
    if "plots" not in payload:
        return payload
    # Migrate the original list format, keeping saved rendering options.
    recipes = {}
    for item in payload["plots"]:
        entry = {"kind": item["kind"],
                 "inputs": [source["path"] for source in item["inputs"]]}
        if item.get("upstream"):
            entry["upstream"] = [source["path"] for source in item["upstream"]]
        if item.get("options"):
            entry["options"] = item["options"]
        recipes[item["path"]] = entry
    return recipes


def record_plot(path: Path, kind: str, inputs: list[Path],
                options: dict | None = None, upstream: list[Path] | None = None) -> None:
    """Persist a plot recipe before rendering; leave historical run layouts alone."""
    root = _batch_root(path)
    if root is None:
        return
    manifest = root / "plots.json"
    plots = _read_recipes(manifest)
    plots[_relative(root, path)] = _entry(root, kind, inputs, options or {}, upstream or [])
    write_json(manifest, dict(sorted(plots.items())))


def comparison_sources(path: Path) -> list[Path]:
    payload = json.loads(path.read_text())
    modality = path.parent.parent
    sources = [modality / "runs.json"]
    for experiment in payload["experiments"]:
        evaluation = modality / experiment["group"] / "evaluations" / payload["cfg"]
        sources.append(evaluation / "metrics.json")
        run = modality / experiment["group"]
        optional = (evaluation / "metadata.json", run / "params.json",
                    run / "split_manifest.json")
        sources.extend(source for source in optional if source.is_file())
    return sources


def curve_sources(path: Path, keys: list[str]) -> list[Path]:
    modality = path.parent.parent
    return [modality / "runs.json", *[modality / key / "curves.json" for key in keys]]


def heatmap_metadata(metadata: dict, label: str) -> dict:
    if label != "Trajectory":
        return metadata
    return {
        "sample_ids": [f"{scene}/{camera}" for scene, camera in
                       zip(metadata["human_scene_ids"], metadata["human_camera_ids"])],
        "scene_ids": metadata["human_scene_ids"],
        "task_ids": metadata["human_task_ids"],
        "camera_ids": metadata["human_camera_ids"],
    }


def record_heatmap(path: Path, matrix: np.ndarray, max_items: int, label: str,
                   sorted_by_task: bool) -> None:
    if _batch_root(path) is None:
        return
    source = path.parent / "similarity_matrix.npy"
    original_shape = np.load(source, mmap_mode="r", allow_pickle=False).shape
    record_plot(path, "heatmap", [source, path.parent / "metadata.json"], {
        "label": label, "max_items": max_items, "sort": sorted_by_task,
        "first_camera": original_shape != matrix.shape,
    })


def _restore_charts(directory: Path) -> None:
    """Rebuild missing aggregate data from saved runs, without evaluating models."""
    from bise.common.comparison import _write_comparison, load_saved_experiments

    registry_path = directory / "runs.json"
    if not registry_path.is_file():
        return
    charts = directory / "charts"
    if all((charts / f"{name}.json").is_file() for name in ("curves", "cfg2", "cfg3")):
        return
    registry = json.loads(registry_path.read_text())
    if not (charts / "curves.json").is_file():
        series = {}
        for key, entry in registry["groups"].items():
            source = resolve_path(entry["run_dir"]) / "curves.json"
            if not entry.get("raw") and source.is_file():
                series[key] = json.loads(source.read_text())
        if series:
            write_json(charts / "curves.json", series)

    for cfg in ("cfg2", "cfg3"):
        if (charts / f"{cfg}.json").is_file():
            continue
        experiments = load_saved_experiments(registry, cfg)
        if experiments:
            print(f"Rebuild {directory.name} {cfg}: {len(experiments)} saved runs", flush=True)
            _write_comparison(registry, cfg, experiments, charts)


def index_plots(root: str | Path) -> dict:
    """Index existing data, including figures whose PNG has been deleted."""
    from bise.modalities.video.figures import keep_first_camera_per_scene

    root = resolve_path(root).resolve()
    if (root / 'scope.json').exists() and (root / 'params.json').exists():
        params = json.loads((root / 'params.json').read_text())
        if params.get('schema_version') == 3:
            from bise.modalities.semantic.figures import prepare_figures
            recipes = {}
            for source, upstream in prepare_figures(root):
                recipes[_relative(root, source.with_suffix('.png'))] = _entry(
                    root, 'semantic', [source], {}, upstream)
            write_json(root / 'plots.json', recipes)
            return recipes
    manifest = root / "plots.json"
    recipes = _read_recipes(manifest)
    plots = {}

    def add(path: Path, kind: str, inputs: list[Path], options: dict | None = None,
            upstream: list[Path] | None = None) -> None:
        old = recipes.get(_relative(root, path), {})
        if old.get("kind") == kind:
            options = old.get("options", {})
        plots[_relative(root, path)] = _entry(root, kind, inputs, options or {}, upstream or [])

    for modality in ("trajectory", "video"):
        directory = root / modality
        _restore_charts(directory)
        for source in sorted(directory.glob("*/curves.json")):
            if source.parent.name == "charts":
                series = json.loads(source.read_text())
                registry = json.loads((directory / "runs.json").read_text())
                add(source.with_suffix(".png"), "curves", [source], {
                    "labels": {key: registry["groups"][key].get("label", key) for key in series},
                    "title": f"{modality} / {registry.get('batch', root.name)}",
                }, curve_sources(source, list(series)))
            else:
                add(source.with_suffix(".png"), "run_curves", [source])

        for source in sorted((directory / "charts").glob("cfg*.json")):
            payload = json.loads(source.read_text())
            cfg = payload["cfg"]
            test = "per-run test" if cfg == "cfg2" else "shared test"
            add(source.with_suffix(".png"), "comparison", [source], {
                "title": f"{modality} {payload.get('batch', root.name)} {cfg} / "
                         f"{test} / human to robot",
            }, comparison_sources(source))

        for source in sorted(directory.glob("*/evaluations/*/similarity_matrix.npy")):
            metadata_path = source.parent / "metadata.json"
            label = "Trajectory" if modality == "trajectory" else "Video"
            matrix = np.load(source, allow_pickle=False)
            metadata = heatmap_metadata(json.loads(metadata_path.read_text()), label)
            try:
                keep_first_camera_per_scene(matrix, metadata)
                first_camera = True
            except ValueError:
                if modality != "trajectory":
                    raise
                first_camera = False
            for sort, filename in ((False, "similarity_heatmap.png"),
                                   (True, "task_scene_sorted_similarity_heatmap.png")):
                if sort and not first_camera:
                    continue
                add(source.parent / filename, "heatmap", [source, metadata_path], {
                    "label": label, "max_items": 120, "sort": sort,
                    "first_camera": first_camera,
                })

    from bise.modalities.semantic.figures import prepare_figures
    for source, upstream in prepare_figures(root / "semantic"):
        add(source.with_suffix(".png"), "semantic", [source], upstream=upstream)

    unknown = sorted(_relative(root, path) for path in root.rglob("*.png")
                     if _relative(root, path) not in plots)
    if unknown:
        raise ValueError(f"No saved plotting data for PNG files: {unknown}")
    if not plots:
        raise FileNotFoundError(f"No saved plot data found under {root}")
    payload = dict(sorted(plots.items()))
    write_json(manifest, payload)
    return payload


def redraw_plots(root: str | Path) -> dict:
    """Write the complete dependency index, then render every saved plot."""
    from bise.common.comparison import _plot, plot_curves
    from bise.common.run_artifacts import plot_training_curves
    from bise.modalities.video.figures import (
        keep_first_camera_per_scene, plot_similarity_heatmap, plot_sorted_similarity_heatmap,
    )

    root = resolve_path(root).resolve()
    payload = index_plots(root)
    for index, (relative, entry) in enumerate(payload.items(), 1):
        print(f"Redraw {index}/{len(payload)}: {relative}", flush=True)
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        inputs = [root / source for source in entry["inputs"]]
        options = entry.get("options", {})
        if entry["kind"] == "run_curves":
            plot_training_curves(json.loads(inputs[0].read_text()), path)
        elif entry["kind"] == "curves":
            plot_curves(json.loads(inputs[0].read_text()), options["labels"],
                        path, options["title"])
        elif entry["kind"] == "comparison":
            _plot(json.loads(inputs[0].read_text())["experiments"], path, options["title"])
        elif entry["kind"] == "heatmap":
            matrix = np.load(inputs[0], allow_pickle=False)
            metadata = heatmap_metadata(json.loads(inputs[1].read_text()), options["label"])
            if options["first_camera"]:
                matrix, metadata = keep_first_camera_per_scene(matrix, metadata)
            args = {"max_items": options["max_items"], "modality_label": options["label"]}
            if options["sort"]:
                plot_sorted_similarity_heatmap(matrix, metadata, path, **args)
            else:
                plot_similarity_heatmap(matrix, path, **args)
        elif entry["kind"] == "semantic":
            from bise.modalities.semantic.figures import render_figure
            render_figure(inputs[0], path)
        else:
            raise ValueError(f"Unknown plot kind: {entry['kind']}")
    return payload
