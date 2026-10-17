"""Paper figures and saved-data-only redraw for the semantic chapter."""

import json
import textwrap
from pathlib import Path

import numpy as np

from bise.common.paths import resolve_path
from bise.modalities.semantic.representations import REPRESENTATIONS
from bise.modalities.semantic.store import write_json


METRICS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10")


def prepare_figures(directory: Path, overwrite: bool = False) -> list[tuple[Path, list[Path]]]:
    if not directory.is_dir():
        return []
    independent = (directory / 'scope.json').exists()
    params_path = directory / 'params.json'
    settings = json.loads(params_path.read_text()) if params_path.exists() else {}
    allowed = settings.get('comparison_representations', REPRESENTATIONS)
    allowed = [rep for rep in allowed if rep not in settings.get('disabled', [])]
    active = [rep for rep in REPRESENTATIONS if rep in allowed and
              (directory / rep / 'annotations.jsonl').exists()]
    if independent:
        cfgs = list(json.loads((directory / 'scope.json').read_text())['sources'])
        if any(not (directory / 'evaluations' / cfg / 'metrics.json').exists() and
               any((directory / rep / 'evaluations' / cfg / 'metrics.json').exists() for rep in active)
               for cfg in cfgs):
            from bise.modalities.semantic.artifacts import aggregate
            aggregate(directory, active, cfgs)
    recipes = {}

    def filter_data(kind, data):
        if not independent:
            return True
        if kind == 'retrieval':
            for levels in data['metrics'].values():
                for level, values in levels.items():
                    levels[level] = {rep: value for rep, value in values.items() if rep in active}
            return bool(data['metrics'].get('human_to_robot', {}).get('task'))
        if kind == 'views':
            data['views'] = [row for row in data['views'] if row['representation'] in active]
            return bool(data['views'])
        if kind == 'case':
            data['texts'] = {rep: text for rep, text in data['texts'].items() if rep in active}
            return bool(data['texts'])
        if kind == 'cost':
            for variant in data['variants']:
                for levels in variant['metrics'].values():
                    for level, values in levels.items():
                        levels[level] = {rep: value for rep, value in values.items() if rep in allowed}
            return any(variant['metrics'].get('human_to_robot', {}).get('task')
                       for variant in data['variants'])
        return False

    def save(name: str, kind: str, data: dict, upstream: Path):
        path = directory / 'charts' / (name + '.json')
        recipe = {'kind': kind, 'data': data, 'source': str(upstream.relative_to(directory))}
        if not overwrite and path.exists():
            recipe = json.loads(path.read_text())
            if kind == 'retrieval':
                old_reps = set(recipe['data']['metrics'].get('human_to_robot', {}).get('task', {}))
                new_reps = set(data['metrics'].get('human_to_robot', {}).get('task', {}))
                if old_reps != new_reps:
                    recipe['data'] = data
            elif kind == 'case' and recipe['data']['sample_id'] != data['sample_id']:
                recipe['data'] = data
        if filter_data(kind, recipe['data']):
            write_json(path, recipe)
            recipes[path] = [upstream] if upstream.is_file() else []

    for source in sorted((directory / 'evaluations').glob('*/metrics.json')):
        payload = json.loads(source.read_text())
        save(payload['cfg'], 'retrieval', payload, source)
    source = directory / 'cost/summary.json'
    if source.exists():
        save('cost', 'cost', json.loads(source.read_text()), source)
    source = directory / 'analysis.json'
    if source.exists():
        payload = json.loads(source.read_text())
        if payload.get('views'):
            save('views', 'views', payload, source)
        for name, example in payload.get('examples', {}).items():
            save('cases/' + name, 'case', example, source)
    # Saved recipes remain usable when aggregation files are absent.
    for path in sorted((directory / 'charts').rglob('*.json')):
        if path in recipes:
            continue
        payload = json.loads(path.read_text())
        kind = payload.get('kind')
        if kind not in ('retrieval', 'cost', 'views', 'case'):
            continue
        if filter_data(kind, payload['data']):
            write_json(path, payload)
            upstream = directory / payload.get('source', '')
            recipes[path] = [upstream] if upstream.is_file() else []
        elif independent:
            path.unlink()
            path.with_suffix('.png').unlink(missing_ok=True)
    return list(recipes.items())


def render_figure(source: Path, target: Path, dpi: int = 150) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(source.read_text())
    data, kind = payload["data"], payload["kind"]
    if kind == "retrieval":
        figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
        levels = data["metrics"].get("human_to_robot", {})
        for axis, level in zip(axes, ("task", "scene")):
            values = levels.get(level, {})
            reps = [rep for rep in REPRESENTATIONS if rep in values]
            width = 0.8 / max(1, len(reps))
            for index, rep in enumerate(reps):
                ys = [values[rep].get(metric) or 0 for metric in METRICS]
                axis.bar(np.arange(len(METRICS)) + (index - (len(reps) - 1) / 2) * width,
                         ys, width, label=rep)
            axis.set_xticks(np.arange(len(METRICS)), METRICS)
            axis.set_ylim(0, 1.05)
            axis.set_ylabel("Score (0–1)")
            axis.set_title(level)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(ncol=max(1, len(reps)), fontsize=9)
        status = " / PARTIAL" if data.get("partial") else ""
        figure.suptitle(f"Semantic / {data['cfg']} / {data['split']} / human to robot{status}")
    elif kind == "cost":
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        cost_rep = next((rep for rep in REPRESENTATIONS if rep in data["variants"][0]["metrics"].get("human_to_robot", {}).get("task", {})), "S1")
        for row in data["variants"]:
            metric = row.get("metrics", {}).get("human_to_robot", {}).get("task", {}).get(cost_rep, {})
            engineering = row["engineering"]
            cost = engineering.get("known_cost_per_video")
            score = metric.get("MRR")
            if cost is not None and score is not None:
                axes[0].scatter(cost, score)
                axes[0].annotate(row["name"], (cost, score), xytext=(3, 4), textcoords="offset points", fontsize=8)
        axes[0].set(xlabel="Known API cost / selected video (CNY)", ylabel=f"{cost_rep} task MRR", ylim=(0, 1.05))
        rows = data["variants"]
        x = np.arange(len(rows))
        for offset, field, label in ((-0.2, "request_seconds_p50", "P50"),
                                      (0.2, "request_seconds_p95", "P95")):
            values = [row["engineering"].get(field) or 0 for row in rows]
            axes[1].bar(x + offset, values, 0.4, label=label)
        axes[1].set_xticks(x, [row["name"] for row in rows], rotation=20)
        axes[1].set_ylabel("API request seconds (cache reads excluded)")
        axes[1].legend()
        status = (f" / PARTIAL {data['common_videos']}/{data['target_videos']}" if
                  data.get('common_videos') != data.get('target_videos') else '')
        figure.suptitle(f"Cost subset / fixed videos / {cost_rep} shown; selected groups in raw data" + status)
    elif kind == "views":
        cfgs = sorted({row["cfg"] for row in data["views"]})
        figure, axes = plt.subplots(1, len(cfgs), figsize=(7 * len(cfgs), 5), squeeze=False,
                                    constrained_layout=True)
        categories = ("HH", "RR", "HR_same_camera", "HR_cross_camera")
        reps = [rep for rep in REPRESENTATIONS if any(row["representation"] == rep for row in data["views"])]
        for axis, cfg in zip(axes[0], cfgs):
            width = 0.8 / max(1, len(reps))
            for index, rep in enumerate(reps):
                means = []
                for category in categories:
                    margins = [row["margin"] for row in data["views"] if row["cfg"] == cfg and
                               row["representation"] == rep and row["kind"] == category and
                               row["margin"] is not None]
                    means.append(float(np.mean(margins)) if margins else np.nan)
                axis.bar(np.arange(len(categories)) + (index - (len(reps) - 1) / 2) * width, means, width, label=rep)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.set_xticks(np.arange(len(categories)), categories, rotation=15)
            axis.set_ylabel("Mean positive − matched different-task similarity")
            axis.set_title(cfg)
            axis.legend(ncol=max(1, len(reps)), fontsize=8)
        figure.suptitle("Independent-video semantic consistency and discrimination" +
                       (' / PARTIAL' if data.get('missing') else ''))
    elif kind == "case":
        from decord import VideoReader, cpu
        reader = VideoReader(str(resolve_path(data["video_path"])), ctx=cpu(0), num_threads=1)
        positions = np.linspace(0, len(data["frames"]["indices"]) - 1, 4, dtype=int)
        indices = [data["frames"]["indices"][int(i)] for i in positions]
        frames = reader.get_batch(indices).asnumpy()
        texts = []
        reps = [rep for rep in REPRESENTATIONS if rep in data["texts"]]
        for rep in reps:
            text = data["texts"][rep]
            if len(text) > 1200:
                text = text[:1200] + " ... [full text in report/JSON]"
            texts.append("\n".join(textwrap.wrap(text, 115)))
        heights = [3.0] + [max(1.1, 0.21 * (text.count('\n') + 2)) for text in texts]
        figure = plt.figure(figsize=(15, sum(heights)))
        grid = figure.add_gridspec(len(reps) + 1, 4, height_ratios=heights,
                                  left=0.025, right=0.975, bottom=0.015,
                                  top=1 - 0.55 / sum(heights), hspace=0.12)
        for column, (frame, position) in enumerate(zip(frames, positions)):
            axis = figure.add_subplot(grid[0, column])
            axis.imshow(frame)
            axis.set_title(f"{data['frames']['timestamps_seconds'][int(position)]:.2f}s")
            axis.axis("off")
        for row, (rep, text) in enumerate(zip(reps, texts), 1):
            axis = figure.add_subplot(grid[row, :])
            axis.axis("off")
            axis.text(0, 1, rep + "\n" + text, va="top", fontsize=9, transform=axis.transAxes)
        figure.suptitle(data["sample_id"], fontsize=11)
    else:
        raise ValueError(f"Unknown semantic figure type: {kind}")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi)
    plt.close(figure)


def plot_all(directory: Path, overwrite: bool = True) -> list[str]:
    outputs, index = [], {}
    for source, upstream in prepare_figures(directory, overwrite=overwrite):
        path = source.with_suffix(".png")
        index[str(path.relative_to(directory))] = {
            'kind': 'semantic', 'inputs': [str(source.relative_to(directory))],
            'upstream': [str(item.relative_to(directory)) for item in upstream]}
        write_json(directory / 'plots.json', index)
        render_figure(source, path)
        outputs.append(str(path))
    write_json(directory / 'plots.json', index)
    return outputs
