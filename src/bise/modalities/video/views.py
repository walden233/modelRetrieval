"""Camera consistency and scene matching derived only from saved video embeddings."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bise.common.fingerprints import write_json
from bise.common.plots import _batch_root, record_plot
from bise.retrieval.metrics import calculate_label_retrieval_metrics


KINDS = ('HH_cross', 'RR_cross', 'HR_same', 'HR_cross')


def scene_groups(metadata: dict, size: int) -> list[tuple[tuple[str, str], list[int]]]:
    for key in ('task_ids', 'scene_ids', 'camera_ids', 'sample_ids'):
        if len(metadata.get(key, [])) != size:
            raise ValueError(f'{key} must have one entry per video pair ({size})')
    groups = defaultdict(list)
    for i, key in enumerate(zip(metadata['task_ids'], metadata['scene_ids'])):
        groups[tuple(map(str, key))].append(i)
    for key, indices in groups.items():
        cameras = [str(metadata['camera_ids'][i]) for i in indices]
        if len(cameras) != len(set(cameras)):
            raise ValueError(f'Duplicate camera within scene: {key}')
    return sorted(groups.items())


def scene_max_similarity(matrix: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Scene aggregation requires a square paired-video matrix')
    groups = scene_groups(metadata, len(matrix))
    result = np.array([[matrix[np.ix_(left, right)].max() for _, right in groups]
                       for _, left in groups], dtype=matrix.dtype)
    scene_metadata = {
        'task_ids': [key[0] for key, _ in groups],
        'scene_ids': [key[1] for key, _ in groups],
        'sample_ids': [key[0] + '::' + key[1] for key, _ in groups],
        'camera_ids': ['max' for _ in groups],
    }
    return result, scene_metadata


def analyze_views(human: np.ndarray, robot: np.ndarray, metadata: dict) -> dict:
    human, robot = np.asarray(human), np.asarray(robot)
    if human.ndim != 2 or human.shape != robot.shape or not len(human):
        raise ValueError('Human and robot embeddings must have the same nonempty shape')
    for vectors in (human, robot):
        if not np.isfinite(vectors).all() or not np.allclose(
                np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('View analysis requires finite L2-normalized video embeddings')
    groups = scene_groups(metadata, len(human))
    blocks, excluded = defaultdict(list), []
    for key, indices in groups:
        if len(indices) != 2:
            excluded.append({'task_id': key[0], 'scene_id': key[1], 'cameras': len(indices)})
            continue
        order = sorted(indices, key=lambda i: str(metadata['camera_ids'][i]))
        pair = tuple(str(metadata['camera_ids'][i]) for i in order)
        blocks[pair].append((key, order))
    observations, retrieval = [], []
    for pair, members in sorted(blocks.items()):
        # Same/cross-camera retrieval uses the identical scenes and gallery size.
        # A camera pair identifies a block; camera IDs are never relabeled across scenes.
        task_ids = np.asarray([key[0] for key, _ in members])
        scene_ids = np.asarray([key[0] + '::' + key[1] for key, _ in members])
        a, b = np.asarray([indices for _, indices in members]).T
        arrays = {'H_A': human[a], 'H_B': human[b], 'R_A': robot[a], 'R_B': robot[b]}
        comparisons = {
            'HH_cross': [('H_A', 'H_B'), ('H_B', 'H_A')],
            'RR_cross': [('R_A', 'R_B'), ('R_B', 'R_A')],
            'HR_same': [('H_A', 'R_A'), ('H_B', 'R_B')],
            'HR_cross': [('H_A', 'R_B'), ('H_B', 'R_A')],
        }
        for kind, links in comparisons.items():
            for query, gallery in links:
                similarity = arrays[query] @ arrays[gallery].T
                metrics = calculate_label_retrieval_metrics(
                    similarity, scene_ids, scene_ids, stable_sort=True)
                task_metrics = calculate_label_retrieval_metrics(
                    similarity, task_ids, task_ids, stable_sort=True)
                retrieval.append({'kind': kind, 'cameras': list(pair), 'query': query,
                                  'gallery': gallery, 'eligible': len(members) > 1,
                                  'metrics': metrics, 'task_metrics': task_metrics})
                for i, (key, _) in enumerate(members):
                    negatives = similarity[i, task_ids != task_ids[i]]
                    negative = float(negatives.mean()) if len(negatives) else None
                    positive = float(similarity[i, i])
                    observations.append({
                        'kind': kind, 'task_id': key[0], 'scene_id': key[1],
                        'cameras': list(pair), 'query': query, 'gallery': gallery,
                        'positive_cosine': positive, 'negative_cosine': negative,
                        'margin': positive - negative if negative is not None else None,
                        'negative_count': len(negatives), 'gallery_size': len(members),
                    })
    summary = {}
    for kind in KINDS:
        rows = [row for row in observations if row['kind'] == kind]
        stats = {'pairs': len(rows), 'positive_cosine': None, 'negative_cosine': None,
                 'margin': None, 'positive_std': None, 'scene_R@1': None, 'scene_MRR': None,
                 'task_R@1': None, 'task_MRR': None,
                 'negative_pairs': sum(row['negative_cosine'] is not None for row in rows),
                 'retrieval_queries': 0}
        for field in ('positive_cosine', 'negative_cosine', 'margin'):
            values = [row[field] for row in rows if row[field] is not None]
            stats[field] = float(np.mean(values)) if values else None
        stats['positive_std'] = float(np.std([row['positive_cosine'] for row in rows])) if rows else None
        parts = [row['metrics'] for row in retrieval if row['kind'] == kind and row['eligible']]
        count = sum(part['valid_queries'] for part in parts)
        stats['retrieval_queries'] = count
        for metric in ('R@1', 'MRR'):
            stats['scene_' + metric] = (sum(part[metric] * part['valid_queries'] for part in parts)
                                       / count if count else None)
            task_parts = [row['task_metrics'] for row in retrieval
                          if row['kind'] == kind and row['eligible']]
            stats['task_' + metric] = (sum(part[metric] * part['valid_queries'] for part in task_parts)
                                      / count if count else None)
        summary[kind] = stats
    scene_matrix, scene_meta = scene_max_similarity(human @ robot.T, metadata)
    scene_metrics = {}
    for direction, matrix in [('human_to_robot', scene_matrix), ('robot_to_human', scene_matrix.T)]:
        scene_metrics[direction] = {level: calculate_label_retrieval_metrics(
            matrix, scene_meta[key], scene_meta[key], stable_sort=True)
            for level, key in [('task', 'task_ids'), ('scene', 'sample_ids')]}
    same, cross = summary['HR_same'], summary['HR_cross']
    gaps = {field: same[field] - cross[field] if same[field] is not None and
            cross[field] is not None else None
            for field in ('positive_cosine', 'margin', 'scene_R@1', 'scene_MRR')}
    return {
        'schema_version': 1, 'video_pairs': len(human), 'scenes': len(groups),
        'complete_scenes': sum(len(rows) for rows in blocks.values()), 'excluded_scenes': excluded,
        'protocol': {
            'positive': 'same scene; HH/RR use distinct cameras; HR_same/cross use physical camera IDs',
            'negative': 'different task, same target camera, within the same physical-camera-pair block',
            'retrieval': 'task/scene positives; one candidate per scene; same/cross use identical galleries by scene; singleton blocks excluded',
            'missing_controls': 'positive cosine uses all complete scenes; negatives/margins omit queries with no different-task negatives',
            'aggregation': 'query-weighted across both camera directions and camera-pair blocks',
            'scene_similarity': 'max over all human-view x robot-view combinations for EVERY scene pair',
            'interpretation': 'scene max is best-view matching, not a measure of cross-camera invariance',
        },
        'camera_blocks': [{'cameras': list(pair), 'scenes': len(rows)} for pair, rows in sorted(blocks.items())],
        'summary': summary, 'same_minus_cross': gaps,
        'scene_max_metrics': scene_metrics, 'retrieval_blocks': retrieval, 'observations': observations,
    }


def prepare_views(directory: Path) -> dict:
    metadata = json.loads((directory / 'metadata.json').read_text())
    human = np.load(directory / 'human_embeddings.npy', allow_pickle=False)
    robot = np.load(directory / 'robot_embeddings.npy', allow_pickle=False)
    matrix = np.load(directory / 'similarity_matrix.npy', allow_pickle=False)
    if matrix.shape != (len(human), len(robot)) or not np.allclose(matrix, human @ robot.T, atol=1e-5):
        raise ValueError(f'Saved similarity matrix does not match embeddings: {directory}')
    payload = analyze_views(human, robot, metadata)
    write_json(directory / 'views.json', payload)
    fields = ['kind', *next(iter(payload['summary'].values())).keys()]
    with (directory / 'views.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({'kind': kind, **row} for kind, row in payload['summary'].items())
    return payload


def plot_views(data: dict, target: Path, source_dir: Path | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = [data['summary'][kind] for kind in KINDS]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for offset, field, label in [(-.2, 'positive_cosine', 'Same-scene positive'),
                                  (.2, 'negative_cosine', 'Different-task negative')]:
        values = [row[field] if row[field] is not None else np.nan for row in rows]
        bars = axes[0].bar(x + offset, values, .4, label=label)
        axes[0].bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    axes[0].set(ylabel='Cosine similarity', title='Consistency and background similarity', ylim=(-1, 1.12))
    axes[0].legend(fontsize=8)
    bars = axes[1].bar(x, [row['margin'] if row['margin'] is not None else np.nan for row in rows])
    axes[1].bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    axes[1].set(ylabel='Positive minus negative cosine', title='Discrimination margin')
    axes[1].margins(y=.25)
    for offset, field, label in [(-.2, 'task_R@1', 'Task R@1'), (.2, 'task_MRR', 'Task MRR')]:
        bars = axes[2].bar(x + offset, [row[field] if row[field] is not None else np.nan for row in rows],
                           .4, label=label)
        axes[2].bar_label(bars, fmt='%.3f', padding=2, fontsize=8)
    axes[2].set(ylabel='Score', title='Matched-gallery task retrieval', ylim=(0, 1.15))
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, ['H-H cross', 'R-R cross', 'H-R same', 'H-R cross'], rotation=18)
        axis.axhline(0, color='black', linewidth=.6)
        axis.grid(axis='y', alpha=.15)
    source_dir = source_dir or target.parent
    cfg = source_dir.name
    figure.suptitle(f"{source_dir.parent.parent.name} / {cfg} / camera consistency / "
                    f"{data['complete_scenes']}/{data['scenes']} complete scenes")
    record_plot(target, 'video_views', [source_dir / 'views.json'], upstream=[
        source_dir / name for name in ('human_embeddings.npy', 'robot_embeddings.npy', 'metadata.json')])
    figure.savefig(target, dpi=180)
    plt.close(figure)


def export_views(directory: Path) -> dict:
    from bise.modalities.video.figures import plot_video_heatmap

    data = prepare_views(directory)
    matrix = np.load(directory / 'similarity_matrix.npy', allow_pickle=False)
    metadata = json.loads((directory / 'metadata.json').read_text())
    for level in ('camera', 'scene'):
        plot_video_heatmap(matrix, metadata, directory / f'{level}_heatmap.png', level)
    plot_views(data, directory / 'views.png')
    for legacy in ('similarity_heatmap.png', 'task_scene_sorted_similarity_heatmap.png'):
        (directory / legacy).unlink(missing_ok=True)
    root = _batch_root(directory / 'views.png')
    if root is not None and (root / 'plots.json').exists():
        index = json.loads((root / 'plots.json').read_text())
        for legacy in ('similarity_heatmap.png', 'task_scene_sorted_similarity_heatmap.png'):
            index.pop((directory / legacy).resolve().relative_to(root.resolve()).as_posix(), None)
        write_json(root / 'plots.json', index)
    return data
