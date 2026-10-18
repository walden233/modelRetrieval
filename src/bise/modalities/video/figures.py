import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from bise.common.plots import record_heatmap


def keep_first_camera_per_scene(matrix: np.ndarray, metadata: dict) -> tuple[np.ndarray, dict]:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"similarity_matrix must be square for first-camera heatmap filtering, got shape={matrix.shape}")

    scene_ids = metadata.get("scene_ids", [])
    task_ids = metadata.get("task_ids", [])
    if len(scene_ids) != matrix.shape[0]:
        raise ValueError(f"metadata scene_ids length must match matrix size: metadata={len(scene_ids)}, matrix={matrix.shape}")

    seen_scenes = set()
    selected_indices = []
    for index, scene_id in enumerate(scene_ids):
        key = (str(task_ids[index]) if index < len(task_ids) else "", str(scene_id))
        if key in seen_scenes:
            continue
        seen_scenes.add(key)
        selected_indices.append(index)

    selected_indices = np.array(selected_indices, dtype=int)
    filtered_matrix = matrix[np.ix_(selected_indices, selected_indices)]
    filtered_metadata = _filter_metadata(metadata, selected_indices.tolist())
    return filtered_matrix, filtered_metadata


def plot_video_heatmap(matrix: np.ndarray, metadata: dict, output_path: Path,
                       level: str, max_items: int = 0, source_dir: Path | None = None) -> None:
    from bise.common.plots import record_plot
    from bise.modalities.video.views import scene_groups, scene_max_similarity

    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Video heatmaps require a square paired-video similarity matrix')
    scene_groups(metadata, len(matrix))
    if level == 'scene':
        matrix, metadata = scene_max_similarity(matrix, metadata)
        order = np.arange(len(matrix))
        title = 'Scene level / max over all human-view x robot-view pairs'
        groups = metadata['task_ids']
    elif level == 'camera':
        order = np.asarray(sorted(range(len(matrix)), key=lambda i: (
            str(metadata['camera_ids'][i]), str(metadata['task_ids'][i]),
            str(metadata['scene_ids'][i]), str(metadata['sample_ids'][i]))))
        title = 'Camera level / every video / sorted by camera, task, scene'
        groups = [metadata['camera_ids'][i] for i in order]
    else:
        raise ValueError(f'Unknown video heatmap level: {level}')
    matrix = matrix[np.ix_(order, order)]
    sample = _sample_indices(len(matrix), max_items) if max_items > 0 else np.arange(len(matrix))
    displayed = matrix[np.ix_(sample, sample)]
    groups = [str(groups[i]) for i in sample]
    source_dir = source_dir or output_path.parent
    record_plot(output_path, 'video_heatmap', [source_dir / 'similarity_matrix.npy',
                                              source_dir / 'metadata.json'],
                {'level': level, 'max_items': max_items})
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    image = ax.imshow(displayed, aspect='equal', cmap='viridis', vmin=-1, vmax=1,
                      interpolation='nearest')
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            width, alpha = (.3, .25) if level == 'scene' else (.5, .75)
            ax.axhline(i - .5, color='white', linewidth=width, alpha=alpha)
            ax.axvline(i - .5, color='white', linewidth=width, alpha=alpha)
    if level == 'camera':
        starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
        ends = starts[1:] + [len(groups)]
        centers = [(start + end - 1) / 2 for start, end in zip(starts, ends)]
        labels = [groups[start] for start in starts]
        ax.set_xticks(centers, labels, rotation=30)
        ax.set_yticks(centers, labels)
    ax.set(title=f'{title}\n{displayed.shape[0]} x {displayed.shape[1]} '
                 f'(full matrix: {matrix.shape[0]} x {matrix.shape[1]})',
           xlabel='Robot camera videos' if level == 'camera' else 'Robot scenes',
           ylabel='Human camera videos' if level == 'camera' else 'Human scenes')
    fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label='Cosine similarity [-1, 1]')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_similarity_heatmap(matrix: np.ndarray, output_path: Path, max_items: int = 120, modality_label: str = "Video") -> None:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"similarity_matrix must be 2-D, got shape={matrix.shape}")
    record_heatmap(output_path, matrix, max_items, modality_label, sorted_by_task=False)
    row_indices = _sample_indices(matrix.shape[0], max_items)
    col_indices = _sample_indices(matrix.shape[1], max_items)
    matrix = matrix[np.ix_(row_indices, col_indices)]

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"{modality_label} Retrieval Similarity")
    ax.set_xlabel("Robot candidates")
    ax.set_ylabel("Human queries")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sorted_similarity_heatmap(
    matrix: np.ndarray,
    metadata: dict,
    output_path: Path,
    max_items: int = 120,
    modality_label: str = "Video",
) -> None:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"similarity_matrix must be 2-D, got shape={matrix.shape}")

    sort_keys = _sort_keys(metadata)
    if len(sort_keys) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "metadata length must match a square similarity matrix for sorted heatmap: "
            f"metadata={len(sort_keys)}, matrix={matrix.shape}"
        )

    record_heatmap(output_path, matrix, max_items, modality_label, sorted_by_task=True)
    order = np.array(sorted(range(len(sort_keys)), key=lambda index: sort_keys[index]))
    sorted_matrix = matrix[np.ix_(order, order)]
    sorted_keys = [sort_keys[index] for index in order]
    sampled_indices = _sample_indices(sorted_matrix.shape[0], max_items)
    sampled_matrix = sorted_matrix[np.ix_(sampled_indices, sampled_indices)]
    sampled_keys = [sorted_keys[index] for index in sampled_indices]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    image = ax.imshow(sampled_matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"{modality_label} Retrieval Similarity Sorted by Task")
    ax.set_xlabel("Robot candidates sorted by task/scene")
    ax.set_ylabel("Human queries sorted by task/scene")
    _draw_task_boundaries(ax, sampled_keys)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _filter_metadata(metadata: dict, indices: list[int]) -> dict:
    filtered = {}
    for key, values in metadata.items():
        if isinstance(values, list) and len(values) >= max(indices, default=-1) + 1:
            filtered[key] = [values[index] for index in indices]
        else:
            filtered[key] = values
    return filtered


def _sort_keys(metadata: dict) -> list[tuple[str, str, str, str]]:
    sample_ids = metadata.get("sample_ids", [])
    task_ids = metadata.get("task_ids", [])
    scene_ids = metadata.get("scene_ids", [])
    camera_ids = metadata.get("camera_ids", [])
    return [
        (
            str(task_ids[index]) if index < len(task_ids) else "",
            str(scene_ids[index]) if index < len(scene_ids) else "",
            str(camera_ids[index]) if index < len(camera_ids) else "",
            str(sample_ids[index]) if index < len(sample_ids) else "",
        )
        for index in range(len(sample_ids))
    ]


def _draw_task_boundaries(ax, sort_keys: list[tuple[str, str, str, str]]) -> None:
    task_boundaries = []
    for index in range(1, len(sort_keys)):
        if sort_keys[index][0] != sort_keys[index - 1][0]:
            task_boundaries.append(index - 0.5)

    for boundary in task_boundaries:
        ax.axhline(boundary, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(boundary, color="white", linewidth=0.8, alpha=0.8)


def _sample_indices(length: int, max_items: int) -> np.ndarray:
    if length <= max_items:
        return np.arange(length)
    return np.linspace(0, length - 1, num=max_items, dtype=int)
