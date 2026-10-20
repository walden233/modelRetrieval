"""Diagnostics of saved detections, not a ground-truth accuracy evaluation."""

import csv
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from .wilor import EDGES


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def camera_quality(data, legacy):
    counts = np.diff(data['frame_ptr'])
    points = data['joints_image_px']
    size = data['image_size_wh']
    finite = np.isfinite(points).all(axis=(1, 2))
    inside = ((points >= 0) & (points < size)).all(axis=2).mean(axis=1)
    previous = np.asarray(legacy['hand_count'])[data['frame_index']] if legacy else None
    return {
        'frames': len(counts), 'detection_coverage': float((counts > 0).mean()),
        'two_or_more_fraction': float((counts >= 2).mean()),
        'tracks': len(np.unique(data['track_id'])), 'detections': int(counts.sum()),
        'finite_fraction': float(finite.mean()) if len(finite) else None,
        'median_in_frame_joint_fraction': float(np.median(inside)) if len(inside) else None,
        'median_detection_score': float(np.median(data['scores'])) if len(points) else None,
        'mediapipe_detection_coverage': float((previous > 0).mean()) if previous is not None else None,
        'elapsed_seconds': float(data['elapsed_seconds']),
        'max_source_gap_seconds': float(np.diff(data['timestamp_ms']).max()/1000)
        if len(counts) > 1 else 0,
        'source_gaps_over_0_5_seconds': int((np.diff(data['timestamp_ms']) > 500).sum()),
    }


def track_quality(data):
    frame_ids = np.repeat(np.arange(len(data['timestamp_ms'])), np.diff(data['frame_ptr']))
    rows = []
    for track_id in np.unique(data['track_id']):
        selected = data['track_id'] == track_id
        indices = frame_ids[selected]
        time = data['timestamp_ms'][indices] / 1000
        xyz = data['joints_root_m'][selected]
        boxes = data['boxes'][selected]
        pixel = data['joints_image_px'][selected, 0]
        size = np.maximum(np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1), 1)
        dt = np.diff(time)
        consecutive = (np.diff(indices) == 1) & (dt > 0) & (dt <= .25)
        step = np.linalg.norm(np.diff(pixel, axis=0), axis=1) / ((size[1:] + size[:-1]) / 2)
        speed = step[consecutive] / dt[consecutive]
        bones = np.stack([np.linalg.norm(xyz[:, a] - xyz[:, b], axis=1) for a, b in EDGES], axis=1)
        cv = bones.std(axis=0) / np.maximum(bones.mean(axis=0), 1e-6)
        normal = np.cross(xyz[:, 5] - xyz[:, 0], xyz[:, 17] - xyz[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        flips = (normal[1:] * normal[:-1]).sum(axis=1) < 0
        rows.append({
            'track_id': int(track_id), 'observations': int(selected.sum()),
            'coverage': len(np.unique(indices)) / len(data['frame_index']),
            'first_frame': int(data['frame_index'][indices[0]]),
            'last_frame': int(data['frame_index'][indices[-1]]),
            'sufficient_temporal_support': bool(selected.sum() >= 5),
            'median_bone_cv': float(np.median(cv)) if selected.sum() >= 5 else None,
            'p90_wrist_speed_handboxes_per_second': float(np.quantile(speed, .9)) if len(speed) else None,
            'large_step_fraction': float((step[consecutive] > .5).mean()) if consecutive.any() else None,
            'palm_flip_fraction': float(flips[consecutive].mean()) if consecutive.any() else None,
            'ambiguous_association_fraction': float(data['association_ambiguous'][selected].mean()),
            'role': 'unassigned',
        })
    return rows


def fundamental(first, second):
    """Construct F for calibrated optical cameras in one common base frame."""
    k1, t1 = first
    k2, t2 = second
    relative = np.linalg.inv(t2) @ t1
    r, t = relative[:3, :3], relative[:3, 3]
    skew = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(k2).T @ skew @ r @ np.linalg.inv(k1)


def epipolar_distance(points1, points2, matrix):
    x1 = np.concatenate([points1, np.ones((*points1.shape[:-1], 1))], axis=-1)
    x2 = np.concatenate([points2, np.ones((*points2.shape[:-1], 1))], axis=-1)
    line2, line1 = x1 @ matrix.T, x2 @ matrix
    numerator = np.abs((x2 * line2).sum(axis=-1))
    return .5 * numerator * (1 / np.maximum(np.linalg.norm(line1[..., :2], axis=-1), 1e-9)
                             + 1 / np.maximum(np.linalg.norm(line2[..., :2], axis=-1), 1e-9))


def cross_view_quality(first, second, matrix, tolerance_ms=50):
    residuals, gaps = [], []
    ambiguity = []
    for i, time in enumerate(first['timestamp_ms']):
        j = int(np.argmin(np.abs(second['timestamp_ms'] - time)))
        gap = abs(second['timestamp_ms'][j] - time)
        if gap > tolerance_ms:
            continue
        a, b = first['frame_ptr'][i:i+2]
        c, d = second['frame_ptr'][j:j+2]
        if a == b or c == d:
            continue
        p, q = first['joints_image_px'][a:b], second['joints_image_px'][c:d]
        errors = epipolar_distance(p[:, None], q[None], matrix)
        costs = np.median(errors, axis=-1)
        for row, col in zip(*linear_sum_assignment(costs)):
            residuals.append(errors[row, col])
            gaps.append(gap)
            others = np.delete(costs[row], col)
            ambiguity.append(bool(len(others) and others.min() - costs[row, col] < 2))
    values = np.concatenate(residuals) if residuals else []
    return {'hand_observation_pairs': len(residuals),
            'median_epipolar_px': float(np.median(values)) if len(values) else None,
            'p90_epipolar_px': float(np.quantile(values, .9)) if len(values) else None,
            'median_time_gap_ms': float(np.median(gaps)) if gaps else None,
            'ambiguous_matching_fraction': float(np.mean(ambiguity)) if ambiguity else None}


def analyze(root, dataset_root):
    cameras, tracks, views = [], [], []
    for folder in sorted(root.glob('cfg*/task_*/scene_*')):
        identity = folder.relative_to(root).as_posix()
        cfg, task, scene = identity.split('/')
        source = dataset_root / f'RH20T_{cfg}' / task / scene
        old_path = source / 'human_hand_3d.npy'
        old = np.load(old_path, allow_pickle=True).item()['cameras'] if old_path.exists() else {}
        calibration_path = source / 'hand_camera_intrinsics_resolved.npy'
        if not calibration_path.exists():
            calibration_path = source / 'hand_camera_intrinsics.npy'
        intrinsic = np.load(calibration_path, allow_pickle=True).item()
        extrinsic = json.loads((source / 'T_base_from_camera.json').read_text())['cameras']
        saved, calibration = {}, {}
        for path in sorted(folder.glob('*.npz')):
            camera = path.stem
            with np.load(path, allow_pickle=False) as loaded:
                data = dict(loaded)
            saved[camera] = data
            cameras.append({'scene': identity, 'camera': camera,
                            **camera_quality(data, old.get(camera))})
            tracks.extend({'scene': identity, 'camera': camera, **row} for row in track_quality(data))
            if camera in intrinsic and camera in extrinsic:
                k = np.asarray(intrinsic[camera], dtype=float)[:, :3].copy()
                k[0] *= data['image_size_wh'][0] / 1280
                k[1] *= data['image_size_wh'][1] / 720
                calibration[camera] = (k, np.asarray(extrinsic[camera]['matrix']))
        for a, b in itertools.combinations(calibration, 2):
            views.append({'scene': identity, 'camera_a': a, 'camera_b': b,
                          **cross_view_quality(saved[a], saved[b], fundamental(calibration[a], calibration[b]))})
    if not cameras:
        raise ValueError(f'No saved WiLoR results under {root}')
    write_csv(root / 'cameras.csv', cameras)
    write_csv(root / 'tracks.csv', tracks)
    write_csv(root / 'views.csv', views)
    fig, axes = plt.subplots(1, 4, figsize=(23, 5))
    labels = [f"{r['scene'].replace('task_', 't').replace('scene_', 's')}\n{r['camera'][-4:]}" for r in cameras]
    x = np.arange(len(cameras))
    axes[0].bar(x-.18, [r['detection_coverage'] for r in cameras], .36, label='WiLoR')
    axes[0].bar(x+.18, [r['mediapipe_detection_coverage'] if r['mediapipe_detection_coverage'] is not None
                        else np.nan for r in cameras], .36, label='MediaPipe')
    axes[0].set(title='At least one detected hand / all frames', ylim=(0, 1.05))
    axes[0].set_xticks(x, labels, rotation=65, ha='right', fontsize=7)
    axes[0].legend()
    for ax, key, title in [(axes[1], 'median_bone_cv', 'Per-track median bone length CV'),
                           (axes[2], 'large_step_fraction', 'Wrist steps > 0.5 hand-box diagonal')]:
        ax.bar(range(len(tracks)), [r[key] if r[key] is not None else np.nan for r in tracks])
        ax.set(title=title, xlabel='Camera-local tracks (see tracks.csv)')
    available = [r for r in views if r['median_epipolar_px'] is not None]
    axes[3].bar(range(len(available)), [r['median_epipolar_px'] for r in available])
    axes[3].set(title='Cross-view median epipolar residual (px)',
                xlabel='Camera pairs (see views.csv)', yscale='log')
    fig.suptitle('WiLoR pilot diagnostics: detection coverage is NOT pose accuracy')
    fig.tight_layout()
    fig.savefig(root / 'quality.png', dpi=160)
    plt.close(fig)
    lines = ['# WiLoR pilot quality', '',
             '| Scene / camera | Frames | WiLoR detected | MediaPipe detected | Tracks |',
             '|---|---:|---:|---:|---:|']
    for row in cameras:
        previous = row['mediapipe_detection_coverage']
        previous = f'{previous:.1%}' if previous is not None else 'NA'
        lines.append(f"| {row['scene']} / {row['camera']} | {row['frames']} | "
                     f"{row['detection_coverage']:.1%} | {previous} | {row['tracks']} |")
    lines += ['', '## Cross-view diagnostics', '',
              '| Scene | Cameras | Paired hand observations | Median / P90 epipolar px |',
              '|---|---|---:|---:|']
    for row in views:
        value = (f"{row['median_epipolar_px']:.2f} / {row['p90_epipolar_px']:.2f}"
                 if row['median_epipolar_px'] is not None else 'NA')
        lines.append(f"| {row['scene']} | {row['camera_a']} / {row['camera_b']} | "
                     f"{row['hand_observation_pairs']} | {value} |")
    lines += ['', '## Interpretation limits', '',
              '- All hands are retained. Track IDs are camera-local, provisional, and have no task role.',
              '- Empty preparation frames count in detection coverage; this is not a miss rate.',
              '- MediaPipe comparison uses raw hand_count on identical frames, not depth-valid 21-point coverage.',
              '- The detection thresholds differ; higher coverage does not prove higher precision or pose accuracy.',
              '- joints_root_m is wrist-relative MANO geometry. joints_camera_estimate_m uses the model assumed focal length; it is not calibrated robot-base motion.',
              '- Bone CV and wrist-step flags are diagnostics, not automatic rejection rules. Real motion and ID errors both affect them.',
              '- Track counts are fragments, not the number of physical hands. Source timestamp gaps can split an otherwise stable hand track; see cameras.csv.',
              '- Tracks with fewer than five observations have no bone-stability score; a short fragment is not evidence of good quality.',
              '- views.csv uses nearest physical timestamps within 50 ms and optimistic pairwise epipolar matching; ambiguous identities and calibration errors remain. No fused trajectory is claimed.',
              '- Plot dependencies: quality.png <- cameras.csv, tracks.csv, views.csv; raw data is each camera NPZ.',
              '', 'Overlay videos and contact sheets are beside each camera NPZ. Source videos and existing trajectories are unchanged.']
    (root / 'report.md').write_text('\n'.join(lines)+'\n')
    return cameras, tracks, views
