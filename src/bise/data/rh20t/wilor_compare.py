"""Compare synchronized 3D hands in the existing N1 frame, independently of fusion."""

import itertools
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bise.modalities.semantic.store import write_npz
from .hand_3d_extractor import palm_pose_from_landmarks
from .human_representation import hand_xyz_features
from .wilor_quality import write_csv


def synchronized_rows(cameras, names, hand, tolerance_ms):
    """Match valid physical timestamps once per source frame, without interpolation."""
    indices = {c: np.flatnonzero(cameras[c]['valid_frame'][hand]) for c in names}
    if any(not len(i) for i in indices.values()):
        return np.empty((0, len(names)), dtype=int)
    reference = max(names, key=lambda c: len(indices[c]))
    proposals = []
    for i in indices[reference]:
        timestamp = cameras[reference]['timestamp_ms'][i]
        row, times = [], []
        for camera in names:
            candidates = indices[camera]
            j = int(candidates[np.argmin(abs(cameras[camera]['timestamp_ms'][candidates]-timestamp))])
            row.append(j)
            times.append(int(cameras[camera]['timestamp_ms'][j]))
        spread = max(times)-min(times)
        if spread <= tolerance_ms:
            proposals.append((spread, tuple(row)))
    chosen, used = [], [set() for _ in names]
    for _, row in sorted(proposals):
        if any(index in used[col] for col, index in enumerate(row)):
            continue
        chosen.append(row)
        for col, index in enumerate(row):
            used[col].add(index)
    return np.asarray(sorted(chosen), dtype=int).reshape(-1, len(names))


def common_anchor(cameras, names, hand, settings):
    rows = synchronized_rows(cameras, names, hand, settings['time_tolerance_ms'])
    count = settings['anchor_frames']
    for start in range(len(rows)-count+1):
        window = rows[start:start+count]
        poses, okay = {}, True
        for col, camera in enumerate(names):
            data = cameras[camera]
            indices = window[:, col]
            times = data['timestamp_ms'][indices]
            if (np.diff(times) <= 0).any() or (np.diff(times) > settings['max_window_gap_ms']).any():
                okay = False
                break
            local_poses = [palm_pose_from_landmarks(data['points_base_m'][hand, i], np.ones(21, bool))
                           for i in indices]
            if not np.isfinite(local_poses).all():
                okay = False
                break
            center = local_poses[count//2]
            for pose in local_poses:
                angle = np.degrees(np.arccos(np.clip(
                    (np.trace(center[:3, :3].T @ pose[:3, :3])-1)/2, -1, 1)))
                distance = np.linalg.norm(center[:3, 3]-pose[:3, 3])
                if (angle > settings['max_anchor_rotation_degrees']
                        or distance > settings['max_anchor_displacement_m']):
                    okay = False
                    break
            poses[camera] = center
        if okay:
            return {'poses': poses,
                    'indices': {c: int(window[count//2, col]) for col, c in enumerate(names)},
                    'window_rows': window}
    return None


def pair_metrics(first, second, rows, hand, anchor, names):
    a, b = names
    pa = first['points_base_m'][hand, rows[:, 0]]
    pb = second['points_base_m'][hand, rows[:, 1]]
    observed = np.ones(len(rows), bool)
    # Reuse the training representation exactly: initial palm center + rotation, no scale.
    na = hand_xyz_features(pa, observed, 'palm_relative', anchor['poses'][a])
    nb = hand_xyz_features(pb, observed, 'palm_relative', anchor['poses'][b])
    wa = first['points_base_m'][hand, anchor['indices'][a], 0]
    wb = second['points_base_m'][hand, anchor['indices'][b], 0]
    da = (pa[:, 0]-wa) @ anchor['poses'][a][:3, :3]
    db = (pb[:, 0]-wb) @ anchor['poses'][b][:3, :3]
    def rmse(x, y):
        return float(np.sqrt(np.mean(np.sum((x-y)**2, axis=-1))))
    norm = np.linalg.norm(da)*np.linalg.norm(db)
    # A static hand has no meaningful motion-direction cosine.
    cosine = float(np.sum(da*db)/norm) if norm > 1e-8 else None
    metrics = {'base_joint_rmse_m': rmse(pa, pb), 'n1_joint_rmse_m': rmse(na, nb),
               'n1_wrist_motion_rmse_m': rmse(da, db),
               'n1_root_relative_rmse_m': rmse(na-na[:, :1], nb-nb[:, :1]),
               'wrist_displacement_cosine': cosine,
               'anchor_time_a_ms': int(first['timestamp_ms'][anchor['indices'][a]]),
               'anchor_time_b_ms': int(second['timestamp_ms'][anchor['indices'][b]]),
               'median_time_gap_ms': float(np.median(abs(
                   first['timestamp_ms'][rows[:, 0]]-second['timestamp_ms'][rows[:, 1]])))}
    arrays = {'n1_a': na, 'n1_b': nb, 'wrist_displacement_a': da, 'wrist_displacement_b': db,
              'timestamp_a_ms': first['timestamp_ms'][rows[:, 0]],
              'timestamp_b_ms': second['timestamp_ms'][rows[:, 1]],
              'source_frame_a': first['frame_index'][rows[:, 0]],
              'source_frame_b': second['frame_index'][rows[:, 1]],
              'anchor_pose_a': anchor['poses'][a], 'anchor_pose_b': anchor['poses'][b]}
    return metrics, arrays


def candidate_metrics(first, second, rows, hand):
    fields = ('shared_candidate_joint_rmse_m', 'view_candidate_joint_rmse_m',
              'shared_candidate_motion_rmse_m', 'view_candidate_motion_rmse_m')
    result = {'candidate_paired_frames': 0, 'shared_candidate_available_frames': 0,
              'view_candidate_available_frames': 0, **{k: None for k in fields}}
    if not len(rows) or any('points_n1_shared' not in d or 'points_n1_view' not in d
                            for d in (first, second)):
        return result
    # Evaluate the two formal candidates on exactly the same physical observations.
    availability = {}
    for name in ('shared', 'view'):
        valid = np.ones(len(rows), bool)
        for d, col in ((first, 0), (second, 1)):
            valid &= d[f'n1_{name}_valid'][hand, rows[:, col]].all(axis=-1)
        availability[name] = valid
        result[f'{name}_candidate_available_frames'] = int(valid.sum())
    valid = availability['shared'] & availability['view']
    selected = rows[valid]
    result['candidate_paired_frames'] = len(selected)
    if len(selected) < 2:
        return result
    for name in ('shared', 'view'):
        a = first[f'points_n1_{name}'][hand, selected[:, 0]]
        b = second[f'points_n1_{name}'][hand, selected[:, 1]]
        result[f'{name}_candidate_joint_rmse_m'] = float(np.sqrt(np.mean(np.sum((a-b)**2, axis=-1))))
        da, db = a[:, 0]-a[0, 0], b[:, 0]-b[0, 0]
        result[f'{name}_candidate_motion_rmse_m'] = float(np.sqrt(np.mean(np.sum((da-db)**2, axis=-1))))
    return result


def plot_scene(arrays, records, path, gap_ms):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    colors = plt.get_cmap('tab10')
    for number, row in enumerate(records):
        if row['status'] != 'ok':
            continue
        key = row['key']
        for side, style in (('a', '-'), ('b', '--')):
            times = arrays[f'{key}_timestamp_{side}_ms']
            anchor = row[f'anchor_time_{side}_ms']
            wrist = arrays[f'{key}_wrist_displacement_{side}'].copy()
            # Break plot lines over gaps; no invented observations across missing intervals.
            gaps = np.r_[False, np.diff(times) > gap_ms]
            wrist[gaps] = np.nan
            for axis in range(3):
                axes[axis].plot((times-anchor)/1000, wrist[:, axis]*100, style,
                               color=colors(number % 10),
                               label=f'{row["hand"]} {row["camera_a"][-4:]}-{row["camera_b"][-4:]} {side}')
    for axis, label in zip(axes, 'xyz'):
        axis.set(title=f'N1 wrist displacement {label}', xlabel='Time from common anchor (s)', ylabel='cm')
        if axis.lines:
            axis.legend(fontsize=6)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(root, scenes, settings):
    if settings['anchor_frames'] < 2:
        raise ValueError('anchor_frames must be at least two')
    output_rows = []
    for identity in scenes:
        folder = root / identity / 'trajectory'
        cameras = {p.stem: dict(np.load(p, allow_pickle=False)) for p in sorted(folder.glob('*.npz'))
                   if p.stem not in ('fused', 'normalized')}
        if len(cameras) < 2:
            raise ValueError(f'Need at least two built cameras: {folder}')
        names = list(cameras)
        first = cameras[names[0]]
        records, arrays = [], {}
        for hand, physical_hand in enumerate(first['hands']):
            scene_anchor = common_anchor(cameras, names, hand, settings)
            for a, b in itertools.combinations(names, 2):
                rows = synchronized_rows(cameras, [a, b], hand, settings['time_tolerance_ms'])
                anchor = scene_anchor or common_anchor(cameras, [a, b], hand, settings)
                key = f'{physical_hand}_{a}_{b}'
                row = {'scene': identity, 'hand': str(physical_hand), 'role': str(first['roles'][hand]),
                       'camera_a': a, 'camera_b': b, 'key': key, 'paired_frames': len(rows),
                       'anchor_scope': 'all_views' if scene_anchor else 'pair',
                       'status': 'ok' if anchor and len(rows) >= settings['min_paired_frames'] else
                                 'insufficient_overlap' if len(rows) < settings['min_paired_frames'] else 'no_stable_anchor',
                       **{k: None for k in ('base_joint_rmse_m', 'n1_joint_rmse_m',
                           'n1_wrist_motion_rmse_m', 'n1_root_relative_rmse_m', 'wrist_displacement_cosine',
                           'anchor_time_a_ms', 'anchor_time_b_ms', 'median_time_gap_ms')}}
                row.update(candidate_metrics(cameras[a], cameras[b], rows, hand))
                if row['status'] == 'ok':
                    metrics, paired = pair_metrics(cameras[a], cameras[b], rows, hand, anchor, [a, b])
                    row.update(metrics)
                    arrays.update({f'{key}_{k}': value for k, value in paired.items()})
                records.append(row)
        arrays['records_json'] = np.asarray(json.dumps(records))
        arrays['settings_json'] = np.asarray(json.dumps(settings))
        write_npz(folder / 'normalized.npz', **arrays)
        plot_scene(arrays, records, folder / 'normalized.png', settings['max_window_gap_ms'])
        output_rows.extend(records)
        print('COMPARE', identity, [(r['hand'], r['camera_a'][-4:], r['camera_b'][-4:],
                                    r['status'], r['n1_joint_rmse_m']) for r in records], flush=True)
    write_csv(root / 'normalized.csv', output_rows)
    available = [r for r in output_rows if r['status'] == 'ok']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(available))
    labels = [f'{r["scene"].replace("task_", "t").replace("scene_", "s")} / {r["hand"]}\n'
              f'{r["camera_a"][-4:]}-{r["camera_b"][-4:]}' for r in available]
    axes[0].bar(x-.18, [100*r['base_joint_rmse_m'] for r in available], .36, label='Base frame')
    axes[0].bar(x+.18, [100*r['n1_joint_rmse_m'] for r in available], .36, label='N1 (no scaling)')
    axes[0].set(title='Paired 21-joint Euclidean RMSE', ylabel='cm')
    axes[0].legend()
    axes[1].bar(x-.18, [100*r['n1_wrist_motion_rmse_m'] for r in available], .36, label='Wrist displacement')
    axes[1].bar(x+.18, [100*r['n1_root_relative_rmse_m'] for r in available], .36, label='Wrist-relative joints')
    axes[1].set(title='Motion and wrist-relative hand disagreement', ylabel='cm')
    axes[1].legend()
    axes[2].bar(x, [r['wrist_displacement_cosine'] if r['wrist_displacement_cosine'] is not None
                   else np.nan for r in available])
    axes[2].set(title='Wrist displacement cosine (static: undefined)', ylim=(-1, 1))
    for ax in axes:
        ax.set_xticks(x, labels, rotation=60, ha='right', fontsize=7)
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(root / 'normalized.png', dpi=150)
    plt.close(fig)
    candidates = [r for r in output_rows if r['shared_candidate_joint_rmse_m'] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(candidates))
    labels = [f'{r["scene"].replace("task_", "t").replace("scene_", "s")} / {r["hand"]}\n'
              f'{r["camera_a"][-4:]}-{r["camera_b"][-4:]}' for r in candidates]
    for ax, metric, title in zip(axes, ('joint', 'motion'),
                               ('Joint disagreement', 'Wrist displacement disagreement')):
        ax.bar(x-.18, [100*r[f'shared_candidate_{metric}_rmse_m'] for r in candidates], .36, label='Shared N1')
        ax.bar(x+.18, [100*r[f'view_candidate_{metric}_rmse_m'] for r in candidates], .36, label='Per-view N1')
        ax.set(title=title+' (identical paired samples)', ylabel='RMSE (cm)')
        ax.set_xticks(x, labels, rotation=60, ha='right', fontsize=7)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'candidates.png', dpi=150)
    plt.close(fig)
    lines = ['# N1 cross-view 3D trajectory comparison', '',
             'Exact training N1: initial palm center and orientation, translation/rotation only, no scaling.',
             'Common stable source-time window; one anchor per view, fixed for the entire trajectory.',
             'A shared all-view anchor is preferred; otherwise each available camera pair has its own anchor.',
             'No DTW, time warping, per-frame rotation alignment, or interpolation. GLM is not called.', '',
             '| Scene / hand | Cameras | Matched | Anchor | Base / N1 joint RMSE (cm) | Wrist motion RMSE (cm) | Motion cosine | Status |',
             '|---|---|---:|---|---:|---:|---:|---|']
    for r in output_rows:
        valid = r['status'] == 'ok'
        errors = f'{100*r["base_joint_rmse_m"]:.2f} / {100*r["n1_joint_rmse_m"]:.2f}' if valid else 'NA'
        motion = f'{100*r["n1_wrist_motion_rmse_m"]:.2f}' if valid else 'NA'
        cosine = f'{r["wrist_displacement_cosine"]:.3f}' if r['wrist_displacement_cosine'] is not None else 'NA'
        lines.append(f'| {r["scene"]} / {r["hand"]} {r["role"]} | {r["camera_a"]}-{r["camera_b"]} | '
                     f'{r["paired_frames"]} | {r["anchor_scope"]} | {errors} | {motion} | {cosine} | {r["status"]} |')
    lines += ['', 'These are cross-view consistency diagnostics, not accuracy against ground truth.',
              'Normalization similarity does not validate metric calibration, identity or triangulation. Existing fusion gates are unchanged.',
              'RMSE is the root mean squared Euclidean joint distance, in meters in CSV.',
              'Plot dependencies: normalized.png <- normalized.csv; per-scene trajectory/normalized.png <- trajectory/normalized.npz.']
    lines += ['', '## Formal N1 candidates', '',
              'Shared N1 uses ONE main-hand palm frame after reconstruction for all views and both hands.',
              'Per-view N1 estimates its own palm frame per camera/hand at the shared source time when available. Both preserve meters; neither rescales.',
              'Both metrics below use the SAME paired source observations. Anchor times are saved in each camera NPZ.',
              'Shared rigid normalization preserves original cross-view distances; lower disagreement alone cannot decide retrieval quality.', '',
              '| Scene / hand | Cameras | Available shared / per-view / common | Shared / per-view joint RMSE (cm) | Shared / per-view motion RMSE (cm) |',
              '|---|---|---:|---:|---:|']
    for r in output_rows:
        if not r['paired_frames']:
            continue
        joint = (f'{100*r["shared_candidate_joint_rmse_m"]:.2f} / {100*r["view_candidate_joint_rmse_m"]:.2f}'
                 if r['shared_candidate_joint_rmse_m'] is not None else 'NA')
        motion = (f'{100*r["shared_candidate_motion_rmse_m"]:.2f} / {100*r["view_candidate_motion_rmse_m"]:.2f}'
                  if r['shared_candidate_motion_rmse_m'] is not None else 'NA')
        lines.append(f'| {r["scene"]} / {r["hand"]} | {r["camera_a"]}-{r["camera_b"]} | '
                     f'{r["shared_candidate_available_frames"]} / {r["view_candidate_available_frames"]} / '
                     f'{r["candidate_paired_frames"]} | {joint} | {motion} |')
    lines += ['', 'candidates.png <- normalized.csv. Legacy points_initial_palm remains the separate wrist/width auxiliary representation.']
    (root / 'normalized.md').write_text('\n'.join(lines)+'\n')
    return output_rows
