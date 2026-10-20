"""Metric N1 candidates: one scene frame after reconstruction, or one frame per view."""

import numpy as np

from .hand_3d_extractor import palm_pose_from_landmarks


def stable_anchor(points, valid, timestamps, settings, reference_timestamp_ms=None):
    indices = np.flatnonzero(valid.all(axis=-1) & np.isfinite(points).all(axis=(-1, -2)))
    count = settings['anchor_frames']
    starts = list(range(len(indices)-count+1))
    if reference_timestamp_ms is not None:
        starts.sort(key=lambda i: abs(int(timestamps[indices[i+count//2]])-reference_timestamp_ms))
    for start in starts:
        window = indices[start:start+count]
        if (reference_timestamp_ms is not None
                and abs(int(timestamps[window[count//2]])-reference_timestamp_ms) > settings['time_tolerance_ms']):
            continue
        gaps = np.diff(timestamps[window])
        if (gaps <= 0).any() or (gaps > settings['max_window_gap_ms']).any():
            continue
        poses = [palm_pose_from_landmarks(points[i], valid[i]) for i in window]
        if not np.isfinite(poses).all():
            continue
        pose = poses[count//2]
        angles = [np.degrees(np.arccos(np.clip(
            (np.trace(pose[:3, :3].T @ p[:3, :3])-1)/2, -1, 1))) for p in poses]
        distances = [np.linalg.norm(p[:3, 3]-pose[:3, 3]) for p in poses]
        if (max(angles) <= settings['max_anchor_rotation_degrees']
                and max(distances) <= settings['max_anchor_displacement_m']):
            return int(window[count//2]), pose
    return None


def apply_frame(points, valid, pose):
    result = np.full_like(points, np.nan)
    if np.isfinite(pose).all():
        result[valid] = (points[valid]-pose[:3, 3]) @ pose[:3, :3]
    return result


def add_candidates(fused, cameras, settings):
    """One main-hand frame is shared by every camera AND the support hand, without scaling."""
    if settings['anchor_frames'] < 2:
        raise ValueError('anchor_frames must be at least two')
    labels = list(fused['hands'])
    main_label = str(fused['main_hand'])
    anchor, provenance = None, 'unavailable'
    if main_label in labels:
        hand = labels.index(main_label)
        points, valid = fused['points_base_m'][hand], fused['valid'][hand]
        # Prefer an anchor supported by triangulation. Explicitly label a single-view fallback.
        multi_valid = valid & (fused['method'][hand] == 2)
        anchor = stable_anchor(points, multi_valid, fused['timestamp_ms'], settings)
        if anchor is not None:
            provenance = 'multiview'
        else:
            anchor = stable_anchor(points, valid, fused['timestamp_ms'], settings)
            if anchor is not None:
                provenance = 'single_view'
    frame = np.full((4, 4), np.nan) if anchor is None else anchor[1]
    timestamp = -1 if anchor is None else int(fused['timestamp_ms'][anchor[0]])
    shared = {'n1_shared_pose_in_base': frame,
              'n1_shared_anchor_timestamp_ms': np.asarray(timestamp, np.int64),
              'n1_shared_anchor_source': np.asarray(provenance),
              'n1_shared_reference_hand': np.asarray(main_label)}
    fused.update(shared)
    fused['points_n1_shared'] = apply_frame(fused['points_base_m'], fused['valid'], frame)
    fused['n1_shared_valid'] = np.isfinite(fused['points_n1_shared']).all(axis=-1)
    view_anchors = {}
    for camera, data in cameras.items():
        # Both candidates use the identical source observations, quality masks and physical times.
        valid = np.isfinite(data['points_base_m']).all(axis=-1) & data['valid_frame'][..., None]
        data.update({k: np.array(v, copy=True) for k, v in shared.items()})
        data['points_n1_shared'] = apply_frame(data['points_base_m'], valid, frame)
        data['n1_shared_valid'] = np.isfinite(data['points_n1_shared']).all(axis=-1)
        individual = np.full_like(data['points_base_m'], np.nan)
        poses = np.full((len(labels), 4, 4), np.nan)
        times = np.full(len(labels), -1, np.int64)
        for hand in range(len(labels)):
            candidate = stable_anchor(data['points_base_m'][hand], valid[hand], data['timestamp_ms'], settings,
                                      reference_timestamp_ms=timestamp if anchor is not None else None)
            if candidate is not None:
                index, pose = candidate
                individual[hand] = apply_frame(data['points_base_m'][hand], valid[hand], pose)
                poses[hand], times[hand] = pose, data['timestamp_ms'][index]
        data.update(points_n1_view=individual, n1_view_pose_in_base=poses,
                    n1_view_anchor_timestamp_ms=times,
                    n1_view_valid=np.isfinite(individual).all(axis=-1))
        view_anchors[camera] = {str(label): int(times[i]) for i, label in enumerate(labels)}
    return {'shared_status': 'ok' if anchor is not None else 'no_stable_main_hand_anchor',
            'shared_anchor_source': provenance, 'shared_anchor_timestamp_ms': timestamp,
            'shared_reference_hand': main_label, 'view_anchor_timestamps_ms': view_anchors,
            'units': 'meters', 'scale_normalization': False, 'settings': settings,
            'shared_frame_scope': 'one fixed main-hand frame for all views and both physical hands',
            'view_anchor_policy': 'per-camera/hand palm at the shared source time; missing when unavailable. Independent stable anchor only if the shared candidate itself is unavailable.',
            'candidate_fields': {'shared': 'points_n1_shared', 'per_view': 'points_n1_view'},
            'comparison_note': 'Cross-view distance is invariant under the shared rigid frame. Consistency metrics are diagnostics, not retrieval performance.'}
