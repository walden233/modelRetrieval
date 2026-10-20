"""Role-selected, depth-anchored trajectories with gated multi-view triangulation."""

import itertools
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from bise.modalities.semantic.store import write_json, write_npz
from .hand_3d_extractor import (
    _load_scene_intrinsics, _load_optional_extrinsics, decode_rh20t_depth_frame,
    nearest_timestamp_indices, project_pixel_to_camera, sample_depth_median,
)
from .wilor import draw_hands
from .wilor_quality import EDGES, epipolar_distance, fundamental, write_csv
from .wilor_roles import load_scene, read_roles


def select_observations(data, camera, annotation, settings, include_idle=False):
    roles = ('main', 'support', 'idle') if include_idle else ('main', 'support')
    hands = [h['id'] for h in annotation['hands'] if h['role'] in roles]
    selected = np.full((len(hands), len(data['timestamp_ms'])), -1, np.int32)
    confidence = np.zeros_like(selected, dtype=np.float32)
    segments = next(c['segments'] for c in annotation['cameras'] if c['camera'] == camera)
    levels = {'low': .3, 'medium': .7, 'high': 1.0}
    threshold = levels[settings['min_role_confidence']]
    times = (data['timestamp_ms']-annotation['frames']['context']['origin_timestamp_ms'])/1000
    for seg in segments:
        if seg['hand_id'] not in hands or levels[seg['confidence']] < threshold:
            continue
        hand = hands.index(seg['hand_id'])
        for frame in np.flatnonzero((times >= seg['start_s']-.001) & (times <= seg['end_s']+.001)):
            a, b = data['frame_ptr'][frame:frame+2]
            candidates = np.arange(a, b)[data['track_id'][a:b] == seg['track_id']]
            if len(candidates) == 1:
                row = int(candidates[0])
                points = data['joints_image_px'][row]
                inside = ((points >= 0) & (points < data['image_size_wh'])).all(axis=1).mean()
                if (data['scores'][row] >= settings['min_detection_score']
                        and inside >= settings['min_inside_fraction']
                        and np.isfinite(points).all()
                        and np.isfinite(data['joints_root_m'][row]).all()
                        and not data['association_ambiguous'][row]):
                    if selected[hand, frame] >= 0 and selected[hand, frame] != row:
                        raise ValueError('Two detections assigned to the same physical hand')
                    selected[hand, frame] = row
                    confidence[hand, frame] = levels[seg['confidence']]
    return hands, selected, confidence


def fit_translation(root, measured, valid, settings):
    """Fit only the MANO root translation; reject incompatible surface/background depth."""
    good = valid & np.isfinite(measured).all(axis=1) & np.isfinite(root).all(axis=1)
    for _ in range(3):
        if good.sum() < settings['min_depth_anchors']:
            return None, good
        translation = np.median(measured[good]-root[good], axis=0)
        residual = np.linalg.norm(measured-root-translation, axis=1)
        good &= residual <= settings['max_anchor_residual_m']
    if good.sum() < settings['min_depth_anchors'] or good[[0, 5, 9, 13, 17]].sum() < 2:
        return None, good
    return np.median(measured[good]-root[good], axis=0), good


def recover_camera(data, source, camera, annotation, settings, include_idle=False):
    hands, rows, confidence = select_observations(data, camera, annotation, settings, include_idle)
    h, n = rows.shape
    pixels = np.full((h, n, 21, 2), np.nan, np.float32)
    local = np.full((h, n, 21, 3), np.nan, np.float32)
    for hand, frame in zip(*np.where(rows >= 0)):
        pixels[hand, frame] = data['joints_image_px'][rows[hand, frame]]
        local[hand, frame] = data['joints_root_m'][rows[hand, frame]]
    k, _ = _load_scene_intrinsics(source, camera, *data['image_size_wh'])
    transform, _, _ = _load_optional_extrinsics(source, camera)
    stamps = np.load(source / f'cam_{camera}_human_timestamps.npy', allow_pickle=True).item()
    depth_times = np.asarray(stamps['depth'], dtype=np.int64)
    depth_indices, delta, depth_time_valid = nearest_timestamp_indices(data['timestamp_ms'], depth_times)
    depth_time_valid &= np.abs(delta) <= settings['time_tolerance_ms']
    targets = {}
    for frame in np.flatnonzero(depth_time_valid & (rows >= 0).any(axis=0)):
        targets.setdefault(int(depth_indices[frame]), []).append(int(frame))
    camera_xyz = np.full_like(local, np.nan)
    anchors = np.zeros((h, n), np.int32)
    depth_mad = np.full((h, n), np.nan, np.float32)
    cap = cv2.VideoCapture(str(source / f'cam_{camera}_human_depth.mp4'))
    if not cap.isOpened():
        raise OSError(f'Missing readable depth video: {source}/{camera}')
    try:
        for depth_frame in range(len(depth_times)):
            ok, encoded = cap.read()
            if not ok:
                raise OSError(f'Depth decode failure {camera} frame {depth_frame}')
            if depth_frame not in targets:
                continue
            depth = decode_rh20t_depth_frame(encoded).astype(np.float32)*.001
            height, width = depth.shape
            scale = np.array([width, height])/data['image_size_wh']
            for frame in targets[depth_frame]:
                for hand in np.flatnonzero(rows[:, frame] >= 0):
                    measured = np.full((21, 3), np.nan)
                    valid, mads = np.zeros(21, bool), np.full(21, np.nan)
                    for joint, uv in enumerate(pixels[hand, frame]):
                        if not ((uv >= 0) & (uv < data['image_size_wh'])).all():
                            continue
                        z, count, mad = sample_depth_median(depth, *(uv*scale))
                        if np.isfinite(z) and count >= 3 and mad <= settings['max_depth_mad_m']:
                            measured[joint] = project_pixel_to_camera(*uv, z, k)
                            valid[joint], mads[joint] = True, mad
                    translation, good = fit_translation(local[hand, frame], measured, valid, settings)
                    anchors[hand, frame] = good.sum()
                    if translation is not None:
                        camera_xyz[hand, frame] = local[hand, frame]+translation
                        depth_mad[hand, frame] = np.median(mads[good])
    finally:
        cap.release()
    base = (camera_xyz @ transform[:3, :3].T + transform[:3, 3]
            if transform is not None else np.full_like(camera_xyz, np.nan))
    valid = np.isfinite(base).all(axis=(2, 3))
    # Reject isolated wrist spikes, leaving missing samples rather than interpolating.
    for hand in range(h):
        for frame in range(1, n-1):
            if not valid[hand, frame-1:frame+2].all():
                continue
            dt = np.diff(data['timestamp_ms'][frame-1:frame+2])/1000
            if not ((dt > 0) & (dt <= .25)).all():
                continue
            wrist = base[hand, frame-1:frame+2, 0]
            expected = wrist[0]+(wrist[2]-wrist[0])*dt[0]/dt.sum()
            if (np.linalg.norm(wrist[1]-expected) > settings['max_view_disagreement_m']
                    and np.linalg.norm(wrist[2]-wrist[0]) < settings['max_view_disagreement_m']):
                valid[hand, frame] = False
    base[~valid] = np.nan
    return {'hands': np.asarray(hands), 'roles': np.asarray([
                next(h['role'] for h in annotation['hands'] if h['id'] == identity) for identity in hands]),
            'timestamp_ms': data['timestamp_ms'],
            'frame_index': data['frame_index'], 'observation_index': rows,
            'role_confidence': confidence, 'joints_image_px': pixels, 'joints_root_m': local,
            'joints_camera_m': camera_xyz, 'points_base_m': base, 'valid_frame': valid,
            'depth_anchors': anchors, 'depth_mad_m': depth_mad, 'depth_time_delta_ms': delta,
            'intrinsics': k, 'T_base_from_camera': transform if transform is not None else np.full((4, 4), np.nan)}


def triangulate_joint(observations, settings):
    """Two-view hypotheses, cheirality, ray angle and reprojection consensus."""
    best = None
    for first, second in itertools.combinations(observations, 2):
        a, ka, ta = first
        b, kb, tb = second
        if float(epipolar_distance(a[None], b[None], fundamental((ka, ta), (kb, tb)))[0]) > settings['max_epipolar_px']:
            continue
        ra = ta[:3, :3] @ np.linalg.solve(ka, np.r_[a, 1])
        rb = tb[:3, :3] @ np.linalg.solve(kb, np.r_[b, 1])
        angle = np.degrees(np.arccos(np.clip(np.dot(ra, rb)/np.linalg.norm(ra)/np.linalg.norm(rb), -1, 1)))
        if min(angle, 180-angle) < settings['min_ray_angle_degrees']:
            continue
        pa, pb = ka @ np.linalg.inv(ta)[:3], kb @ np.linalg.inv(tb)[:3]
        homogeneous = cv2.triangulatePoints(pa, pb, a.reshape(2, 1), b.reshape(2, 1)).ravel()
        if abs(homogeneous[3]) < 1e-9:
            continue
        xyz = homogeneous[:3]/homogeneous[3]
        errors = []
        for pixel, k, transform in observations:
            camera = np.linalg.inv(transform) @ np.r_[xyz, 1]
            if camera[2] <= 0:
                errors.append(float('inf'))
                continue
            uv = k @ camera[:3]
            errors.append(float(np.linalg.norm(uv[:2]/uv[2]-pixel)))
        inliers = np.asarray(errors) <= settings['max_reprojection_px']
        if inliers.sum() < 2:
            continue
        error = float(np.median(np.asarray(errors)[inliers]))
        score = (int(inliers.sum()), -error)
        if best is None or score > best[0]:
            best = score, xyz, error
    return (best[1], best[2]) if best is not None else (None, None)


def canonicalize(points, valid):
    """Initial reliable palm frame, retaining the full physical timeline and masks."""
    normalized = np.full_like(points, np.nan)
    transforms = np.full((len(points), 4, 4), np.nan)
    scale = np.full(len(points), np.nan)
    for hand in range(len(points)):
        candidates = np.flatnonzero(valid[hand][:, [0, 5, 9, 17]].all(axis=1))
        for frame in candidates:
            p = points[hand, frame]
            x = p[5]-p[17]
            y = p[9]-p[0]
            width = np.linalg.norm(x)
            if width < 1e-6:
                continue
            x /= width
            y -= x*np.dot(x, y)
            if np.linalg.norm(y) < 1e-6:
                continue
            y /= np.linalg.norm(y)
            rotation = np.stack([x, y, np.cross(x, y)], axis=1)
            normalized[hand] = (points[hand]-p[0]) @ rotation / width
            transforms[hand] = np.eye(4)
            transforms[hand, :3, :3], transforms[hand, :3, 3] = rotation, p[0]
            scale[hand] = width
            break
    return normalized, transforms, scale


def fuse_cameras(cameras, annotation, settings):
    names = list(cameras)
    hands = list(next(iter(cameras.values()))['hands'])
    main = hands.index(annotation['main_hand']) if annotation['main_hand'] in hands else None
    scores = {camera: (int(data['valid_frame'][main].sum()) if main is not None else 0)
              for camera, data in cameras.items()}
    reference = max(names, key=lambda camera: scores[camera])
    # Verify camera pairs before using another view to fill a missing sample.
    verified = np.zeros((len(hands), len(names), len(names)), bool)
    pair_errors = []
    disagreements = []
    for ai, bi in itertools.combinations(range(len(names)), 2):
        a, b = cameras[names[ai]], cameras[names[bi]]
        if not np.isfinite(a['T_base_from_camera']).all() or not np.isfinite(b['T_base_from_camera']).all():
            continue
        matrix = fundamental((a['intrinsics'], a['T_base_from_camera']),
                             (b['intrinsics'], b['T_base_from_camera']))
        for hand in range(len(hands)):
            distances, epipolar = [], []
            for i in np.flatnonzero(a['valid_frame'][hand]):
                j = int(np.argmin(abs(b['timestamp_ms']-a['timestamp_ms'][i])))
                if (abs(b['timestamp_ms'][j]-a['timestamp_ms'][i]) > settings['time_tolerance_ms']
                        or not b['valid_frame'][hand, j]):
                    continue
                distances.append(np.linalg.norm(a['points_base_m'][hand, i, 0]-b['points_base_m'][hand, j, 0]))
                epipolar.append(np.median(epipolar_distance(a['joints_image_px'][hand, i],
                                                         b['joints_image_px'][hand, j], matrix)))
            if distances:
                if hand == main:
                    disagreements.extend(distances)
                okay = (len(distances) >= 5 and np.median(distances) <= settings['max_view_disagreement_m']
                        and np.median(epipolar) <= settings['max_epipolar_px'])
                verified[hand, ai, bi] = verified[hand, bi, ai] = okay
                pair_errors.append({'hand': hands[hand], 'camera_a': names[ai], 'camera_b': names[bi],
                                    'pairs': len(distances), 'median_wrist_m': float(np.median(distances)),
                                    'median_epipolar_px': float(np.median(epipolar)), 'verified': bool(okay)})
    preferred = [max(names, key=lambda c: int(cameras[c]['valid_frame'][hand].sum()))
                 for hand in range(len(hands))]
    times = cameras[reference]['timestamp_ms']
    shape = (len(hands), len(times), 21)
    points = np.full((*shape, 3), np.nan, np.float32)
    method = np.zeros(shape, np.uint8)  # 0 missing, 1 best single view, 2 triangulated
    source_camera = np.full(shape[:2], -1, np.int16)
    reprojection = np.full(shape, np.nan, np.float32)
    for frame, timestamp in enumerate(times):
        nearest = {}
        for camera, data in cameras.items():
            index = int(np.argmin(abs(data['timestamp_ms']-timestamp)))
            if abs(data['timestamp_ms'][index]-timestamp) <= settings['time_tolerance_ms']:
                nearest[camera] = index
        for hand in range(len(hands)):
            primary = preferred[hand]
            available = [c for c, i in nearest.items() if cameras[c]['valid_frame'][hand, i]
                         and (c == primary or verified[hand, names.index(primary), names.index(c)])]
            if not available:
                continue
            # One fixed best view per hand; switch only if its current observation is missing.
            best = max(available, key=lambda c: int(cameras[c]['valid_frame'][hand].sum()))
            index = nearest[best]
            base = cameras[best]['points_base_m'][hand, index]
            points[hand, frame], method[hand, frame] = base, 1
            source_camera[hand, frame] = names.index(best)
            if len(available) < 2:
                continue
            wrists = np.stack([cameras[c]['points_base_m'][hand, nearest[c], 0] for c in available])
            distances = [np.linalg.norm(a-b) for a, b in itertools.combinations(wrists, 2)]
            # Metric depth must independently support at least one camera pair.
            if min(distances) > settings['max_view_disagreement_m']:
                continue
            candidate = np.full((21, 3), np.nan)
            errors = np.full(21, np.nan)
            for joint in range(21):
                observations = [(cameras[c]['joints_image_px'][hand, nearest[c], joint],
                                 cameras[c]['intrinsics'], cameras[c]['T_base_from_camera']) for c in available]
                xyz, error = triangulate_joint(observations, settings)
                if xyz is not None:
                    candidate[joint], errors[joint] = xyz, error
            good = np.isfinite(candidate).all(axis=1)
            if good.sum() < 17 or not good[[0, 5, 9, 17]].all():
                continue
            agreement = np.linalg.norm(candidate[good]-base[good], axis=1)
            lengths = np.asarray([np.linalg.norm(candidate[a]-candidate[b]) for a, b in EDGES])
            original = np.asarray([np.linalg.norm(base[a]-base[b]) for a, b in EDGES])
            ratio = lengths/np.maximum(original, 1e-6)
            comparable = np.isfinite(ratio)
            if (np.median(agreement) > settings['max_view_disagreement_m']
                    or np.linalg.norm(candidate[0]-base[0]) > settings['max_view_disagreement_m']
                    or ((ratio[comparable] < .5) | (ratio[comparable] > 2)).any()):
                continue
            points[hand, frame, good] = candidate[good]
            method[hand, frame, good] = 2
            reprojection[hand, frame, good] = errors[good]
    valid = np.isfinite(points).all(axis=-1)
    normalized, initial_pose, scale = canonicalize(points, valid)
    return {'hands': np.asarray(hands), 'roles': np.asarray([
                next(h['role'] for h in annotation['hands'] if h['id'] == identity) for identity in hands]),
            'main_hand': np.asarray(annotation['main_hand'] or ''), 'timestamp_ms': times,
            'points_base_m': points, 'valid': valid, 'method': method,
            'source_camera': source_camera, 'cameras': np.asarray(names),
            'reprojection_px': reprojection, 'points_initial_palm': normalized,
            'initial_palm_in_base': initial_pose, 'palm_width_m': scale,
            'reference_camera': np.asarray(reference),
            'preferred_cameras': np.asarray(preferred), 'verified_camera_pairs': verified,
            'camera_pair_diagnostics': np.asarray(json.dumps(pair_errors)),
            'median_view_wrist_disagreement_m': np.asarray(np.median(disagreements) if disagreements else np.nan)}


def plot_trajectory(result, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    seconds = (result['timestamp_ms']-result['timestamp_ms'][0])/1000
    for hand, identity in enumerate(result['hands']):
        points = result['points_base_m'][hand]
        role = result['roles'][hand]
        for axis, label in enumerate('xyz'):
            axes[0].plot(seconds, points[:, 0, axis], label=f'{identity} {role} {label}')
        axes[1].plot(seconds, result['valid'][hand].mean(axis=1), label=f'{identity} {role}')
        axes[2].plot(seconds, (result['method'][hand] == 2).mean(axis=1), label=f'{identity} {role}')
    for ax, title in zip(axes, ('Wrist in base frame (m)', 'Valid joint fraction', 'Triangulated joint fraction')):
        ax.set(title=title, xlabel='Source time (s)')
        if len(result['hands']):
            ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    axes[1].set_ylim(-.05, 1.05)
    axes[2].set_ylim(-.05, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def selected_overlay(raw, selected, destination):
    cap = cv2.VideoCapture(str(raw['video_path']))
    width, height = raw['image_size_wh'].astype(int)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = np.median(np.diff(raw['frame_index'])) if len(raw['frame_index']) > 1 else 1
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), fps/stride, (width, height))
    if not writer.isOpened():
        raise OSError(destination)
    try:
        next_source_frame = 0
        for frame, source_index in enumerate(raw['frame_index']):
            while next_source_frame < source_index:
                if not cap.grab():
                    raise OSError(f'Cannot skip overlay frame {next_source_frame}')
                next_source_frame += 1
            ok, image = cap.read()
            next_source_frame += 1
            if not ok:
                raise OSError(f'Cannot decode overlay frame {source_index}')
            for hand, identity in enumerate(selected['hands']):
                row = selected['observation_index'][hand, frame]
                if row < 0:
                    continue
                image = draw_hands(image, [{k: raw[k][row] for k in
                    ('track_id', 'boxes', 'scores', 'joints_image_px')}])
                uv = raw['boxes'][row, :2].astype(int)
                label = f'{identity} {selected["roles"][hand]} 3D={int(selected["valid_frame"][hand, frame])}'
                cv2.putText(image, label, (max(0, uv[0]), max(30, uv[1]-20)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.putText(image, f'source {int(raw["timestamp_ms"][frame])} ms',
                        (4, height-8), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            writer.write(image)
    finally:
        cap.release()
        writer.release()


def build(root, dataset, scenes, settings):
    import time
    annotations = read_roles(root)
    rows = []
    for identity in scenes:
        started = time.perf_counter()
        if identity not in annotations:
            raise ValueError(f'Missing GLM annotation: {identity}; run --stage roles first')
        folder = root / identity
        raw = load_scene(folder)
        annotation = annotations[identity]
        current = {camera: {'size': (folder/f'{camera}.npz').stat().st_size,
                            'mtime_ns': (folder/f'{camera}.npz').stat().st_mtime_ns} for camera in raw}
        if current != annotation['source']:
            raise ValueError(f'Stale roles for {identity}; rerun --stage roles')
        cfg, task, scene = identity.split('/')
        source = dataset / f'RH20T_{cfg}' / task / scene
        output = folder / 'trajectory'
        output.mkdir(parents=True, exist_ok=True)
        cameras = {}
        identity_settings = settings.get('identity', {'enabled': False})
        check_identity = identity_settings.get('enabled', False)
        recovery_settings = ({**settings, 'min_role_confidence': 'low'} if check_identity else settings)
        for camera, data in raw.items():
            cameras[camera] = recover_camera(data, source, camera, annotation, recovery_settings,
                                             include_idle=check_identity)
        from .wilor_identity import correct_identities, finalize_camera
        cameras, annotation, identity_report = correct_identities(cameras, annotation, identity_settings)
        for camera, data in raw.items():
            cameras[camera] = finalize_camera(cameras[camera], {
                'low': .3, 'medium': .7, 'high': 1.0}[settings['min_role_confidence']])
            write_npz(output / f'{camera}.npz', **cameras[camera])
            selected_overlay(data, cameras[camera], output / f'{camera}.mp4')
        result = fuse_cameras(cameras, annotation, settings)
        write_npz(output / 'fused.npz', **result)
        plot_trajectory(result, output / 'quality.png')
        hands = list(result['hands'])
        main = hands.index(annotation['main_hand']) if annotation['main_hand'] in hands else None
        valid = result['valid'][main].all(axis=1) if main is not None else np.zeros(len(result['timestamp_ms']), bool)
        source_seconds = (result['timestamp_ms']-annotation['frames']['context']['origin_timestamp_ms'])/1000
        active = np.zeros(len(valid), bool)
        for camera in annotation['cameras']:
            for segment in camera['segments']:
                if segment['hand_id'] == annotation['main_hand'] and segment['confidence'] in ('high', 'medium'):
                    active |= (source_seconds >= segment['start_s']) & (source_seconds <= segment['end_s'])
        active_coverage = float(valid[active].mean()) if active.any() else 0.
        usable = int(valid.sum()) >= settings['min_trajectory_frames'] and active_coverage >= settings['min_trajectory_coverage']
        summary = {'scene': identity, 'task': annotation['task'], 'main_hand': annotation['main_hand'],
                   'hands': annotation['hands'], 'annotation_attempt_id': annotation['attempt_id'],
                   'reference_camera': str(result['reference_camera']), 'frames': len(valid),
                   'main_valid_frames': int(valid.sum()), 'main_coverage': float(valid.mean()),
                   'main_active_coverage': active_coverage,
                   'identity_swapped_cameras': ','.join(identity_report['swapped_cameras']),
                   'usable_single_view': bool(usable),
                   'main_triangulated_joint_fraction': float((result['method'][main] == 2).mean()) if main is not None else 0,
                   'median_view_wrist_disagreement_m': float(result['median_view_wrist_disagreement_m']),
                   'camera_pairs': json.loads(str(result['camera_pair_diagnostics'])),
                   'identity': identity_report,
                   'camera_quality': {camera: {str(hand): {
                       'selected_2d_frames': int((data['observation_index'][i] >= 0).sum()),
                       'valid_3d_frames': int(data['valid_frame'][i].sum())}
                       for i, hand in enumerate(data['hands'])} for camera, data in cameras.items()},
                   'elapsed_seconds': time.perf_counter()-started, 'settings': settings,
                   'coordinate_note': 'Depth-anchored MANO, using supplied human/robot same-serial extrinsic assumption. Not ground-truth pose.',
                   'method_codes': {'0': 'missing', '1': 'best single depth-anchored view', '2': 'gated triangulation'},
                   'normalization': 'Initial valid palm origin/axes; scale by initial index-pinky MCP span; missing values stay NaN.'}
        write_json(output / 'summary.json', summary)
        rows.append({k: summary[k] for k in ('scene', 'task', 'main_hand', 'reference_camera', 'frames',
                     'main_valid_frames', 'main_coverage', 'main_active_coverage', 'identity_swapped_cameras', 'usable_single_view',
                     'main_triangulated_joint_fraction', 'median_view_wrist_disagreement_m', 'elapsed_seconds')})
        print('BUILD', identity, rows[-1], flush=True)
    write_csv(root / 'trajectories.csv', rows)
    write_json(root / 'trajectory_params.json', settings)
    lines = ['# Role-selected WiLoR trajectories', '',
             '| Scene | Main hand | Valid / source frames | Active-window coverage | Swapped camera | Usable single view | Triangulated joints |',
             '|---|---|---:|---:|---|---|---:|']
    for row in rows:
        lines.append(f'| {row["scene"]} | {row["main_hand"]} | {row["main_valid_frames"]}/{row["frames"]} | '
                     f'{row["main_active_coverage"]:.1%} | {row["identity_swapped_cameras"] or "none"} | {row["usable_single_view"]} | '
                     f'{row["main_triangulated_joint_fraction"]:.1%} |')
    lines += ['', 'GLM roles are predictions requiring visual review, not ground-truth labels.',
              'Identity swaps are geometry-supported corrections, not ground-truth semantic labels; see each summary.json identity field.',
              'Single-view depth anchoring retains MANO hand shape; local depth can still fall on an object/background.',
              'Fusion requires physical-hand identity, source-time matching, depth agreement, epipolar/reprojection, ray-angle and bone-length checks.',
              'No interpolation or forced fusion. Support hands remain separate. No source dataset files are replaced.',
              'quality.png depends on fused.npz; root trajectories.csv aggregates each trajectory/summary.json.']
    (root / 'trajectories.md').write_text('\n'.join(lines)+'\n')
    return rows
