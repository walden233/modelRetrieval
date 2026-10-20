"""Conservative two-hand permutation checks before any multi-view fusion."""

import copy
import itertools

import numpy as np

from .wilor_quality import epipolar_distance, fundamental


HAND_FIELDS = ('observation_index', 'role_confidence', 'joints_image_px', 'joints_root_m',
               'joints_camera_m', 'points_base_m', 'valid_frame', 'depth_anchors', 'depth_mad_m')


def matched_both(first, second, tolerance_ms):
    """Same source samples for both identity hypotheses; one-to-one, no interpolation."""
    a = np.flatnonzero(first['valid_frame'].all(axis=0))
    b = np.flatnonzero(second['valid_frame'].all(axis=0))
    if not len(a) or not len(b):
        return np.empty((0, 2), int)
    proposals = []
    for i in a:
        j = int(b[np.argmin(abs(second['timestamp_ms'][b]-first['timestamp_ms'][i]))])
        gap = abs(int(first['timestamp_ms'][i])-int(second['timestamp_ms'][j]))
        if gap <= tolerance_ms:
            proposals.append((gap, int(i), j))
    chosen, used_a, used_b = [], set(), set()
    for _, i, j in sorted(proposals):
        if i not in used_a and j not in used_b:
            chosen.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return np.asarray(sorted(chosen), int).reshape(-1, 2)


def pair_evidence(first, second, settings):
    rows = matched_both(first, second, settings['time_tolerance_ms'])
    if len(rows) < settings['min_paired_frames']:
        return None
    if not (np.isfinite(first['T_base_from_camera']).all()
            and np.isfinite(second['T_base_from_camera']).all()):
        return None
    matrix = fundamental((first['intrinsics'], first['T_base_from_camera']),
                         (second['intrinsics'], second['T_base_from_camera']))
    hypotheses = []
    for swap in (False, True):
        distances, epipolar = [], []
        for hand in range(2):
            other = 1-hand if swap else hand
            pa = first['points_base_m'][hand, rows[:, 0]]
            pb = second['points_base_m'][other, rows[:, 1]]
            distances.append(float(np.median(np.linalg.norm(pa[:, 0]-pb[:, 0], axis=-1))))
            epipolar.append(float(np.median(epipolar_distance(
                first['joints_image_px'][hand, rows[:, 0]],
                second['joints_image_px'][other, rows[:, 1]], matrix))))
        score = .5*(np.mean(distances)/settings['max_wrist_distance_m']
                    + np.mean(epipolar)/settings['max_epipolar_px'])
        verified = (max(distances) <= settings['max_wrist_distance_m']
                    and max(epipolar) <= settings['max_epipolar_px'])
        hypotheses.append({'score': float(score), 'wrist_distance_m': distances,
                           'epipolar_px': epipolar, 'verified': bool(verified)})
    return {'paired_frames': len(rows), 'rows': rows, 'same': hypotheses[0], 'swap': hypotheses[1]}


def correct_identities(cameras, annotation, settings):
    """Keep semantic role labels anchored by minimum changes to the GLM assignments."""
    names = list(cameras)
    report = {'status': 'unchanged', 'swapped_cameras': [], 'pairs': [], 'settings': settings}
    if not settings.get('enabled', True):
        report['status'] = 'disabled'
        return cameras, annotation, report
    if len(names) < 2 or any(len(d['hands']) != 2 for d in cameras.values()):
        report['status'] = 'not_two_physical_hands'
        return cameras, annotation, report
    if any(list(d['hands']) != list(cameras[names[0]]['hands']) for d in cameras.values()):
        raise ValueError('Physical hand labels must have the same order across views')
    edges = []
    for ai, bi in itertools.combinations(range(len(names)), 2):
        evidence = pair_evidence(cameras[names[ai]], cameras[names[bi]], settings)
        if evidence is not None:
            edges.append((ai, bi, evidence))
            report['pairs'].append({'camera_a': names[ai], 'camera_b': names[bi],
                                    **{k: v for k, v in evidence.items() if k != 'rows'}})
    if not edges:
        report['status'] = 'insufficient_joint_overlap_or_calibration'
        return cameras, annotation, report
    active = sorted({i for a, b, _ in edges for i in (a, b)})
    # One connected camera group is needed to distinguish a local swap from relabeling all hands.
    reached = {active[0]}
    for _ in active:
        for a, b, _ in edges:
            if a in reached or b in reached:
                reached.update((a, b))
    if set(active) != reached:
        report['status'] = 'disconnected_camera_groups'
        return cameras, annotation, report
    variants = []
    for active_bits in itertools.product((0, 1), repeat=len(active)):
        bits = [0]*len(names)
        for i, bit in zip(active, active_bits):
            bits[i] = bit
        if any(e['same']['verified'] and bits[a] != bits[b] and not e['swap']['verified']
               for a, b, e in edges):
            continue  # Never break an already verified original correspondence.
        score = float(np.mean([e['swap' if bits[a] != bits[b] else 'same']['score']
                               for a, b, e in edges]))
        variants.append((score, sum(bits), tuple(bits)))
    best_score = min(v[0] for v in variants)
    best_variants = [v for v in variants if abs(v[0]-best_score) < 1e-8]
    before = float(np.mean([e['same']['score'] for _, _, e in edges]))
    report.update(before_score=before, after_score=best_score)
    if any(v[1] == 0 for v in best_variants):
        return cameras, annotation, report
    if len(active) == 2:
        # Geometry identifies relative correspondence, not which camera's semantic labels are wrong.
        trust = []
        for index in active:
            d = cameras[names[index]]
            values = [float(np.median(d['role_confidence'][hand, d['valid_frame'][hand]]))
                      for hand in range(2) if d['valid_frame'][hand].any()]
            trust.append(float(np.mean(values)) if values else 0.)
        if abs(trust[0]-trust[1]) < settings['min_reference_confidence_margin']:
            report['status'] = 'ambiguous_two_view_role_anchor'
            return cameras, annotation, report
        anchor = active[int(np.argmax(trust))]
        best_variants = [v for v in best_variants if not v[2][anchor]]
    else:
        minimum_changes = min(v[1] for v in best_variants)
        best_variants = [v for v in best_variants if v[1] == minimum_changes]
    if len(best_variants) != 1:
        report['status'] = 'ambiguous_global_assignment'
        return cameras, annotation, report
    best = best_variants[0][2]
    if not any(best):
        return cameras, annotation, report
    drop = before-best_score
    if (drop < settings['min_score_drop']
            or drop/max(before, 1e-8) < settings['min_relative_improvement']):
        report['status'] = 'improvement_too_small'
        return cameras, annotation, report
    # Require separation from other relative assignments, not just the all-labels-inverted equivalent.
    complement = tuple(1-best[i] if i in active else 0 for i in range(len(names)))
    alternatives = [v[0] for v in variants if v[2] not in (best, complement)]
    if alternatives and min(alternatives)-best_score < settings['min_score_drop']:
        report['status'] = 'ambiguous_global_assignment'
        return cameras, annotation, report
    for index, swapped in enumerate(best):
        if not swapped:
            continue
        supporting = [e for a, b, e in edges if index in (a, b) and best[a] != best[b]
                      and e['swap']['verified'] and e['same']['score']-e['swap']['score'] >= settings['min_score_drop']]
        if not supporting:
            report['status'] = 'no_verified_support_for_swap'
            return cameras, annotation, report
    corrected = {c: {k: np.array(v, copy=True) for k, v in d.items()} for c, d in cameras.items()}
    resolved = copy.deepcopy(annotation)
    for index, name in enumerate(names):
        corrected[name]['identity_verified_frame'] = np.zeros_like(corrected[name]['valid_frame'])
        if best[index]:
            for field in HAND_FIELDS:
                corrected[name][field] = corrected[name][field][::-1].copy()
            labels = list(corrected[name]['hands'])
            mapping = dict(zip(labels, labels[::-1]))
            for camera in resolved['cameras']:
                if camera['camera'] == name:
                    for segment in camera['segments']:
                        segment['hand_id'] = mapping.get(segment['hand_id'], segment['hand_id'])
            report['swapped_cameras'].append(name)
    # Low-confidence observations can be retained only at individually verified paired source times.
    for a, b, evidence in edges:
        if not evidence['swap' if best[a] != best[b] else 'same']['verified']:
            continue
        first, second = corrected[names[a]], corrected[names[b]]
        matrix = fundamental((first['intrinsics'], first['T_base_from_camera']),
                             (second['intrinsics'], second['T_base_from_camera']))
        for hand in range(2):
            ia, ib = evidence['rows'].T
            distance = np.linalg.norm(first['points_base_m'][hand, ia, 0]-second['points_base_m'][hand, ib, 0], axis=-1)
            error = np.median(epipolar_distance(first['joints_image_px'][hand, ia],
                                               second['joints_image_px'][hand, ib], matrix), axis=-1)
            good = (distance <= settings['max_wrist_distance_m']) & (error <= settings['max_epipolar_px'])
            if best[a]:
                first['identity_verified_frame'][hand, ia[good]] = True
            if best[b]:
                second['identity_verified_frame'][hand, ib[good]] = True
    report['status'] = 'corrected'
    return corrected, resolved, report


def finalize_camera(data, confidence_threshold):
    """Return task hands and restore the original confidence policy after identity checks."""
    verified = data.get('identity_verified_frame', np.zeros_like(data['valid_frame']))
    accepted = (data['role_confidence'] >= confidence_threshold) | verified
    data['valid_frame'] &= accepted
    data['observation_index'][~accepted] = -1
    for key in ('points_base_m', 'joints_camera_m', 'joints_image_px', 'joints_root_m'):
        data[key][~accepted] = np.nan
    keep = np.flatnonzero(np.isin(data['roles'], ('main', 'support')))
    for key in (*HAND_FIELDS, 'hands', 'roles', 'identity_verified_frame'):
        if key in data:
            data[key] = data[key][keep]
    return data
