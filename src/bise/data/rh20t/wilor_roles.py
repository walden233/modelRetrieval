"""Scene-level hand-role annotation using existing credential/accounting infrastructure."""

import base64
import copy
import itertools
import json
import time
import uuid

import cv2
import numpy as np

from bise.common.paths import resolve_path
from bise.modalities.semantic.annotation import response_content, send_request
from bise.modalities.semantic.credentials import NoActiveKeys, provider_key, redact
from bise.modalities.semantic.store import (
    RequestStore, account_usage, file_lock, item_key, write_json,
)
from .wilor import draw_hands


def load_scene(folder):
    return {p.stem: dict(np.load(p, allow_pickle=False)) for p in sorted(folder.glob('*.npz'))}


def role_context(saved):
    origin = min(int(d['timestamp_ms'][0]) for d in saved.values())
    cameras = {}
    for camera, data in saved.items():
        frames = np.repeat(np.arange(len(data['timestamp_ms'])), np.diff(data['frame_ptr']))
        tracks = []
        for identity in np.unique(data['track_id']):
            rows = np.flatnonzero(data['track_id'] == identity)
            times = (data['timestamp_ms'][frames[rows]] - origin) / 1000
            tracks.append({'track_id': int(identity), 'observations': len(rows),
                           'start_s': float(times.min()), 'end_s': float(times.max())})
        cameras[camera] = {'tracks': tracks,
                           'duration_s': float((data['timestamp_ms'][-1]-origin)/1000)}
    return {'origin_timestamp_ms': origin, 'cameras': cameras}


def validate_roles(content, context):
    data = json.loads(content)
    if not isinstance(data.get('task'), str) or not data['task'].strip():
        raise ValueError('task must be a nonempty description')
    hands = data['hands']
    roles = {h['id']: h['role'] for h in hands}
    if (len(roles) != len(hands) or any(not isinstance(k, str) or not k for k in roles)
            or not set(roles.values()) <= {'main', 'support', 'idle', 'unknown'}):
        raise ValueError('Invalid physical hand IDs or roles')
    main = [identity for identity, role in roles.items() if role == 'main']
    if len(main) > 1 or data.get('main_hand') != (main[0] if main else None):
        raise ValueError('main_hand must match the single main role, or null')
    cameras = data['cameras']
    if len(cameras) != len(context['cameras']) or {
            row['camera'] for row in cameras} != set(context['cameras']):
        raise ValueError('Return each supplied camera exactly once')
    for camera in cameras:
        available = {r['track_id']: r for r in context['cameras'][camera['camera']]['tracks']}
        for seg in camera['segments']:
            if type(seg['track_id']) is not int or seg['track_id'] not in available:
                raise ValueError('Unknown track ID in ' + camera['camera'])
            if seg['hand_id'] not in roles or seg['confidence'] not in ('high', 'medium', 'low'):
                raise ValueError('Unknown hand ID or confidence')
            a, b = seg['start_s'], seg['end_s']
            track = available[seg['track_id']]
            if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float))
                    or not isinstance(b, (int, float)) or not np.isfinite([a, b]).all()
                    or a < 0 or b < a or a > track['end_s']+.15 or b < track['start_s']-.15
                    or b > context['cameras'][camera['camera']]['duration_s']+.15):
                raise ValueError('Invalid source-time interval in ' + camera['camera'])
        for first, second in itertools.combinations(camera['segments'], 2):
            overlap = min(first['end_s'], second['end_s']) - max(first['start_s'], second['start_s'])
            if overlap > .001 and (first['track_id'] == second['track_id'] or
                                   first['hand_id'] == second['hand_id']):
                raise ValueError('Conflicting overlapping assignments in ' + camera['camera'])
    return data


def sample_indices(data, count, maximum):
    """Uniform coverage plus representative observations of otherwise unseen fragments."""
    indices = set(np.linspace(0, len(data['frame_index'])-1,
                              min(count, len(data['frame_index'])), dtype=int).tolist())
    frames = np.repeat(np.arange(len(data['frame_index'])), np.diff(data['frame_ptr']))
    for track in sorted(np.unique(data['track_id']),
                        key=lambda t: -(data['track_id'] == t).sum()):
        rows = np.flatnonzero(data['track_id'] == track)
        if len(indices) >= maximum:
            break
        if not set(frames[rows]) & indices:
            indices.add(int(frames[rows[len(rows)//2]]))
    return sorted(indices)


def prepare_images(saved, folder, settings, context):
    content, metadata = [], {}
    for camera, data in saved.items():
        selected = sample_indices(data, settings['frames_per_camera'], settings['max_frames_per_camera'])
        cap = cv2.VideoCapture(str(data['video_path']))
        previews, samples = [], []
        try:
            for index in selected:
                frame = int(data['frame_index'][index])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
                if not ok:
                    raise OSError(f'Cannot read {camera} frame {frame}')
                a, b = data['frame_ptr'][index:index+2]
                detections = [{k: data[k][i] for k in
                               ('track_id', 'boxes', 'scores', 'joints_image_px')} for i in range(a, b)]
                image = draw_hands(image, detections)
                seconds = float((data['timestamp_ms'][index]-context['origin_timestamp_ms'])/1000)
                label = f'{camera}  scene {seconds:.3f}s  f{frame}'
                cv2.rectangle(image, (0, 0), (image.shape[1], 23), (0, 0, 0), -1)
                cv2.putText(image, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
                ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise OSError('JPEG encoding failed')
                url = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()
                content += [{'type': 'text', 'text': label},
                            {'type': 'image_url', 'image_url': {'url': url}}]
                samples.append({'source_frame': frame, 'scene_seconds': seconds,
                                'track_ids': data['track_id'][a:b].astype(int).tolist()})
                previews.append(cv2.resize(image, (320, 180)))
        finally:
            cap.release()
        padding = (-len(previews)) % 4
        previews += [np.zeros_like(previews[0]) for _ in range(padding)]
        sheet = np.concatenate([np.concatenate(previews[i:i+4], axis=1)
                                for i in range(0, len(previews), 4)])
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f'{camera}.jpg'), sheet)
        metadata[camera] = samples
    return content, metadata


def annotate_roles(root, config, scenes, overwrite=False, sender=send_request):
    settings = config['roles']
    semantic = json.loads(resolve_path(settings['provider_config']).read_text())
    provider = semantic['providers'][settings['provider']]
    if 'api_key' in provider:
        raise ValueError('Use the managed key pool or environment, not a literal api_key')
    prompt = resolve_path(settings['prompt']).read_text()
    signature = {'prompt': prompt, 'provider': {k: v for k, v in provider.items()
                                               if k not in ('key_pool', 'api_key_env')},
                 'frames_per_camera': settings['frames_per_camera'],
                 'max_frames_per_camera': settings['max_frames_per_camera']}
    directory = root / 'roles'
    directory.mkdir(parents=True, exist_ok=True)
    params = directory / 'params.json'
    if params.exists() and json.loads(params.read_text()) != signature:
        raise ValueError('Role annotation settings changed; use a new output directory')
    write_json(params, signature)
    store = RequestStore(directory)
    with file_lock(directory / '.lock'):
        store.recover(lambda raw, record, frames: validate_roles(response_content(raw), frames['context']))
        existing = store.annotations()
        failures = []
        execution_id = uuid.uuid4().hex
        started = time.perf_counter()
        store.event({'event': 'execution_start', 'execution_id': execution_id,
                     'selected': len(scenes), 'concurrency': 1})
        for number, identity in enumerate(scenes, 1):
            folder = root / identity
            saved = load_scene(folder)
            if not saved or not all(bool(d['complete']) for d in saved.values()):
                raise ValueError(f'Complete per-camera extraction required: {identity}')
            context = role_context(saved)
            record = {'sample_id': identity, 'source': {
                camera: {'size': (folder / f'{camera}.npz').stat().st_size,
                         'mtime_ns': (folder / f'{camera}.npz').stat().st_mtime_ns}
                for camera in saved}}
            if not overwrite and item_key(record) in existing:
                print(f'ROLES {number}/{len(scenes)} SKIP {identity}', flush=True)
                continue
            provider_key(provider)
            images, samples = prepare_images(saved, folder / 'roles', settings, context)
            frame_info = {'context': context, 'samples': samples}
            body = {'model': provider['model'], 'stream': False, **provider.get('request', {}),
                    'messages': [{'role': 'user', 'content': [
                        {'type': 'text', 'text': prompt + '\nScene context:\n' + json.dumps(context)}, *images]}]}
            previous_errors = []
            for attempt in range(1, settings['max_attempts']+1):
                attempt_id = uuid.uuid4().hex
                store.event({'event': 'started', 'attempt_id': attempt_id, 'execution_id': execution_id,
                             'attempt': attempt, 'item_key': item_key(record), 'sample_id': identity})
                request_start = time.perf_counter()
                try:
                    request_body = copy.deepcopy(body)
                    if previous_errors:
                        request_body['messages'].append({'role': 'user', 'content':
                            'Return complete valid JSON. Correct these errors: ' + '; '.join(previous_errors)})
                    if provider.get('send_request_id'):
                        request_body['request_id'] = attempt_id
                    raw = sender(provider, request_body, settings['timeout_seconds'])
                    store.event({'event': 'response', 'attempt_id': attempt_id,
                                 'execution_id': execution_id, 'record': record, 'frames': frame_info,
                                 'raw_response': raw, 'accounting': account_usage(raw, provider.get('price', {})),
                                 'timings': {'request_seconds': time.perf_counter()-request_start}})
                    try:
                        result = validate_roles(response_content(raw), context)
                    except (ValueError, KeyError, TypeError) as exc:
                        store.event({'event': 'validation', 'attempt_id': attempt_id,
                                     'valid': False, 'error': str(exc)})
                        previous_errors.append(str(exc))
                        if attempt == settings['max_attempts']:
                            failures.append(identity)
                            print('ROLES FAILED', identity, 'invalid schema', flush=True)
                        continue
                    store.accept(record, result, attempt_id, frame_info)
                    store.event({'event': 'validation', 'attempt_id': attempt_id, 'valid': True})
                    print(f'ROLES {number}/{len(scenes)} OK {identity} '
                          f'{time.perf_counter()-request_start:.1f}s', flush=True)
                    break
                except NoActiveKeys:
                    raise
                except Exception as exc:
                    store.event({'event': 'error', 'attempt_id': attempt_id, 'execution_id': execution_id,
                                 'sample_id': identity, 'error': redact(provider, str(exc)),
                                 'timings': {'request_seconds': time.perf_counter()-request_start}})
                    if attempt == settings['max_attempts']:
                        failures.append(identity)
                        print('ROLES FAILED', identity, type(exc).__name__, flush=True)
                    else:
                        time.sleep(min(2**attempt, 10))
        store.event({'event': 'execution_end', 'execution_id': execution_id,
                     'elapsed_seconds': time.perf_counter()-started, 'failed': failures})
        summary = store.summarize()
        summary['failed_scenes'] = failures
        write_json(directory / 'summary.json', summary)
    if failures:
        raise RuntimeError('Role annotation failed; rerun --stage roles: ' + ', '.join(failures))
    return store.annotations()


def read_roles(root):
    return {row['sample_id']: row for row in RequestStore(root / 'roles').annotations().values()}
