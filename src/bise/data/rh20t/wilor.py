"""Independent WiLoR inference outputs; never replace the dataset hand files."""

import inspect
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from bise.common.paths import resolve_path


EDGES = [(0, start) for start in (1, 5, 9, 13, 17)] + [
    (start + j, start + j + 1) for start in (1, 5, 9, 13, 17) for j in range(3)
]


class HandTracker:
    """Conservative per-camera association; IDs are not anatomical side or task role."""

    def __init__(self, max_gap_seconds=0.5, max_wrist_distance=1.5):
        self.max_gap = max_gap_seconds
        self.max_distance = max_wrist_distance
        self.tracks = {}
        self.next_id = 0

    def update(self, boxes, joints, right, timestamp_ms):
        ids = np.full(len(boxes), -1, dtype=np.int32)
        ambiguous = np.zeros(len(boxes), dtype=bool)
        active = [i for i, x in self.tracks.items()
                  if 0 <= timestamp_ms - x['time'] <= 1000 * self.max_gap]
        if active and len(boxes):
            cost = np.empty((len(active), len(boxes)))
            for row, track_id in enumerate(active):
                previous = self.tracks[track_id]
                for col, box in enumerate(boxes):
                    size = max(np.linalg.norm(box[2:] - box[:2]), previous['size'], 1)
                    distance = np.linalg.norm(joints[col, 0] - previous['wrist']) / size
                    cost[row, col] = distance + .15 * (right[col] != previous['right'])
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] <= self.max_distance:
                    ids[col] = active[row]
                    alternatives = np.delete(cost[:, col], row)
                    ambiguous[col] = bool(len(alternatives) and alternatives.min() - cost[row, col] < .1)
        for j, box in enumerate(boxes):
            if ids[j] < 0:
                ids[j] = self.next_id
                self.next_id += 1
            self.tracks[int(ids[j])] = {
                'time': timestamp_ms, 'wrist': joints[j, 0].copy(),
                'size': np.linalg.norm(box[2:] - box[:2]), 'right': right[j],
            }
        return ids, ambiguous


class WiLoRBackend:
    def __init__(self, config):
        repo, assets = resolve_path(config['repo_dir']), resolve_path(config['model_dir'])
        required = ['wilor_final.ckpt', 'detector.pt', 'model_config.yaml',
                    'mano/MANO_RIGHT.pkl', 'mano_mean_params.npz']
        for name in required:
            if not (assets / name).is_file():
                raise FileNotFoundError(assets / name)
        sys.path.insert(0, str(repo))
        # Compatibility for the official legacy MANO pickle in this isolated process.
        for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                            'object': object, 'str': str, 'unicode': str}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        if not hasattr(inspect, 'getargspec'):
            inspect.getargspec = inspect.getfullargspec
        from wilor.configs import get_config
        from wilor.models import WiLoR
        from ultralytics import YOLO
        from ultralytics.nn import tasks

        self.device = torch.device(config['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.fp16 = config.get('fp16', True) and self.device.type == 'cuda'
        cfg = get_config(str(assets / 'model_config.yaml'))
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
        cfg.MANO.DATA_DIR = str(assets)
        cfg.MANO.MODEL_PATH = str(assets / 'mano')
        cfg.MANO.MEAN_PARAMS = str(assets / 'mano_mean_params.npz')
        cfg.freeze()
        self.model = WiLoR(cfg, init_renderer=False)
        state = torch.load(assets / 'wilor_final.ckpt', map_location='cpu', weights_only=False)
        missing, unexpected = self.model.load_state_dict(state['state_dict'], strict=False)
        if any(k.startswith(('backbone.', 'refine_net.')) for k in missing):
            raise ValueError(f'Missing reconstruction weights: {missing}')
        del state
        self.model.eval().requires_grad_(False).to(self.device)
        if self.fp16:
            self.model.half()
        # Ultralytics 8.1 predates Torch 2.6's weights_only default; load this official asset explicitly.
        previous = tasks.torch_safe_load
        try:
            tasks.torch_safe_load = lambda file: (
                torch.load(file, map_location='cpu', weights_only=False), str(file))
            self.detector = YOLO(str(assets / 'detector.pt')).to(self.device)
        finally:
            tasks.torch_safe_load = previous
        self.cfg, self.config = cfg, config
        self.load_info = {'missing_keys': missing, 'unexpected_keys': unexpected}

    @torch.inference_mode()
    def predict(self, image):
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils.renderer import cam_crop_to_full
        prediction = self.detector(image, conf=self.config['detection_confidence'],
                                   verbose=False, device=str(self.device))[0]
        boxes = prediction.boxes.xyxy.cpu().numpy()
        right = prediction.boxes.cls.cpu().numpy().astype(np.int32)
        scores = prediction.boxes.conf.cpu().numpy()
        if not len(boxes):
            return []
        source = ViTDetDataset(self.cfg, image, boxes, right, rescale_factor=2.0,
                               fp16=self.fp16)
        loader = torch.utils.data.DataLoader(source, batch_size=self.config['batch_size'])
        results = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(batch)
            cam = output['pred_cam'].float().clone()
            cam[:, 1] *= 2 * batch['right'] - 1
            focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(image.shape[:2])
            translation = cam_crop_to_full(cam, batch['box_center'].float(),
                                           batch['box_size'].float(), batch['img_size'].float(), focal)
            for n in range(len(cam)):
                index = int(batch['personid'][n])
                xyz = output['pred_keypoints_3d'][n].float().cpu().numpy().copy()
                xyz[:, 0] *= 2 * right[index] - 1
                trans = translation[n].cpu().numpy()
                camera = xyz + trans
                pixel = camera[:, :2] / camera[:, 2:] * focal
                pixel += np.array([image.shape[1], image.shape[0]]) / 2
                results.append({
                    'boxes': boxes[index], 'right': right[index], 'scores': scores[index],
                    'joints_image_px': pixel.astype(np.float32),
                    'joints_root_m': (xyz - xyz[0]).astype(np.float32),
                    'joints_camera_estimate_m': camera.astype(np.float32),
                    'model_focal_px': np.float32(focal),
                    'mano_shape': output['pred_mano_params']['betas'][n].float().cpu().numpy(),
                    'mano_pose': output['pred_mano_params']['hand_pose'][n].float().cpu().numpy(),
                    'mano_orientation': output['pred_mano_params']['global_orient'][n].float().cpu().numpy(),
                })
        return results


def draw_hands(image, detections):
    image = image.copy()
    colors = [(40, 210, 60), (230, 110, 30), (40, 100, 240), (210, 50, 210)]
    for d in detections:
        color = colors[int(d['track_id']) % len(colors)]
        x1, y1, x2, y2 = d['boxes'].astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, f"H{int(d['track_id'])} {d['scores']:.2f}",
                    (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        points = d['joints_image_px']
        if np.isfinite(points).all():
            points = np.clip(points, -10000, 10000).astype(int)
            for a, b in EDGES:
                cv2.line(image, tuple(points[a]), tuple(points[b]), color, 1)
            for point in points:
                cv2.circle(image, tuple(point), 2, color, -1)
    return image


def extract_video(backend, source, camera, output, config, max_frames=None):
    video = source / f'cam_{camera}_human.mp4'
    destination = output / f'{camera}.npz'
    timestamps = np.load(source / f'cam_{camera}_human_timestamps.npy', allow_pickle=True).item()
    timestamps = np.asarray(timestamps['color'], dtype=np.int64)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(video)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count != len(timestamps):
        raise ValueError(f'RGB/timestamp count mismatch: {video}')
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(str(output / f'{camera}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps / config['frame_stride'], (width, height))
    if not writer.isOpened():
        raise OSError('Cannot create overlay video')
    tracker = HandTracker(**config.get('tracking', {}))
    rows, indices, times, ptr, previews = [], [], [], [0], []
    preview_indices = set(np.linspace(0, min(count, max_frames or count) - 1, 6).astype(int))
    start = time.monotonic()
    try:
        for frame_index in range(count):
            ok, image = cap.read()
            if not ok:
                raise OSError(f'Failed to decode {video} frame {frame_index}')
            if max_frames is not None and frame_index >= max_frames:
                break
            if frame_index % config['frame_stride']:
                continue
            detections = backend.predict(image)
            if detections:
                ids, ambiguous = tracker.update(
                    np.stack([d['boxes'] for d in detections]),
                    np.stack([d['joints_image_px'] for d in detections]),
                    np.asarray([d['right'] for d in detections]), int(timestamps[frame_index]))
                for d, track_id, uncertain in zip(detections, ids, ambiguous):
                    d.update(track_id=track_id, association_ambiguous=uncertain)
                    rows.append(d)
            indices.append(frame_index)
            times.append(timestamps[frame_index])
            ptr.append(len(rows))
            annotated = draw_hands(image, detections)
            cv2.putText(annotated, f'{camera} f{frame_index} source {(times[-1]-timestamps[0])/1000:.2f}s',
                        (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            writer.write(annotated)
            if frame_index in preview_indices:
                previews.append(annotated)
            if len(indices) % 25 == 0:
                print(f'{source.parent.name}/{source.name}/{camera}: {frame_index+1}/{count} '
                      f'frames, {len(rows)} hands, {time.monotonic()-start:.1f}s', flush=True)
    finally:
        cap.release()
        writer.release()
    shapes = {'boxes': (4,), 'right': (), 'scores': (), 'joints_image_px': (21, 2),
              'joints_root_m': (21, 3), 'joints_camera_estimate_m': (21, 3),
              'model_focal_px': (), 'mano_shape': (10,), 'mano_pose': (15, 3, 3),
              'mano_orientation': (1, 3, 3), 'track_id': (), 'association_ambiguous': ()}
    arrays = {key: (np.stack([row[key] for row in rows]) if rows else
                    np.empty((0, *shape), dtype=np.float32)) for key, shape in shapes.items()}
    arrays.update(frame_index=np.asarray(indices), timestamp_ms=np.asarray(times), frame_ptr=np.asarray(ptr),
                  video_path=np.asarray(str(video.resolve())), image_size_wh=np.asarray([width, height]),
                  elapsed_seconds=np.asarray(time.monotonic()-start),
                  complete=np.asarray(max_frames is None or max_frames >= count),
                  coordinate_note=np.asarray('camera_estimate uses the model assumed focal; NOT calibrated robot-base coordinates'))
    temp = destination.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(destination)
    if previews:
        cv2.imwrite(str(output / f'{camera}.jpg'), np.concatenate(previews, axis=1))
    return {'camera': camera, 'frames': len(indices), 'detections': len(rows),
            'seconds': round(time.monotonic()-start, 2)}
