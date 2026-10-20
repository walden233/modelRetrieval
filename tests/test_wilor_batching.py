"""Cross-frame crop ownership, empty frames and source-order batching."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from bise.data.rh20t.wilor import WiLoRBackend, frame_predictions


def test_batched_reconstruction_matches_single_frames_with_local_hand_indices(monkeypatch):
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, cfg, image, boxes, right, **kwargs):
            self.image, self.boxes, self.right = image, boxes, right

        def __len__(self):
            return len(self.boxes)

        def __getitem__(self, index):
            return {'img': torch.tensor([self.boxes[index, 0]]), 'personid': index,
                    'right': self.right[index], 'box_center': self.boxes[index, :2],
                    'box_size': 10., 'img_size': np.array(self.image.shape[1::-1])}

    dataset = ModuleType('wilor.datasets.vitdet_dataset')
    dataset.ViTDetDataset = Dataset
    monkeypatch.setitem(sys.modules, dataset.__name__, dataset)
    renderer = ModuleType('wilor.utils.renderer')
    renderer.cam_crop_to_full = lambda cam, center, size, image, focal: torch.zeros_like(cam)
    monkeypatch.setitem(sys.modules, renderer.__name__, renderer)

    class Model:
        def __init__(self):
            self.calls = []

        def __call__(self, batch):
            count = len(batch['img'])
            self.calls.append(count)
            points = batch['img'][:, None, :].expand(count, 21, 3).clone()
            points[:, :, 2] = 1
            return {'pred_cam': torch.ones(count, 3), 'pred_keypoints_3d': points,
                    'pred_mano_params': {'betas': torch.zeros(count, 10),
                                        'hand_pose': torch.eye(3).repeat(count, 15, 1, 1),
                                        'global_orient': torch.eye(3).repeat(count, 1, 1, 1)}}

    backend = WiLoRBackend.__new__(WiLoRBackend)
    backend.device, backend.fp16 = torch.device('cpu'), False
    backend.cfg = SimpleNamespace(EXTRA=SimpleNamespace(FOCAL_LENGTH=100.),
                                  MODEL=SimpleNamespace(IMAGE_SIZE=10))
    backend.config = {'batch_size': 4, 'detection_confidence': .3}
    backend.model = Model()
    calls = []

    def detector(images, **kwargs):
        calls.append(len(images))
        predictions = []
        for image in images:
            frame = int(image[0, 0, 0])
            count = [2, 0, 2, 1][frame]
            boxes = torch.tensor([[frame*10+i+1, 2, 8, 9] for i in range(count)]).reshape(-1, 4)
            predictions.append(SimpleNamespace(boxes=SimpleNamespace(
                xyxy=boxes, cls=torch.arange(count) % 2,
                conf=torch.full((count,), .8+frame*.01))))
        return predictions

    backend.detector = detector
    images = [np.full((20, 20+frame*10, 3), frame, np.uint8) for frame in range(4)]
    batched = backend.predict_many(images)
    assert calls == [4] and backend.model.calls == [4, 1]
    assert [len(row) for row in batched] == [2, 0, 2, 1]
    sequential = [backend.predict(image) for image in images]
    for first, second in zip(batched, sequential):
        for hand_a, hand_b in zip(first, second):
            for key in hand_a:
                np.testing.assert_allclose(hand_a[key], hand_b[key])
    assert batched[2][0]['boxes'][0] == 21
    assert batched[2][0]['model_focal_px'] == 400
    assert backend.predict_many([]) == []


def test_frame_batching_preserves_stride_prefix_and_short_tail():
    class Capture:
        index = 0

        def read(self):
            image = np.array([self.index])
            self.index += 1
            return True, image

    class Backend:
        batches = []

        def predict_many(self, images):
            self.batches.append(len(images))
            return [[int(image[0])] for image in images]

    cap, backend = Capture(), Backend()
    rows = list(frame_predictions(backend, cap, 20, {'frame_stride': 2, 'frame_batch_size': 4}, 11))
    assert backend.batches == [4, 2]
    assert [row[0] for row in rows] == [0, 2, 4, 6, 8, 10]
    assert [row[2] for row in rows] == [[i] for i in [0, 2, 4, 6, 8, 10]]
    assert cap.index == 11
