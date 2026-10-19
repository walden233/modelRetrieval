"""Keep CPU video pixels and ordering identical under spawned loader workers."""

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor

from bise.data.rh20t.video_dataset import RH20TVideoDataset
from bise.modalities.video.loading import video_loader_options


@pytest.mark.parametrize("device_kind", ("cpu", "cuda"))
def test_real_mp4_pixels_match_between_serial_and_spawned_workers(tmp_path, device_kind):
    if device_kind == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    scene = tmp_path / 'task_0001' / 'scene_1'
    scene.mkdir(parents=True)
    for camera in ('a', 'b'):
        for domain in ('human', 'robot'):
            writer = cv2.VideoWriter(str(scene / f'cam_{camera}_{domain}.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 8, (32, 32))
            assert writer.isOpened()
            for i in range(8):
                writer.write(np.full((32, 32, 3), i * 20, dtype=np.uint8))
            writer.release()
    processor = VideoMAEImageProcessor(do_resize=False, do_center_crop=False)
    source = RH20TVideoDataset(str(tmp_path), processor, num_frames=4,
                               deterministic=True, decoder_threads=1)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        serial = next(iter(DataLoader(source, batch_size=2, **video_loader_options(
            {'num_workers': 0}, torch.device('cpu')))))
        parallel = DataLoader(source, batch_size=2, **video_loader_options(
            {'num_workers': 2, 'prefetch_factor': 1}, torch.device(device_kind)))
        for _ in range(2):
            batch = next(iter(parallel))
            assert batch['sample_ids'] == serial['sample_ids']
            if device_kind == "cuda":
                frames = batch['human_pixel_values']
                assert frames.device.type == "cpu" and frames.is_pinned()
                copied = frames.to('cuda', non_blocking=True).cpu()
                torch.testing.assert_close(copied, serial['human_pixel_values'], rtol=0, atol=0)
            for key in ('human_pixel_values', 'robot_pixel_values',
                        'task_indices', 'scene_indices'):
                torch.testing.assert_close(batch[key], serial[key], rtol=0, atol=0)
        del parallel
    finally:
        torch.set_num_threads(threads)
