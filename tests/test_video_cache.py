"""Cache hits bypass decoding while preserving pixels, sampling and augmentation."""

import os
import numpy as np
import torch
from transformers import VideoMAEImageProcessor

from bise.data.rh20t import video_dataset
from bise.data.rh20t.video_dataset import RH20TVideoDataset
from bise.modalities.video.cache import VideoPixelCache


def test_cache_hit_skips_decode_and_matches_uncached_pixels(tmp_path, monkeypatch):
    scene = tmp_path / 'data/task_0001/scene_1'
    scene.mkdir(parents=True)
    for domain in ('human', 'robot'):
        (scene / f'cam_a_{domain}.mp4').write_bytes(b'fixture')
    calls = []
    def frames(path, num_frames, **kwargs):
        calls.append(str(path))
        return [np.full((8, 8, 3), i * 17, dtype=np.uint8) for i in range(num_frames)]
    monkeypatch.setattr(video_dataset, 'sample_video_frames', frames)
    processor = VideoMAEImageProcessor(do_resize=False, do_center_crop=False)
    kwargs = dict(root_dir=str(tmp_path / 'data'), processor=processor, num_frames=4,
                  sampling_strategy='uniform', deterministic=False)
    uncached = RH20TVideoDataset(**kwargs)[0]
    cached = RH20TVideoDataset(**kwargs, cache_dir=str(tmp_path / 'cache'))
    first = cached[0]
    assert len(calls) == 4
    def no_decode(*args, **kwargs):
        raise AssertionError('A cache hit must not decode')
    monkeypatch.setattr(video_dataset, 'sample_video_frames', no_decode)
    for seed in (53, 54, 55):
        reused = RH20TVideoDataset(**kwargs, seed=seed, cache_dir=str(tmp_path / 'cache'))[0]
        for key in ('human_pixel_values', 'robot_pixel_values'):
            torch.testing.assert_close(reused[key], uncached[key], rtol=0, atol=0)
            torch.testing.assert_close(reused[key], first[key], rtol=0, atol=0)
    assert len(list((tmp_path / 'cache').rglob('*.npz'))) == 1
    random_source = RH20TVideoDataset(**{**kwargs, 'sampling_strategy': 'segment_random'},
                                      cache_dir=str(tmp_path / 'cache'))
    assert random_source.pixel_cache is None


def test_cache_invalidation_and_incomplete_entry_recovery(tmp_path):
    files = [tmp_path / 'a.mp4', tmp_path / 'b.mp4']
    for path in files: path.write_bytes(b'video')
    processor = VideoMAEImageProcessor(do_resize=False, do_center_crop=False)
    cache = VideoPixelCache(tmp_path / 'cache', processor, 4, 'uniform', None)
    path = cache.path(files, None)
    pixels = torch.randn(2, 4, 3, 8, 8)
    cache.save(path, pixels)
    torch.testing.assert_close(cache.load(path), pixels, rtol=0, atol=0)
    assert cache.path(files, 99) == path
    assert VideoPixelCache(tmp_path / 'cache', processor, 8, 'uniform', None).path(files, None) != path
    changed = VideoMAEImageProcessor(do_resize=False, do_center_crop=False, do_normalize=False)
    assert VideoPixelCache(tmp_path / 'cache', changed, 4, 'uniform', None).path(files, None) != path
    before = files[0].stat()
    os.utime(files[0], ns=(before.st_atime_ns, before.st_mtime_ns + 10_000))
    assert cache.path(files, None) != path
    path.write_bytes(b'interrupted entry')
    assert cache.load(path) is None
    cache.save(path, pixels)
    torch.testing.assert_close(cache.load(path), pixels, rtol=0, atol=0)
    assert not list(path.parent.glob('.*.npz'))
