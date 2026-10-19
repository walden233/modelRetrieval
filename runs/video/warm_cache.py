"""Prepare lossless video pixels on CPU without constructing a backbone or training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoVideoProcessor, VideoMAEImageProcessor

from bise.common import load_json_config
from bise.common.evaluation import effective_config
from bise.modalities.video.factory import build_video_dataset
from bise.modalities.video.loading import video_loader_options
from bise.modalities.video.models.backbone_registry import backbone_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Prepared run params.json with bound manifests')
    parser.add_argument('--cfg', nargs='+', choices=('cfg2', 'cfg3'), default=['cfg2', 'cfg3'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-dir', default='artifacts/cache/video')
    args = parser.parse_args()
    config = load_json_config(args.config)
    model = config['model']
    processor_type = (VideoMAEImageProcessor if model['backbone_type'] == 'videomae'
                      else AutoVideoProcessor)
    processor = processor_type.from_pretrained(backbone_source(model), local_files_only=True)
    for cfg in args.cfg:
        current = effective_config(config, 'video', cfg)
        current['dataset']['cache_dir'] = args.cache_dir
        source = build_video_dataset(current['dataset'], processor, is_train=False,
                                     metadata_only=True)
        if source.pixel_cache is None:
            raise ValueError('Cache preparation requires deterministic sampling')
        loader = DataLoader(source, batch_size=1, shuffle=False, **video_loader_options(
            {'num_workers': args.workers}, torch.device('cpu')))
        for _ in tqdm(loader, desc=f'Cache {cfg}', total=len(loader)):
            pass
        del loader
        print(f'{cfg}: {len(source)} pairs cached at {source.pixel_cache.root}', flush=True)


if __name__ == '__main__':
    main()
