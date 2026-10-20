"""WiLoR extraction, quality, GLM roles and gated trajectory reconstruction."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from bise.common.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hand/wilor.json')
    parser.add_argument('--stage', choices=['all', 'extract', 'quality', 'roles', 'build', 'compare'], default='all')
    parser.add_argument('--scene', nargs='+', help='cfg2/task_XXXX/scene_X; overrides config selection')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-frames', type=int, help='Prefix smoke test only; does not overwrite complete results')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.scene:
        config['scenes'] = args.scene
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if config['frame_stride'] < 1 or config['batch_size'] < 1:
        parser.error('frame_stride and batch_size must be positive')
    root, dataset = resolve_path(config['output_dir']), resolve_path(config['dataset_root'])
    if root.resolve().is_relative_to(dataset.resolve()):
        parser.error('Output must be outside the source dataset')
    for identity in config['scenes']:
        parts = identity.split('/')
        if (len(parts) != 3 or parts[0] not in ('cfg2', 'cfg3')
                or not parts[1].startswith('task_') or not parts[2].startswith('scene_')
                or any(part in ('.', '..') for part in parts)):
            parser.error(f'Invalid scene: {identity}')
    if args.stage in ('all', 'extract'):
        from bise.data.rh20t.wilor import WiLoRBackend, extract_video
        import numpy as np
        import torch
        root.mkdir(parents=True, exist_ok=True)
        snapshot = {**config, 'max_frames': args.max_frames,
                    'repo_revision': subprocess.check_output(
                        ['git', '-C', str(resolve_path(config['repo_dir'])), 'rev-parse', 'HEAD'], text=True).strip()}
        previous = root / 'params.json'
        extraction_keys = set(config) - {'roles', 'trajectory', 'comparison'}
        old = json.loads(previous.read_text()) if previous.exists() else {}
        if old and any(old.get(k) != snapshot.get(k) for k in extraction_keys | {'max_frames', 'repo_revision'}):
            raise ValueError('Extraction settings differ; choose another --output-dir')
        previous.write_text(json.dumps(snapshot, indent=2)+'\n')
        backend = None
        started = time.monotonic()
        for identity in config['scenes']:
            cfg, task, scene = identity.split('/')
            source = dataset / f'RH20T_{cfg}' / task / scene
            videos = sorted(source.glob('cam_*_human.mp4'))
            if not videos:
                raise FileNotFoundError(source)
            output = root / identity
            output.mkdir(parents=True, exist_ok=True)
            for video in videos:
                camera = video.name[4:-len('_human.mp4')]
                target = output / f'{camera}.npz'
                if target.exists() and not args.overwrite:
                    with np.load(target, allow_pickle=False) as data:
                        complete = bool(data['complete'])
                    if complete or args.max_frames is not None:
                        print('SKIP', target, flush=True)
                        continue
                if backend is None:
                    backend = WiLoRBackend(config)
                print('EXTRACT', identity, camera, flush=True)
                print(extract_video(backend, source, camera, output, config, args.max_frames), flush=True)
        print('Extraction elapsed seconds:', round(time.monotonic()-started, 2), flush=True)
        if torch.cuda.is_available() and backend is not None:
            print('GPU peak allocated GiB:', round(torch.cuda.max_memory_allocated()/1024**3, 3), flush=True)
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage in ('all', 'quality'):
        from bise.data.rh20t.wilor_quality import analyze
        cameras, tracks, views = analyze(root, dataset)
        print(f'Quality: {len(cameras)} cameras, {len(tracks)} tracks, {len(views)} camera pairs; {root / "report.md"}')
    if args.stage in ('all', 'roles'):
        from bise.data.rh20t.wilor_roles import annotate_roles
        annotate_roles(root, config, config['scenes'], overwrite=args.overwrite)
    if args.stage in ('all', 'build'):
        from bise.data.rh20t.wilor_trajectory import build
        build(root, dataset, config['scenes'], config['trajectory'])
    if args.stage in ('all', 'compare'):
        from bise.data.rh20t.wilor_compare import compare
        compare(root, config['scenes'], config['comparison'])


if __name__ == '__main__':
    main()
