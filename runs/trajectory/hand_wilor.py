"""WiLoR extraction, quality, GLM roles and gated trajectory reconstruction."""

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from bise.common.paths import resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/hand/wilor.json')
    parser.add_argument('--stage', choices=['all', 'extract', 'quality', 'roles', 'build', 'compare'], default='all')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--scene', nargs='+', help='cfg2/task_XXXX/scene_X; overrides config selection')
    selection.add_argument('--all-scenes', action='store_true', help='All dataset scenes in the selected cfgs')
    parser.add_argument('--cfg', nargs='+', choices=['cfg2', 'cfg3'], default=['cfg2', 'cfg3'])
    parser.add_argument('--list-only', action='store_true', help='Check selection and print counts without loading models')
    parser.add_argument('--archive', action='store_true', help='Package the output as .tar.gz after successful extraction/pipeline')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-frames', type=int, help='Prefix smoke test only; does not overwrite complete results')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.scene:
        config['scenes'] = args.scene
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.all_scenes:
        dataset = resolve_path(config['dataset_root'])
        config['scenes'] = [f'{cfg}/{p.parent.name}/{p.name}' for cfg in args.cfg
                            for p in sorted((dataset/f'RH20T_{cfg}').glob('task_*/scene_*'))
                            if any(p.glob('cam_*_human.mp4'))]
        if not config['scenes']:
            parser.error('No human videos found in the selected cfgs')
    if args.archive and (args.stage not in ('all', 'extract') or args.max_frames is not None):
        parser.error('--archive requires complete --stage extract or all, without --max-frames')
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
    if args.list_only:
        counts = {}
        for identity in config['scenes']:
            cfg, task, scene = identity.split('/')
            videos = list((dataset/f'RH20T_{cfg}'/task/scene).glob('cam_*_human.mp4'))
            if not videos:
                raise FileNotFoundError(identity)
            row = counts.setdefault(cfg, {'scenes': 0, 'videos': 0})
            row['scenes'] += 1
            row['videos'] += len(videos)
        print(json.dumps({'selection': counts, 'output_dir': str(root), 'archive': args.archive}, indent=2))
        return
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
        build(root, dataset, config['scenes'], config['trajectory'], config['comparison'])
    if args.stage in ('all', 'compare'):
        from bise.data.rh20t.wilor_compare import compare
        compare(root, config['scenes'], config['comparison'])
    if args.archive:
        destination = root.with_name(root.name+'.tar.gz')
        temporary = destination.with_name(destination.name+'.part')
        print('ARCHIVE', destination, flush=True)
        try:
            with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
                archive.add(root, arcname=root.name,
                            filter=lambda item: None if item.name.endswith(('.lock', '.tmp', '.part')) else item)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print('ARCHIVED', destination, destination.stat().st_size, 'bytes', flush=True)


if __name__ == '__main__':
    main()
