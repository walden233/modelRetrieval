"""Compare all scenes versus per-task scene caps without changing existing results."""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.common.paths import resolve_path
from bise.modalities.semantic.scene_evaluation import evaluate_scenes
from bise.modalities.semantic.workflow import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/semantic/ch3.json')
    parser.add_argument('--cfg', default='cfg2')
    parser.add_argument('--split', default='all', choices=['all', 'train', 'val', 'test'])
    parser.add_argument('--caps', default='all,3,2,1', help='Maximum scenes per task')
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--only', help='Comma-separated representations; default all configured')
    parser.add_argument('--output', help='New directory; existing directories are never overwritten')
    args = parser.parse_args()
    config = load_config(args.config)
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = resolve_path(args.output) if args.output else (
        resolve_path(config['root']) / 'scene_eval' / f'{args.cfg}-{args.split}-{stamp}')
    caps = list(dict.fromkeys(cap.strip() for cap in args.caps.split(',') if cap.strip()))
    reps = [rep.strip() for rep in args.only.split(',')] if args.only else None
    if not caps:
        parser.error('--caps cannot be empty')
    try:
        evaluate_scenes(config, output, args.cfg, args.split, caps, args.seed, reps)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')
    print(f'Scene evaluation complete: {output}', flush=True)


if __name__ == '__main__':
    main()
