"""Run S1/S2c/S2w/S3/S4/S5 experiments under artifacts/semantic/v2."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.modalities.semantic.representations import REPRESENTATIONS
from bise.modalities.semantic.workflow import load_config, run


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/semantic/ch3.json')
    parser.add_argument('--stage', default='all', choices=[
        'all', 'prepare', 'annotate', 'encode', 'eval', 'analyze', 'plot', 'report', 'cost', 'scene',
        'remove'])
    parser.add_argument('--cfg', type=comma_list)
    parser.add_argument('--split', choices=['all', 'train', 'val', 'test'])
    parser.add_argument('--only', type=comma_list, help='Independent representations: ' + ','.join(REPRESENTATIONS))
    parser.add_argument('--provider', help='Annotation provider key from config.providers')
    parser.add_argument('--providers', type=comma_list,
                        help='Only these cost providers may call APIs; completed others use cache')
    parser.add_argument('--frames', type=int)
    parser.add_argument('--concurrency', type=int, help='Override all provider concurrency settings')
    parser.add_argument('--limit', type=int, help='Smoke limit: videos for annotation/cost, scenes for teacher')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Retained for existing commands; partial results are now automatic')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true', help='Submit selected successful items again; preserves attempts')
    parser.add_argument('--retry-failed', action='store_true', help='Retry missing/failed items (also default behavior)')
    parser.add_argument('--retry-unknown', action='store_true', help='Retry interrupted requests with unknown billing outcome')
    parser.add_argument('--fresh-benchmark', action='store_true', help='Fresh cost-subset API calls, explicitly billed again')
    parser.add_argument('--videos', help='Explicit additional video paths, one per line')
    parser.add_argument('--add-manifest', action='append')
    parser.add_argument('--run', help='Video run whose train split is used for teacher export')
    parser.add_argument('--source', choices=list(REPRESENTATIONS),
                        help='Scene teacher source representation (default: config.scene.source)')
    parser.add_argument('--repair', action='store_true',
                        help='Scene only: reuse valid teachers; regenerate invalid/null teachers')
    parser.add_argument('--scene-ids', type=comma_list,
                        help='Extra scene IDs to regenerate with --stage scene --repair')
    parser.add_argument('--video-run', help='Saved video run for strictly matched failure analysis')
    parser.add_argument('--trajectory-run', help='Saved trajectory run for matched failure analysis')
    args = parser.parse_args(argv)
    if args.only and not set(args.only) <= set(REPRESENTATIONS):
        parser.error('--only must contain ' + ','.join(REPRESENTATIONS))
    if args.stage == 'remove' and not args.only:
        parser.error('--stage remove requires explicit --only')
    if args.source and args.stage != 'scene':
        parser.error('--source requires --stage scene')
    if args.repair and args.stage != 'scene':
        parser.error('--repair requires --stage scene')
    if args.scene_ids and not args.repair:
        parser.error('--scene-ids requires --repair')
    if (args.videos or args.add_manifest) and args.stage not in ('annotate', 'prepare'):
        parser.error('Additional videos require --stage annotate or prepare')
    return args


def main():
    args = parse_args()
    try:
        result = run(load_config(args.config), args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
