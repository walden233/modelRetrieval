"""Analyze camera consistency and redraw heatmaps without loading a video model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.common.paths import resolve_path
from bise.modalities.video.views import export_views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='Video run; repeat for comparisons')
    parser.add_argument('--cfg', default='cfg2,cfg3', help='Comma-separated saved evaluation folders')
    args = parser.parse_args()
    for run in args.run:
        for cfg in args.cfg.split(','):
            directory = resolve_path(run) / 'evaluations' / cfg.strip()
            data = export_views(directory)
            print(f"{directory}: {data['complete_scenes']}/{data['scenes']} complete scenes")
            for kind, row in data['summary'].items():
                print(kind, row)


if __name__ == '__main__':
    main()
