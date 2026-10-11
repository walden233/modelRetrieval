"""Redraw retrain plots, rebuilding missing chart data from saved run results."""

import argparse

from _bootstrap import bootstrap
bootstrap()

from bise.common.paths import resolve_path
from bise.common.plots import index_plots, redraw_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--batch", help="Batch under artifacts/runs/retrain")
    target.add_argument("--root", help="Batch directory")
    parser.add_argument("--index-only", action="store_true",
                        help="Write plots.json without drawing images")
    args = parser.parse_args()
    root = resolve_path(args.root or f"artifacts/runs/retrain/{args.batch}")
    payload = index_plots(root) if args.index_only else redraw_plots(root)
    print(f"{'Indexed' if args.index_only else 'Redrawn'} {len(payload)} plots; "
          f"dependencies: {root / 'plots.json'}")


if __name__ == "__main__":
    main()
