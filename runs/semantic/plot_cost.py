"""Redraw the cost overview from saved results without annotation or evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.common.paths import resolve_path
from bise.modalities.semantic.cost_report import export_cost_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='artifacts/semantic/v2')
    args = parser.parse_args()
    paths = export_cost_report(resolve_path(args.root))
    if not paths:
        parser.error('No cost/summary.json found under --root')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
