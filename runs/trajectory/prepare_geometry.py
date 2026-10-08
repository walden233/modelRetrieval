"""Prepare frozen grasp trajectories for the s43 trajectory experiments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.common.paths import resolve_path
from bise.data.rh20t.grasp_geometry import geometry_plan, prepare_grasp_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain/s43/grippers.json")
    parser.add_argument("--manifests", default="artifacts/manifests/retrain/s43")
    parser.add_argument("--check", action="store_true",
                        help="Report missing inputs without writing")
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.check:
        _, report = geometry_plan(config, args.manifests)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 2
    result = prepare_grasp_geometry(config, args.manifests)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
