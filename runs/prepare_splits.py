"""Freeze the retrain scene splits, trajectory inventories, and paired videos."""

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

from bise.data.rh20t.splits import prepare_protocols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain/run.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    result = prepare_protocols(config["dataset_root"], config["source_dir"],
                               config["data_dir"], config["manifest_dir"], config["seed"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
