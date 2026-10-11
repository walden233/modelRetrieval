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
    parser.add_argument("--batch", help="Batch whose frozen manifests will be prepared")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    batch = args.batch or config["batch"]
    if Path(batch).name != batch or batch in (".", ".."):
        parser.error("--batch must be a single directory name")
    output = Path(config["manifest_dir"]) / batch
    result = prepare_protocols(config["dataset_root"], config.get("source_dir"),
                               output, config["seed"],
                               config.get("prepare", {}).get("cfg2_test_first96", True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
