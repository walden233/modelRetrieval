"""Freeze the retrain scene splits, trajectory inventories, and paired videos."""

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

from bise.data.rh20t.splits import prepare_protocols, relocate_protocols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain/s43/run.json")
    parser.add_argument("--batch", help="Batch whose frozen manifests will be prepared")
    parser.add_argument("--seed", type=int,
                        help="Override the seed for new shared splits; reuse existing manifests")
    parser.add_argument("--relocate", action="store_true",
                        help="Rebind an existing copied batch to local paths without resampling")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    seed = int(config.get("seed", 42) if args.seed is None else args.seed)
    batch = args.batch or (config["batch"] if args.seed is None else f"s{seed}")
    if Path(batch).name != batch or batch in (".", ".."):
        parser.error("--batch must be a single directory name")
    manifest_batch = config.get("manifest_batch", batch)
    if Path(manifest_batch).name != manifest_batch or manifest_batch in (".", ".."):
        parser.error("manifest_batch must be a single directory name")
    output = Path(config["manifest_dir"]) / manifest_batch
    if args.relocate:
        relocate_protocols(config["dataset_root"], output)
    result = prepare_protocols(config["dataset_root"], config.get("source_dir"),
                               output, seed,
                               config.get("prepare", {}).get("cfg2_test_limit", 98))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
