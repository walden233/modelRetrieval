#!/usr/bin/env python3
"""Rebuild and overwrite the cfg2/cfg3 hand-3D validity manifests."""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bise.data.rh20t.hand_3d_trajectory_dataset import (  # noqa: E402
    build_hand_3d_trajectory_manifest,
)


CONFIGS = ("cfg2", "cfg3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="dataset/RH20T_subset",
        help="Directory containing RH20T_cfg2 and RH20T_cfg3.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/manifests/trajectory_hand_3d_v1",
        help="Directory in which cfg2_valid.json and cfg3_valid.json are replaced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    missing = [dataset_root / f"RH20T_{cfg}" for cfg in CONFIGS
               if not (dataset_root / f"RH20T_{cfg}").is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories: " + ", ".join(map(str, missing)))

    manifests = {}
    for cfg in CONFIGS:
        source = dataset_root / f"RH20T_{cfg}"
        print(f"[{cfg}] scanning {source}", flush=True)
        manifest = build_hand_3d_trajectory_manifest(str(source))
        manifests[cfg] = manifest
        print(
            f"[{cfg}] valid scenes={len(manifest['scenes'])}, "
            f"exclusions={len(manifest['exclusions'])}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    staged = {}
    try:
        for cfg, manifest in manifests.items():
            destination = output_dir / f"{cfg}_valid.json"
            temporary = output_dir / f".{destination.name}.tmp.{os.getpid()}"
            temporary.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            staged[destination] = temporary
        for destination, temporary in staged.items():
            temporary.replace(destination)
            print(f"[{destination.stem[:4]}] replaced {destination}", flush=True)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)

    print("Finished rebuilding cfg2_valid.json and cfg3_valid.json.", flush=True)


if __name__ == "__main__":
    main()
