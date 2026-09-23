"""Freeze valid inputs and a shared clean split before trajectory experiments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.data.rh20t.trajectory_manifest import build_trajectory_manifest
from bise.data.rh20t.trajectory_dataset import RH20TTrajectoryDataset
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/RH20T_subset")
    parser.add_argument("--output-dir", default="artifacts/manifests/trajectory_first_batch")
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to replace frozen manifests: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for cfg in ("cfg2", "cfg3"):
        root = Path(args.dataset_root) / f"RH20T_{cfg}"
        manifest = build_trajectory_manifest(str(root))
        path = out / f"{cfg}_valid.json"
        path.write_text(json.dumps(manifest, indent=2))
        dataset = RH20TTrajectoryDataset(str(root), manifest_path=str(path))
        split_config = ({"unit": "scene", "seed": 42,
                         "ratios": {"train": .8, "val": .1, "test": .1}}
                        if cfg == "cfg2" else {"all_as_test": True})
        split = build_split_manifest(split_trajectory_dataset(dataset, split_config))
        (out / f"{cfg}_split.json").write_text(json.dumps(split, indent=2))
        print(json.dumps({"cfg": cfg, "valid_scenes": len(dataset),
                          "valid_camera_pairs": sum(len(e["cameras"]) for e in manifest["scenes"]),
                          "split_counts": {k: len(v) for k, v in split.items()},
                          "manifest": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
