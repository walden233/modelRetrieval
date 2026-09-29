"""Freeze cfg2/cfg3 hand-trajectory inventories and the shared cfg2 test scenes."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.data.rh20t.hand_3d_trajectory_dataset import (
    RH20THand3DTrajectoryDataset, build_hand_3d_trajectory_manifest,
)
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/RH20T_subset")
    parser.add_argument("--output-dir", default="artifacts/manifests/trajectory_hand_3d_v1")
    parser.add_argument("--shared-test-output",
                        default="artifacts/manifests/rh20t_shared_v1/cfg2_fixed_test_scenes.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace frozen manifests: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for cfg in ("cfg2", "cfg3"):
        root = Path(args.dataset_root) / f"RH20T_{cfg}"
        manifest = build_hand_3d_trajectory_manifest(str(root))
        manifest_path = output / f"{cfg}_valid.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        dataset = RH20THand3DTrajectoryDataset(str(root), manifest_path=str(manifest_path))
        split = ({"unit": "scene", "seed": 42,
                  "ratios": {"train": .8, "val": .1, "test": .1}}
                 if cfg == "cfg2" else {"all_as_test": True})
        splits = build_split_manifest(split_trajectory_dataset(dataset, split))
        (output / f"{cfg}_split.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
        if cfg == "cfg2":
            shared_path = Path(args.shared_test_output)
            shared_payload = {"test": splits["test"]}
            if shared_path.exists():
                if json.loads(shared_path.read_text(encoding="utf-8")) != shared_payload:
                    raise ValueError(f"Existing shared fixed test differs: {shared_path}")
            else:
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                shared_path.write_text(json.dumps(shared_payload, indent=2), encoding="utf-8")
        print(json.dumps({"cfg": cfg, "scenes": len(dataset),
                          "camera_sequences": sum(len(x["cameras"]) for x in manifest["scenes"]),
                          "split_scenes": {k: len(v) for k, v in splits.items()},
                          "excluded_records": len(manifest["exclusions"]),
                          "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
