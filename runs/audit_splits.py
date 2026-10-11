"""Audit frozen protocol files, planned configs, and optional completed runs."""

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap
bootstrap()

from bise.common.comparison import load_runs
from bise.common.fingerprints import file_hash, write_json
from bise.common.paths import resolve_path
from bise.data.rh20t.splits import load_protocol, load_split_map, validate_split


def audit(config: dict, files: bool = False, runs: str | None = None) -> dict:
    report = {"protocols": {}, "groups": {}}
    manifests = config.get("eval_manifests")
    if manifests is None:
        directory = resolve_path(config["manifest_dir"]) / config["batch"]
        manifests = {cfg: directory / cfg / "metadata.json" for cfg in ("cfg2", "cfg3")}
    for cfg, path in manifests.items():
        payload = load_protocol(path)
        root = resolve_path(payload["data_root"])
        data = json.loads(resolve_path(payload["data_manifest"]).read_text())
        if files:
            for index, (identity, pairs) in enumerate(payload["video_pairs"].items()):
                for pair in pairs:
                    for domain in ("human", "robot"):
                        source = root / pair[f"{domain}_path"]
                        if not source.is_file():
                            raise FileNotFoundError(f"Frozen video file missing: {source}")
                if index % 50 == 0:
                    print(f"Audit {cfg} files {index + 1}/{len(payload['video_pairs'])}",
                          flush=True)
            for entry in data["scenes"]:
                for name, key in (("human_hand_3d.npy", "human_sha256"),
                                  ("tcp_base.npy", "robot_sha256")):
                    source = root / entry["scene_id"] / name
                    if file_hash(source) != entry[key]:
                        raise ValueError(f"Frozen trajectory changed: {source}")
        report["protocols"][cfg] = {"test": len(payload["test"]), "val": len(payload["val"]),
                                    "shared_scenes": len(payload["eligible_scenes"]),
                                    "video_scenes": len(payload["video_pairs"]),
                                    "sha256": file_hash(resolve_path(path))}
    from bise.modalities.trajectory.factory import (
        build_trajectory_dataset, split_trajectory_dataset,
    )
    from bise.modalities.trajectory.factory import build_split_manifest as trajectory_manifest
    from bise.modalities.video.factory import build_video_dataset, split_video_dataset
    from bise.modalities.video.factory import build_split_manifest as video_manifest
    from bise.data.rh20t.task_split import split_task_dataset
    for modality in ("trajectory", "video"):
        for key, entry in config[modality].items():
            if entry.get("raw"):
                continue
            current = json.loads(resolve_path(entry["config"]).read_text())
            split = {**current["split"]}
            split.setdefault("seed", current["seed"])
            if modality == "trajectory":
                source = build_trajectory_dataset(current)
                if files and (current["model_params"].get("grasp_geometry")
                        or current["model_params"].get("use_gripper")):
                    for index in range(len(source)):
                        source[index]
                subsets = (split_task_dataset(source, split, modality,
                                             current.get("eval_manifests", {}).get("cfg2"))
                           if split.get("scene_counts") and not split.get("manifest_path")
                           else split_trajectory_dataset(source, split))
                mapping = trajectory_manifest(subsets)
            else:
                source = build_video_dataset(current["dataset"], object(), False)
                subsets = (split_task_dataset(source, split, modality,
                                             current.get("eval_manifests", {}).get("cfg2"))
                           if split.get("scene_counts") and not split.get("manifest_path")
                           else split_video_dataset(source, split))
                mapping = video_manifest(subsets)
            counts = validate_split(mapping, split.get("unit", "scene"))
            report["groups"][f"{modality}/{key}"] = counts
    if runs:
        registry = load_runs(runs)
        report["runs"] = {}
        for key, entry in registry["groups"].items():
            if entry.get("raw"):
                continue
            run = resolve_path(entry["run_dir"])
            params = json.loads((run / "params.json").read_text())
            report["runs"][key] = validate_split(load_split_map(run / "split_manifest.json"),
                                                  params["split"].get("unit", "scene"))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retrain/run.json")
    parser.add_argument("--files", action="store_true",
                        help="Verify trajectory hashes and video file existence")
    parser.add_argument("--runs", help="Additionally audit a new batch runs.json")
    parser.add_argument("--output", help="Optional JSON audit report")
    args = parser.parse_args()
    report = audit(json.loads(Path(args.config).read_text()), args.files, args.runs)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
