"""Prepare training metadata without constructing models or decoding samples."""

import copy
import json
from pathlib import Path

from .evaluation import bind_protocols
from .fingerprints import write_json


def prepare_training_run(config: dict, modality: str, run: Path,
                         split_manifest: Path | None = None) -> None:
    config = copy.deepcopy(config)
    if split_manifest is not None and split_manifest.is_file():
        config.setdefault("split", {})["manifest_path"] = str(split_manifest)
    if modality == "trajectory":
        from bise.modalities.trajectory.factory import (
            build_trajectory_dataset, build_split_manifest, split_trajectory_dataset,
        )
        if (config["model_params"].get("grasp_geometry")
                or config["model_params"].get("use_gripper")):
            from bise.data.rh20t.grasp_geometry import bind_geometry
            bind_geometry(config)
        dataset = build_trajectory_dataset(config)
        split_dataset = split_trajectory_dataset
    else:
        from bise.modalities.video.factory import (
            build_video_dataset, build_split_manifest, split_video_dataset,
        )
        # Dataset construction only inventories samples; preprocessing is never invoked here.
        dataset = build_video_dataset(config["dataset"], processor=object(), is_train=True,
                                      metadata_only=True)
        split_dataset = split_video_dataset
    split = dict(config.get("split") or {})
    split.setdefault("seed", int(config.get("seed", 42)))
    if split.get("scene_counts") and not split.get("manifest_path"):
        from bise.data.rh20t.task_split import split_task_dataset
        subsets = split_task_dataset(dataset, split, modality,
                                     config.get("eval_manifests", {}).get("cfg2"))
    else:
        subsets = split_dataset(dataset, split)
    mapping = build_split_manifest(subsets)
    from bise.data.rh20t.splits import validate_split
    counts = validate_split(mapping, split.get("unit", "scene"))
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "split_manifest.json", mapping)
    write_json(run / "data_manifest.json", dataset.manifest)
    write_json(run / "split_info.json", {
        "protocol": config["protocol"], "unit": split.get("unit", "scene"),
        "counts": counts,
        "use_fixed_val": split.get("unit") != "task" and split.get("use_fixed_val", True),
        **({"scene_counts": split["scene_counts"]} if split.get("scene_counts") else {}),
    })
    config["split"] = {**split, "manifest_path": str(run / "split_manifest.json")}
    config.setdefault("checkpoint_selection", {
        "loss_weight": 0.1, "mrr_weight": 1.0, "ndcg_weight": 1.0,
    })
    bind_protocols(run, config)
    write_json(run / "params.json", config)
    write_json(run / "status.json", {"state": "prepared"})


def validate_prepared_run(run: Path) -> None:
    for name in ("params.json", "split_manifest.json", "data_manifest.json",
                 "split_info.json", "protocols.json", "status.json"):
        if not (run / name).is_file():
            raise FileNotFoundError(f"Prepared run is missing {name}: {run}")
    if (json.loads((run / "status.json").read_text()).get("state") != "prepared"
            or any((run / name).exists() for name in ("best_model.pth", "last_model.pth"))):
        raise ValueError(f"Expected an untrained prepared run: {run}")
