"""Frozen scene splits and camera pairs for the retrain protocol."""

import json
from pathlib import Path

import numpy as np

from bise.common.paths import resolve_path
from bise.common.fingerprints import file_hash, object_hash, write_json


SPLITS = ("train", "val", "test")
PROTOCOL = "retrain_v2"


def scene_id(value: str) -> str:
    parts = value.split("::")
    if len(parts) == 4 and parts[0] == "rh20t":
        return f"{parts[1]}/{parts[2]}"
    if len(value.split("/")) == 2:
        return value
    raise ValueError(f"Unsupported RH20T identity: {value}")


def load_split_map(path: str | Path) -> dict:
    payload = json.loads(resolve_path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Split manifest must be an object")
    mapping = {key: payload[key] for key in SPLITS if key in payload}
    if not mapping:
        raise ValueError("Split manifest has no train/val/test lists")
    seen = set()
    for key, values in mapping.items():
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            raise ValueError(f"Invalid identities in split {key}")
        seen.update(values)
    return mapping


def validate_split(mapping: dict, unit: str = "scene") -> dict:
    scenes = {key: {scene_id(x) for x in mapping.get(key, [])} for key in SPLITS}
    counts = {key: {"samples": len(mapping.get(key, [])), "scenes": len(scenes[key]),
                    "tasks": len({x.split('/')[0] for x in scenes[key]})} for key in SPLITS}
    for i, left in enumerate(SPLITS):
        values = mapping.get(left, [])
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate samples in {left}")
        for right in SPLITS[i + 1:]:
            if unit == "task":
                tasks = {x.split('/')[0] for x in scenes[left]}
                other = {x.split('/')[0] for x in scenes[right]}
                if tasks & other:
                    raise ValueError(f"Task overlap {left}/{right}: {sorted(tasks & other)[:5]}")
    return counts


def protocol_metadata_path(path: str | Path) -> Path:
    path = resolve_path(path)
    if path.name in ("split_manifest_trajectory.json", "split_manifest_video.json"):
        return path.parent / "metadata.json"
    return path


def protocol_hash(path: str | Path) -> str:
    path = protocol_metadata_path(path)
    metadata = json.loads(path.read_text())
    if metadata.get("schema_version") == 3:
        return object_hash({"metadata": file_hash(path), "splits": {
            modality: file_hash(resolve_path(source))
            for modality, source in metadata["split_manifests"].items()
        }})
    return file_hash(path)


def _batch_protocol(metadata: dict) -> dict:
    trajectory = load_split_map(metadata["split_manifests"]["trajectory"])
    video = load_split_map(metadata["split_manifests"]["video"])
    validate_split(trajectory)
    validate_split(video)
    data = json.loads(resolve_path(metadata["data_manifest"]).read_text())
    entries = {entry["scene_id"]: entry for entry in data["scenes"]}
    pairs = _video_pairs(resolve_path(metadata["data_root"]), entries, False, hashes=False)
    for name in SPLITS:
        scenes, samples = trajectory.get(name, []), video.get(name, [])
        if len(samples) != 2 * len(scenes):
            raise ValueError(f"Batch {name} requires two video pairs per trajectory scene")
        for index, identity in enumerate(scenes):
            selected = samples[2 * index:2 * index + 2]
            if any(scene_id(sample) != identity for sample in selected):
                raise ValueError(f"Batch trajectory/video {name} scene order differs")
            scene_pairs = []
            for sample in selected:
                parts = sample.split("::")
                if len(parts) != 4 or parts[0] != "rh20t":
                    raise ValueError(f"Invalid frozen video sample: {sample}")
                camera = parts[3]
                scene_pairs.append({"sample_id": sample, "camera_id": camera,
                                    **{f"{domain}_path": f"{identity}/cam_{camera}_{domain}.mp4"
                                       for domain in ("human", "robot")}})
            pairs[identity] = scene_pairs
    return {**metadata, **trajectory, "video_pairs": pairs,
            "eligible_scenes": sorted(entries.keys() & pairs.keys()),
            "inventory_sha256": object_hash(pairs)}


def load_protocol(path: str | Path, data_root: str | Path | None = None,
                  video_hashes: bool = False) -> dict:
    path = protocol_metadata_path(path)
    payload = json.loads(path.read_text())
    if payload.get("protocol") != PROTOCOL or payload.get("schema_version") not in (2, 3):
        raise ValueError(f"Expected {PROTOCOL} manifest: {path}")
    if payload["schema_version"] == 3:
        payload = _batch_protocol(payload)
    mapping = {key: payload[key] for key in SPLITS if key in payload}
    validate_split(mapping)
    if data_root is not None and resolve_path(data_root) != resolve_path(payload["data_root"]):
        raise ValueError(f"Manifest belongs to a different data root: {path}")
    data_path = resolve_path(payload["data_manifest"])
    if file_hash(data_path) != payload["data_sha256"]:
        raise ValueError(f"Frozen trajectory inventory changed: {data_path}")
    data = json.loads(data_path.read_text())
    if (data.get("human_source") != "hand_3d_base"
            or resolve_path(data["data_root"]) != resolve_path(payload["data_root"])):
        raise ValueError(f"Invalid frozen trajectory inventory: {data_path}")
    if object_hash(payload["video_pairs"]) != payload["inventory_sha256"]:
        raise ValueError(f"Frozen video inventory changed: {path}")
    eligible = payload["eligible_scenes"]
    if len(set(eligible)) != len(eligible):
        raise ValueError("Duplicate eligible scenes")
    if not set(mapping.get("test", []) + mapping.get("val", [])) <= set(eligible):
        raise ValueError("Evaluation contains a scene outside the shared eligible pool")
    if payload["cfg"] == "cfg2":
        if len(mapping.get("test", [])) != 120 or len(mapping.get("val", [])) != 50:
            raise ValueError("cfg2 requires exactly 120 test and 50 validation scenes")
    elif payload["cfg"] == "cfg3":
        from collections import Counter
        counts = Counter(x.split('/')[0] for x in mapping.get("test", []))
        tasks = {entry["task_id"] for entry in data["scenes"]}
        if set(counts) != tasks or any(n != 2 for n in counts.values()):
            raise ValueError("cfg3 requires two evaluation scenes per task")
    else:
        raise ValueError(f"Unknown cfg in manifest: {path}")
    for identity, pairs in payload["video_pairs"].items():
        if len(pairs) != 2 or len({p["camera_id"] for p in pairs}) != 2:
            raise ValueError(f"Expected two distinct paired cameras: {identity}")
        task, name = identity.split("/")
        for pair in pairs:
            if pair["sample_id"] != f"rh20t::{task}::{name}::{pair['camera_id']}":
                raise ValueError(f"Inconsistent video sample identity: {identity}")
            for domain in ("human", "robot"):
                source = Path(pair[f"{domain}_path"])
                camera = source.name.removesuffix(f"_{domain}.mp4").removeprefix("cam_")
                if str(source.parent) != identity or camera != pair["camera_id"]:
                    raise ValueError(f"Inconsistent paired video path: {source}")
    for identity in mapping.get("test", []) + mapping.get("val", []):
        if identity not in payload["video_pairs"]:
            raise ValueError(f"Scene lacks frozen video pairs: {identity}")
    if video_hashes:
        root = resolve_path(payload["data_root"])
        for pairs in payload["video_pairs"].values():
            for pair in pairs:
                for domain in ("human", "robot"):
                    key = f"{domain}_sha256"
                    if key not in pair:
                        pair[key] = file_hash(root / pair[f"{domain}_path"])
        if payload["schema_version"] == 3:
            payload["inventory_sha256"] = object_hash(payload["video_pairs"])
    return payload


def shared_scene_split(available: list[str], config: dict) -> dict:
    if config.get("unit", "scene") != "scene":
        raise ValueError("Shared scene split requires unit=scene")
    payload = load_protocol(config["shared_manifest_path"])
    test = payload["test"]
    frozen = payload.get("train", []) + payload["val"] + test
    missing = set(frozen) - set(available)
    if missing:
        raise ValueError(f"Shared scenes missing from dataset: {sorted(missing)[:5]}")
    remaining = sorted(set(frozen if payload["schema_version"] == 3 else available) - set(test))
    if config.get("use_fixed_val", True):
        val = payload["val"]
    else:
        count = int(config.get("val_count", 50))
        if count < 1 or count >= len(remaining):
            raise ValueError("val_count must leave a nonempty training pool")
        rng = np.random.default_rng(int(config.get("seed", 42)))
        val = sorted(rng.choice(remaining, count, replace=False).tolist())
    train = (payload["train"] if payload["schema_version"] == 3
             and config.get("use_fixed_val", True) else sorted(set(remaining) - set(val)))
    mapping = {"train": list(train), "val": list(val), "test": list(test)}
    validate_split(mapping)
    return mapping


def _video_pairs(root: Path, hand_entries: dict, decode: bool, hashes: bool = True) -> dict:
    inventory = {}
    paths = sorted(p for p in root.glob("task_*/scene_*") if p.is_dir())
    for index, path in enumerate(paths):
        identity = str(path.relative_to(root))
        human = {p.name.removesuffix("_human.mp4"): p for p in path.glob("*_human.mp4")}
        robot = {p.name.removesuffix("_robot.mp4"): p for p in path.glob("*_robot.mp4")}
        cameras = sorted(human.keys() & robot.keys())
        if len(cameras) < 2:
            continue
        # Prefer cameras that also have valid hand trajectories, then freeze the choice.
        valid_hand = set(hand_entries.get(identity, {}).get("cameras", {}))
        cameras.sort(key=lambda c: (c.removeprefix("cam_") not in valid_hand, c))
        pairs = []
        for camera in cameras[:2]:
            item = {"camera_id": camera.removeprefix("cam_"),
                    "sample_id": (f"rh20t::{path.parent.name}::{path.name}::"
                                  f"{camera.removeprefix('cam_')}")}
            for domain, files in (("human", human), ("robot", robot)):
                file = files[camera]
                if decode:
                    from decord import VideoReader, cpu
                    reader = VideoReader(str(file), ctx=cpu(0), num_threads=1)
                    if len(reader) < 1:
                        raise ValueError(f"Empty video: {file}")
                    reader.get_batch(np.linspace(0, len(reader) - 1, 16, dtype=int)).asnumpy()
                    item[f"{domain}_frames"] = len(reader)
                item[f"{domain}_path"] = str(file.relative_to(root))
                if hashes:
                    item[f"{domain}_sha256"] = file_hash(file)
            pairs.append(item)
        inventory[identity] = pairs
        if (decode or hashes) and index % 25 == 0:
            print(f"{root.name}: freeze video pairs {index + 1}/{len(paths)}", flush=True)
    return inventory


def prepare_protocols(dataset_root: str | Path, source_dir: str | Path,
                      output_dir: str | Path,
                      seed: int = 42, decode: bool = True) -> dict:
    dataset_root, source_dir = resolve_path(dataset_root), resolve_path(source_dir)
    output_dir = resolve_path(output_dir)
    result = {}
    for cfg in ("cfg2", "cfg3"):
        directory = output_dir / cfg
        target = directory / "metadata.json"
        if target.exists():
            frozen = load_protocol(target, dataset_root / f"RH20T_{cfg}")
            if frozen["seed"] != seed:
                raise ValueError("Existing batch protocol has a different seed; use a new batch")
            result[cfg] = str(target)
            continue
        root = dataset_root / f"RH20T_{cfg}"
        data = json.loads((source_dir / f"{cfg}_valid.json").read_text())
        if data.get("human_source") != "hand_3d_base" or resolve_path(data["data_root"]) != root:
            raise ValueError(f"Wrong trajectory inventory for {cfg}")
        entries = {entry["scene_id"]: entry for entry in data["scenes"]}
        for identity, entry in entries.items():
            for name, key in (("human_hand_3d.npy", "human_sha256"),
                              ("tcp_base.npy", "robot_sha256")):
                if file_hash(root / identity / name) != entry[key]:
                    raise ValueError(f"Trajectory source changed: {root / identity / name}")
        pairs = _video_pairs(root, entries, decode)
        eligible = sorted(entries.keys() & pairs.keys())
        rng = np.random.default_rng(seed)
        if cfg == "cfg2":
            if len(eligible) <= 170:
                raise ValueError("cfg2 needs more than 170 shared valid scenes")
            shuffled = rng.permutation(eligible).tolist()
            test, val = sorted(shuffled[:120]), sorted(shuffled[120:170])
        else:
            test, val = [], []
            for task in sorted({x.split('/')[0] for x in eligible}):
                scenes = [x for x in eligible if x.split('/')[0] == task]
                if len(scenes) < 2:
                    raise ValueError(f"Fewer than two shared valid scenes for cfg3 {task}")
                test.extend(sorted(rng.choice(scenes, 2, replace=False).tolist()))
        data_path = directory / "data_manifest.json"
        if data_path.exists() and json.loads(data_path.read_text()) != data:
            raise FileExistsError(f"Frozen data inventory differs: {data_path}")
        write_json(data_path, data)
        mapping = {"train": sorted(set(eligible) - set(test + val)) if cfg == "cfg2" else [],
                   "val": val, "test": test}
        trajectory_path = directory / "split_manifest_trajectory.json"
        video_path = directory / "split_manifest_video.json"
        video_mapping = {name: [pair["sample_id"] for identity in scenes
                                for pair in pairs[identity]] for name, scenes in mapping.items()}
        for path, split in ((trajectory_path, mapping), (video_path, video_mapping)):
            if path.exists() and load_split_map(path) != split:
                raise FileExistsError(f"Frozen batch split differs: {path}")
            write_json(path, split)
        payload = {"schema_version": 3, "protocol": PROTOCOL, "cfg": cfg, "seed": seed,
                   "data_root": str(root), "data_manifest": str(data_path),
                   "data_sha256": file_hash(data_path),
                   "split_manifests": {"trajectory": str(trajectory_path),
                                       "video": str(video_path)}}
        if cfg == "cfg3":
            payload["scenes_per_task"] = 2
        write_json(target, payload)
        load_protocol(target, root)
        result[cfg] = str(target)
        print(f"{cfg}: {len(test)} test, {len(val)} val, {len(eligible)} shared valid scenes",
              flush=True)
    return result
