"""Expand frozen camera-pair splits without decoding videos or hashing data."""

import json
import random
from collections import defaultdict
from pathlib import Path

from bise.common.paths import project_root, resolve_path


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return str(path.resolve())


def video_record(cfg: str, root: Path, task: str, scene: str, camera: str,
                 role: str, split: str = "extra") -> dict:
    path = root / task / scene / f"cam_{camera}_{role}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen video missing: {path}")
    stat = path.stat()
    scene_id = f"{cfg}/{task}/{scene}"
    return {
        "sample_id": f"{scene_id}/{camera}/{role}", "cfg": cfg,
        "task_id": f"{cfg}/{task}", "scene_id": scene_id, "camera_id": camera,
        "pair_id": f"{scene_id}/{camera}",
        "source_pair_id": f"rh20t::{task}::{scene}::{camera}",
        "role": role, "video_path": relative_path(path), "split": split,
        "source": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    }


def expand_split(path: str | Path, cfg: str | None = None) -> tuple[list[dict], dict]:
    path = resolve_path(path)
    cfg = cfg or path.parent.name
    mapping = json.loads(path.read_text())
    meta = json.loads((path.parent / "metadata.json").read_text())
    root = resolve_path(meta["data_root"])
    records, seen, scene_splits = [], set(), {}
    for split in ("train", "val", "test"):
        values = mapping.get(split, [])
        if not isinstance(values, list):
            raise ValueError(f"Invalid {split} in {path}")
        for pair in values:
            parts = pair.split("::")
            if len(parts) != 4 or parts[0] != "rh20t":
                raise ValueError(f"Invalid camera-pair identity: {pair}")
            _, task, scene, camera = parts
            if pair in seen:
                raise ValueError(f"Duplicated/cross-split camera pair: {pair}")
            seen.add(pair)
            key = (task, scene)
            if key in scene_splits and scene_splits[key] != split:
                raise ValueError(f"Scene crosses splits: {key}")
            scene_splits[key] = split
            for role in ("human", "robot"):
                records.append(video_record(cfg, root, task, scene, camera, role, split))
    source = {"path": relative_path(path), "data_root": relative_path(root),
              "splits": {key: mapping.get(key, []) for key in ("train", "val", "test")}}
    return records, source


def explicit_videos(path: str | Path, sources: dict) -> list[dict]:
    records = []
    for line in resolve_path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        video = resolve_path(line.strip()).resolve()
        for cfg, source in sources.items():
            root = resolve_path(source["data_root"]).resolve()
            if video.is_relative_to(root):
                relative = video.relative_to(root)
                if len(relative.parts) != 3:
                    raise ValueError(f"Expected task/scene/cam_*_role.mp4: {video}")
                task, scene, filename = relative.parts
                camera, role = filename.removeprefix("cam_").removesuffix(".mp4").rsplit("_", 1)
                if role not in ("human", "robot") or not filename.startswith("cam_"):
                    raise ValueError(f"Unrecognized video filename: {video}")
                records.append(video_record(cfg, root, task, scene, camera, role))
                break
        else:
            raise ValueError(f"Video is outside configured cfg roots: {video}")
    return list({row["sample_id"]: row for row in records}.values())


def sample_cost(records: list[dict], config: dict) -> list[dict]:
    settings = config["cost"]
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if (row["cfg"] == settings["cfg"] and
                (settings["split"] == 'all' or row["split"] == settings["split"])):
            groups[row["task_id"]][row["scene_id"]].append(row)
    count = settings["tasks"]
    if len(groups) < count:
        raise ValueError(f"Cost scope requires {count} distinct tasks, found {len(groups)}")
    rng = random.Random(settings["seed"])
    result = []
    for task in sorted(rng.sample(sorted(groups), count)):
        scene = rng.choice(sorted(groups[task]))
        rows = groups[task][scene]
        if len(rows) != 4 or {x["role"] for x in rows} != {"human", "robot"}:
            raise ValueError(f"Cost scene requires two complete camera pairs: {scene}")
        result.extend(rows)
    return result


def balanced_limit(records: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return records
    if limit <= 0:
        raise ValueError("--limit must be positive")
    grouped = defaultdict(dict)
    # Test scenes come first for useful end-to-end smoke evaluation.
    for row in sorted(records, key=lambda r: (
            {"test": 0, "train": 1, "val": 2}.get(r["split"], 3), r["scene_id"],
            r["camera_id"], r["role"])):
        grouped[row["cfg"]].setdefault(row["scene_id"], []).append(row)
    queues = {}
    for cfg, scenes in sorted(grouped.items()):
        tasks = defaultdict(list)
        for members in scenes.values():
            tasks[(members[0]['split'], members[0]['task_id'])].append(members)
        ordered = []
        while any(tasks.values()):
            for queue in tasks.values():
                if queue:
                    ordered.append(queue.pop(0))
        queues[cfg] = ordered
    result = []
    while any(queues.values()) and len(result) < limit:
        for queue in queues.values():
            if queue:
                result.extend(queue.pop(0))
    return result[:limit]
