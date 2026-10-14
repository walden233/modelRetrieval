"""Add gripper parameters and per-scene grasp trajectories to RH20T_subset."""

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

from bise.common.paths import resolve_path
from bise.common.fingerprints import file_hash
from bise.data.rh20t.grasp_geometry import (
    calibrated_grasp_tcp, measured_widths, validate_profile,
)
from bise.data.rh20t.trajectory_manifest import load_trajectory_file


def add_file(path, content, check):
    """Reuse identical outputs; exclusive creation prevents overwriting any file."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing output differs; left untouched: {path}")
        return "reused"
    if not check:
        with path.open("xb") as output:
            output.write(content)
    return "new"


def scene_geometry(scene, feedback_path, profile, max_gap_ms):
    robot = load_trajectory_file(scene / "tcp_base.npy")
    feedback = load_trajectory_file(feedback_path)
    cameras = {}
    for camera, frames in robot.items():
        if not len(frames):
            continue
        tcp = np.asarray([frame["tcp"] for frame in frames], dtype=np.float32)
        times = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)
        if (tcp.shape != (len(frames), 7) or not np.isfinite(tcp).all()
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or np.any(np.linalg.norm(tcp[:, 3:], axis=1) < 1e-8)):
            raise ValueError(f"Invalid TCP trajectory: {scene}, camera {camera}")
        if camera not in feedback:
            raise ValueError(f"Missing actual gripper feedback: {feedback_path}, camera {camera}")
        opening = measured_widths(feedback[camera], times, profile, max_gap_ms)
        cameras[camera] = calibrated_grasp_tcp(tcp, opening, profile)
    if not cameras:
        raise ValueError(f"No nonempty TCP cameras: {scene}")
    buffer = io.BytesIO()
    # Each camera stores float32 [N, 8]: XYZ, quaternion wxyz, normalized opening.
    np.save(buffer, cameras)
    return buffer.getvalue()


def raw_scene_index(raw_root, hardware):
    """Match copied TCP files by content, independently of subset scene numbering."""
    index = defaultdict(list)
    for cfg in sorted(hardware):
        root = raw_root / f"RH20T_{cfg}"
        if not root.is_dir():
            raise FileNotFoundError(f"Raw archive is missing/unreadable: {root}")
        for scene in sorted(root.glob("task_*_user_*_scene_*_cfg_*")):
            if not re.fullmatch(r"task_\d+_user_\d+_scene_\d+_cfg_\d+", scene.name):
                continue
            tcp = scene / "transformed/tcp_base.npy"
            if tcp.is_file():
                index[file_hash(tcp)].append((scene, cfg))
        print(f"{cfg}: indexed original TCP hashes", flush=True)
    return index


def prepare_dataset_geometry(dataset_root, parameters_path, raw_root, check=False,
                             cfgs=("cfg2", "cfg3")):
    dataset_root = resolve_path(dataset_root)
    parameters = json.loads(resolve_path(parameters_path).read_text())
    raw_root = resolve_path(raw_root)
    index = raw_scene_index(raw_root, parameters["profiles"])
    groups, missing = {}, []
    for cfg in cfgs:
        root = dataset_root / f"RH20T_{cfg}"
        scenes = sorted(root.glob("task_*/scene_*"))
        if not scenes:
            raise ValueError(f"No RH20T scenes found: {root}")
        rows, profiles = [], {}
        for scene in scenes:
            matches = index.get(file_hash(scene / "tcp_base.npy"), [])
            if len(matches) != 1:
                raise ValueError(f"TCP hash matched {len(matches)} original scenes: {scene}")
            original, hardware = matches[0]
            profiles[hardware] = validate_profile(parameters["profiles"][hardware])
            feedback = original / "transformed/gripper.npy"
            for path in (scene / "tcp_base.npy", feedback):
                try:
                    present = path.is_file()
                except OSError:
                    present = False
                if not present:
                    missing.append(str(path))
            rows.append((scene, feedback, profiles[hardware]))
        content = (json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2) + "\n").encode()
        # Check existing parameter files before adding any output.
        add_file(root / "geometry.json", content, check=True)
        groups[root] = (rows, content)
    if not groups:
        raise ValueError(f"No RH20T scenes found: {dataset_root}")
    if missing:
        raise FileNotFoundError(
            f"Missing/unreadable inputs ({len(missing)}); no files added:\n"
            + "\n".join(missing[:3])
        )
    results = {}
    for root, (rows, content) in groups.items():
        counts = {"new": 0, "reused": 0}
        feedback_counts = {"new": 0, "reused": 0}
        for scene, feedback, profile in rows:
            data = scene_geometry(scene, feedback, profile, parameters.get("max_gap_ms", 250))
            # Compare existing outputs before adding either file to this scene.
            raw_feedback = feedback.read_bytes()
            add_file(scene / "gripper.npy", raw_feedback, check=True)
            add_file(scene / "geometry.npy", data, check=True)
            feedback_counts[add_file(scene / "gripper.npy", raw_feedback, check)] += 1
            counts[add_file(scene / "geometry.npy", data, check)] += 1
        parameter_status = add_file(root / "geometry.json", content, check)
        results[root.name] = {"scenes": len(rows), **counts, "gripper": feedback_counts,
                              "parameters": parameter_status}
        action = "would add" if check else "added"
        print(f"{root.name}: {action} {feedback_counts['new']} gripper.npy, "
              f"{counts['new']} geometry.npy; geometry.json {parameter_status}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/RH20T_subset")
    parser.add_argument("--parameters",
                        default="artifacts/manifests/retrain/s43/cfg2/geometry.json")
    parser.add_argument("--raw-root", default="/mnt/d",
                        help="Parent of RH20T_cfg2 and RH20T_cfg3 raw archives")
    parser.add_argument("--cfg", nargs="+", choices=("cfg2", "cfg3"), default=["cfg2", "cfg3"])
    parser.add_argument("--check", action="store_true", help="Validate without adding files")
    args = parser.parse_args()
    try:
        prepare_dataset_geometry(args.dataset_root, args.parameters, args.raw_root,
                                 args.check, args.cfg)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
