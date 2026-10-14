"""Prepare frozen grasp motion and virtual jaw points without changing source data."""

import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from kornia.geometry.conversions import rotation_matrix_to_quaternion

from bise.common.fingerprints import file_hash, object_hash, write_json
from bise.common.paths import resolve_path
from bise.modalities.trajectory.geometry import quaternion_to_matrix, validate_rigid_transform
from .trajectory_manifest import load_trajectory_file


def validate_profile(profile):
    if not profile.get("source"):
        raise ValueError("A sourced TCP-to-grasp reference definition is required")
    transform = validate_rigid_transform(profile.get("tcp_to_grasp"))
    reference = profile.get("reference", "calibrated_grasp")
    if reference not in ("calibrated_grasp", "aligned_tcp_proxy"):
        raise ValueError(f"Unknown grasp reference: {reference}")
    if reference == "aligned_tcp_proxy" and not torch.allclose(transform, torch.eye(4), atol=1e-6):
        raise ValueError("Aligned TCP proxy must preserve the stored frame with an identity transform")
    width = profile.get("max_width_m")
    scale = profile.get("width_scale_to_m")
    if width is None or not np.isfinite(width) or width <= 0:
        raise ValueError("max_width_m must be a positive measured/documented opening")
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError("width_scale_to_m must explicitly convert feedback width to meters")
    return profile


def source_identity(scene):
    path = scene / "T_base_from_camera.json"
    if not path.is_file():
        raise FileNotFoundError(f"Robot source provenance is required: {path}")
    name = json.loads(path.read_text()).get("raw_robot_scene", "")
    match = re.fullmatch(r"task_\d+_user_\d+_scene_\d+_cfg_(\d+)", name)
    if match is None:
        raise ValueError(f"Invalid original robot scene identity: {path}")
    return name, f"cfg{int(match.group(1))}"


def geometry_plan(config, manifest_dir):
    """Resolve original hardware per scene; directory names are not hardware labels."""
    records, counts, errors = {}, {}, []
    for cfg in ("cfg2", "cfg3"):
        inventory = resolve_path(manifest_dir) / cfg / "data_manifest.json"
        data = json.loads(inventory.read_text())
        rows = []
        for entry in data["scenes"]:
            scene = Path(data["data_root"]) / entry["scene_id"]
            try:
                original, hardware = source_identity(scene)
                local = scene / "gripper.npy"
                raw_root = config.get("raw_roots", {}).get(hardware)
                source = (local if local.is_file() else
                          resolve_path(raw_root) / original / "transformed/gripper.npy"
                          if raw_root else local)
                rows.append({"entry": entry, "source_scene": original, "source_cfg": hardware,
                             "source": source, "root": data["data_root"]})
                if not source.is_file():
                    errors.append(f"Missing gripper feedback: {source}")
            except (ValueError, FileNotFoundError) as exc:
                errors.append(str(exc))
        records[cfg] = rows
        counts[cfg] = dict(Counter(row["source_cfg"] for row in rows))
    hardware = {row["source_cfg"] for rows in records.values() for row in rows}
    calibration_errors = {}
    for cfg in sorted(hardware):
        try:
            validate_profile(config.get("profiles", {}).get(cfg, {}))
        except (ValueError, TypeError, RuntimeError) as exc:
            calibration_errors[cfg] = str(exc)
            errors.append(f"{cfg} calibration: {exc}")
    originals = {cfg: {row["source_scene"] for row in rows} for cfg, rows in records.items()}
    overlap = {"inventories": len(originals["cfg2"] & originals["cfg3"])}
    source_split = resolve_path(manifest_dir) / "cfg2/split_manifest_trajectory.json"
    target_split = resolve_path(manifest_dir) / "cfg3/split_manifest_trajectory.json"
    if source_split.is_file() and target_split.is_file():
        by_id = {cfg: {row["entry"]["scene_id"]: row["source_scene"] for row in rows}
                 for cfg, rows in records.items()}
        train = {by_id["cfg2"][s] for s in json.loads(source_split.read_text())["train"]}
        test = {by_id["cfg3"][s] for s in json.loads(target_split.read_text())["test"]}
        overlap["cfg2_train_cfg3_test"] = len(train & test)
    report = {"ready": not errors, "source_cfg_counts": counts,
              "source_overlap": overlap, "calibration_errors": calibration_errors,
              "error_count": len(errors), "errors": errors[:12]}
    return records, report


def measured_widths(series, target_times, profile, max_gap_ms=250):
    """Interpolate actual feedback on absolute TCP timestamps, without phase shifting."""
    rows = sorted((float(t), float(np.asarray(value["gripper_info"]).reshape(-1)[0]))
                  for t, value in series.items())
    if not rows:
        raise ValueError("Gripper feedback needs finite, strictly increasing timestamps")
    times, widths = np.asarray(rows, dtype=np.float64).T
    if (not np.isfinite(times).all() or not np.isfinite(widths).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Gripper feedback needs finite, strictly increasing timestamps")
    widths *= profile["width_scale_to_m"]
    maximum = profile["max_width_m"]
    if np.any(widths < -1e-6) or np.any(widths > maximum + 1e-6):
        raise ValueError("Measured opening exceeds calibration range; check units and hardware")
    target_times = np.asarray(target_times, dtype=np.float64)
    if not np.isfinite(target_times).all():
        raise ValueError("TCP timestamps must be finite")
    if len(times) == 1:
        if np.any(target_times != times[0]):
            raise ValueError("Single gripper feedback sample requires exact TCP timestamps")
        return np.full(target_times.shape, np.clip(widths[0], 0, maximum) / maximum)
    right = np.searchsorted(times, target_times).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    distance = np.minimum(abs(times[left] - target_times), abs(times[right] - target_times))
    inside = (target_times >= times[0]) & (target_times <= times[-1])
    interpolated = inside & (distance > 0)
    if (np.any(distance > max_gap_ms)
            or np.any(interpolated & (times[right] - times[left] > max_gap_ms))):
        raise ValueError("Gripper feedback has a missing interval or unmatched TCP timestamps")
    return np.interp(target_times, times, widths).clip(0, maximum) / maximum


def calibrated_grasp_tcp(tcp, opening, profile):
    """Apply the moving TCP offset, then append measured normalized opening."""
    transform = validate_rigid_transform(profile["tcp_to_grasp"])
    poses = torch.as_tensor(np.asarray(tcp), dtype=torch.float32)
    rotation = quaternion_to_matrix(poses[:, 3:7])
    position = poses[:, :3] + (rotation @ transform[:3, 3, None]).squeeze(-1)
    grasp_rotation = rotation @ transform[:3, :3]
    quaternion = rotation_matrix_to_quaternion(grasp_rotation)
    result = torch.cat((position, quaternion,
                        torch.as_tensor(opening, dtype=torch.float32)[:, None]), dim=-1)
    return result.numpy()


def prepare_grasp_geometry(config, manifest_dir):
    outputs = {}
    # Reuse complete snapshots even if the original raw archive is offline.
    targets = [resolve_path(manifest_dir) / cfg / "geometry.json" for cfg in ("cfg2", "cfg3")]
    for path in targets:
        if path.is_file():
            frozen = json.loads(path.read_text())
            if frozen["profiles"] != config["profiles"]:
                raise ValueError(f"Calibration differs from frozen geometry: {path}")
            if frozen["max_gap_ms"] != config.get("max_gap_ms", 250):
                raise ValueError(f"Feedback timing policy differs from frozen geometry: {path}")
            inventory = json.loads((path.parent / "data_manifest.json").read_text())
            if (resolve_path(frozen["data_root"]) != resolve_path(inventory["data_root"])
                    or set(frozen["scenes"]) != {entry["scene_id"] for entry in inventory["scenes"]}):
                raise ValueError(f"Geometry belongs to a different trajectory inventory: {path}")
            for entry in frozen["scenes"].values():
                if not resolve_path(entry["path"]).is_file():
                    raise FileNotFoundError(f"Prepared grasp trajectory missing: {entry['path']}")
            outputs[path.parent.name] = str(path)
    if len(outputs) == 2:
        return outputs
    records, report = geometry_plan(config, manifest_dir)
    if not report["ready"]:
        details = json.dumps(report, ensure_ascii=False)
        raise ValueError(f"Grasp preparation is not ready: {details}")
    for cfg, rows in records.items():
        target = resolve_path(manifest_dir) / cfg / "geometry.json"
        if target.is_file():
            outputs[cfg] = str(target)
            continue
        data = json.loads((target.parent / "data_manifest.json").read_text())
        prepared = {}
        for row in rows:
            entry = row["entry"]
            profile = config["profiles"][row["source_cfg"]]
            feedback = load_trajectory_file(row["source"])
            robot_path = Path(row["root"]) / entry["scene_id"] / "tcp_base.npy"
            robot = load_trajectory_file(robot_path)
            cameras = {}
            for camera in entry["cameras"]:
                if camera not in feedback:
                    raise ValueError(f"Missing actual feedback camera {camera}: {row['source']}")
                frames = robot[camera]
                opening = measured_widths(feedback[camera], [f["timestamp"] for f in frames],
                                          profile, config.get("max_gap_ms", 250))
                cameras[camera] = calibrated_grasp_tcp([f["tcp"] for f in frames], opening, profile)
            path = target.parent / "geometry" / f"{entry['scene_id']}.npy"
            buffer = io.BytesIO()
            np.save(buffer, cameras)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(buffer.getvalue())
            prepared[entry["scene_id"]] = {
                "path": str(path), "source_scene": row["source_scene"],
                "source_cfg": row["source_cfg"], "feedback_source": str(row["source"]),
            }
        references = {key: config["profiles"][key].get("reference", "calibrated_grasp")
                      for key in report["source_cfg_counts"][cfg]}
        proxy = "aligned_tcp_proxy" in references.values()
        axes = ({"x": "stored TCP x; virtual opening axis", "y": "stored TCP y",
                 "z": "stored TCP z"} if proxy else
                {"x": "jaw closing", "y": "approach", "z": "x cross y"})
        description = ("aligned TCP proxy with virtual jaw points and measured opening" if proxy
                       else "calibrated parallel-jaw contact-plane approximation")
        write_json(target, {"schema_version": 1, "data_root": data["data_root"],
                            "profiles": config["profiles"], "scenes": prepared,
                            "source_cfg_counts": report["source_cfg_counts"][cfg],
                            "source_overlap": report["source_overlap"],
                            "max_gap_ms": config.get("max_gap_ms", 250),
                            "references": references, "axes": axes,
                            "input": "reference XYZ, quaternion wxyz, actual opening / maximum",
                            "model": description})
        outputs[cfg] = str(target)
    return outputs


def bind_geometry(config):
    if not (config["model_params"].get("grasp_geometry")
            or config["model_params"].get("use_gripper")):
        return
    manifests = config["data_params"]["geometry_manifests"]
    config["geometry_hashes"] = {root: file_hash(resolve_path(path))
                                 for root, path in manifests.items()}


def geometry_manifest(config):
    root = Path(config["data_root"]).name
    path = resolve_path(config["data_params"]["geometry_manifests"][root])
    expected = config.get("geometry_hashes", {}).get(root)
    if expected is not None and file_hash(path) != expected:
        raise ValueError(f"Geometry changed since training: {path}")
    return path


def geometry_digest(config, selected):
    path = geometry_manifest(config)
    data = json.loads(path.read_text())
    if "scenes" not in data:
        root = resolve_path(config["data_root"])
        hashes = {identity: file_hash(root / identity / "geometry.npy") for identity in selected}
        return object_hash({"parameters": file_hash(path), "scenes": hashes})
    for identity in selected:
        entry = data["scenes"][identity]
        if not resolve_path(entry["path"]).is_file():
            raise FileNotFoundError(f"Prepared grasp trajectory missing: {entry['path']}")
    return file_hash(path)


def validate_dataset_geometry(config):
    """Validate both local cfg caches without generating per-batch geometry copies."""
    from bise.modalities.trajectory.factory import build_trajectory_dataset
    for path in config["data_manifests"].values():
        current = dict(config)
        inventory = json.loads(resolve_path(path).read_text())
        current["data_root"] = inventory["data_root"]
        dataset = build_trajectory_dataset(current)
        for index in range(len(dataset)):
            dataset[index]
