#!/usr/bin/env python3
"""Map existing RH20T human hand points from camera coordinates to aligned robot base."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.data.rh20t.hand_3d_extractor import validate_rigid_transform


# RH20T cfg2/cfg3: https://github.com/rh20t/rh20t_api/blob/main/configs/configs.json
# Formula: https://github.com/rh20t/rh20t_api/blob/main/rh20t_api/scene.py
CFG2_IN_HAND_SERIAL = "104422070042"
CFG2_TCP_CAMERA = np.eye(4, dtype=np.float64)
CFG2_TCP_CAMERA[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
CFG2_TCP_CAMERA[:3, 3] = [0, 0.077, 0.2915]
CFG2_ALIGN_BASE = np.eye(4, dtype=np.float64)
IN_HAND_SERIAL = "045322071843"
TCP_CAMERA = np.eye(4, dtype=np.float64)
TCP_CAMERA[:3, 3] = [0.027, 0.0775, 0.240]
ALIGN_BASE = np.diag([-1.0, -1.0, 1.0, 1.0])
CFG_MATRICES = {
    2: (CFG2_IN_HAND_SERIAL, CFG2_TCP_CAMERA, CFG2_ALIGN_BASE),
    3: (IN_HAND_SERIAL, TCP_CAMERA, ALIGN_BASE),
}


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    """RH20T pose: xyz + scalar-first quaternion (w, x, y, z)."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Calibration TCP must be a finite xyz+wxyz 7-vector")
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-8:
        raise ValueError("Calibration TCP quaternion has zero norm")
    w, x, y, z = pose[3:] / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix


def load_extrinsics(calibration_dir: Path) -> dict[str, np.ndarray]:
    loaded = np.load(calibration_dir / "extrinsics.npy", allow_pickle=True).item()
    matrices = {}
    for serial, values in loaded.items():
        matrix = np.asarray(values[0], dtype=np.float64)
        matrices[serial] = validate_rigid_transform(matrix).astype(np.float64)
    return matrices


def aligned_base_from_camera(
    camera_extrinsic: np.ndarray, in_hand_extrinsic: np.ndarray, calib_tcp: np.ndarray, cfg: int = 3
) -> np.ndarray:
    """Invert RH20T's aligned base-to-camera extrinsic (optical coordinates)."""
    _, tcp_camera, align_base = CFG_MATRICES[cfg]
    base_world = np.linalg.inv(in_hand_extrinsic) @ tcp_camera @ np.linalg.inv(pose_matrix(calib_tcp))
    base_to_camera = camera_extrinsic @ base_world @ align_base
    return validate_rigid_transform(np.linalg.inv(base_to_camera))


def reconcile_calibration_tcp_height(calib_tcp: np.ndarray, reference_z: float) -> tuple[np.ndarray, float]:
    """Correct a gross cfg3 calibration TCP z outlier against calibration consensus."""
    tcp = np.asarray(calib_tcp, dtype=np.float64).copy()
    delta = float(tcp[2] - reference_z)
    if abs(delta) > 0.15:
        tcp[2] = reference_z
        return tcp, delta
    return tcp, 0.0


def apply_transform(camera: dict, transform: np.ndarray, source: str, transform_file: Path) -> int:
    points = np.asarray(camera["landmarks_camera_m"])
    valid = np.asarray(camera["landmarks_camera_valid"], dtype=bool)
    base = np.full_like(points, np.nan)
    base[valid] = points[valid] @ transform[:3, :3].T + transform[:3, 3]
    camera["landmarks_base_m"] = base
    camera["landmarks_base_valid"] = valid.copy()

    poses = np.asarray(camera["palm_pose_camera"])
    pose_valid = np.isfinite(poses).all(axis=(2, 3))
    base_poses = np.full_like(poses, np.nan)
    base_poses[pose_valid] = transform @ poses[pose_valid]
    camera["palm_pose_base"] = base_poses
    camera["T_base_from_camera"] = transform
    camera["extrinsics_source"] = source
    camera["extrinsics_file"] = str(transform_file)
    return int(valid.sum())


def process_scene(scene_dir: Path, raw_root: Path, raw_scene_id: str,
                  calibration_index: dict, reference_tcp_z: float, cfg: int) -> dict:
    hand_file = scene_dir / "human_hand_3d.npy"
    hand_data = np.load(hand_file, allow_pickle=True).item()
    metadata = json.loads((raw_root / raw_scene_id / "metadata.json").read_text(encoding="utf-8"))
    calib_id = int(metadata["calib"])
    calib_dir = raw_root / "calib" / str(calib_id)
    own_extrinsics = calibration_index[calib_id]
    in_hand_serial = CFG_MATRICES[cfg][0]
    in_hand_extrinsic = own_extrinsics[in_hand_serial]
    original_calib_tcp = np.load(calib_dir / "tcp.npy")
    calib_tcp, tcp_height_correction = reconcile_calibration_tcp_height(original_calib_tcp, reference_tcp_z)
    transforms = {}
    fallback_count = 0
    for serial in hand_data["cameras"]:
        source_calib = calib_id
        if serial not in own_extrinsics:
            choices = [number for number, matrices in calibration_index.items() if serial in matrices]
            if not choices:
                raise KeyError(f"No cfg{cfg} calibration has camera {serial}")
            source_calib = min(choices, key=lambda number: (abs(number - calib_id), number))
            fallback_count += 1
        # Raw extrinsics use a calibration marker frame. A fallback must use
        # the same calibration's in-hand camera and TCP, not this scene's marker frame.
        source_dir = raw_root / "calib" / str(source_calib)
        source_extrinsics = calibration_index[source_calib]
        source_in_hand = (in_hand_extrinsic if source_calib == calib_id
                          else source_extrinsics[in_hand_serial])
        if source_calib == calib_id:
            source_tcp, source_correction = calib_tcp, tcp_height_correction
        else:
            source_tcp, source_correction = reconcile_calibration_tcp_height(
                np.load(source_dir / "tcp.npy"), reference_tcp_z
            )
        transform = aligned_base_from_camera(
            source_extrinsics[serial], source_in_hand, source_tcp, cfg=cfg
        )
        transforms[serial] = {
            "matrix": transform.astype(float).tolist(),
            "source": f"{source_dir / 'extrinsics.npy'}; "
                      f"{source_dir / 'tcp.npy'}; cfg{cfg} RH20T API config"
                      + (f"; calibration TCP z corrected by {-source_correction:.3f} m from calibration consensus"
                         if source_correction else ""),
            "camera_extrinsic_calibration": source_calib,
            "robot_base_calibration": calib_id,
            "fallback_camera_extrinsic": source_calib != calib_id,
            "source_calibration_tcp_z_correction_m": -source_correction,
        }
    transform_file = scene_dir / "T_base_from_camera.json"
    transform_data = {
        "convention": "T_base_from_camera",
        "coordinate_frame": f"RH20T cfg{cfg} aligned robot base, same convention as transformed/tcp_base.npy",
        "assumption": "Human and robot fixed cameras with the same serial share extrinsics",
        "raw_robot_scene": raw_scene_id,
        "robot_calibration_quality": metadata.get("calib_quality"),
        "calibration_tcp_z_original_m": float(original_calib_tcp[2]),
        "calibration_tcp_z_used_m": float(calib_tcp[2]),
        "calibration_tcp_z_correction_m": -tcp_height_correction,
        "cameras": transforms,
    }
    # Finish derivation for every camera before touching any output file.
    transform_temp = transform_file.with_suffix(".json.tmp")
    transform_temp.write_text(json.dumps(transform_data, indent=2), encoding="utf-8")
    transform_temp.replace(transform_file)
    valid_points = 0
    for serial, camera in hand_data["cameras"].items():
        row = transforms[serial]
        valid_points += apply_transform(
            camera, np.asarray(row["matrix"], dtype=np.float32), row["source"], transform_file
        )
    hand_data["coordinate_frames"]["base"] = (
        f"RH20T cfg{cfg} aligned robot base; derived from paired robot calibration by camera serial"
    )
    extrinsics_metadata = {
        "available_for_human": True,
        "file": transform_file.name,
        "assumption": transform_data["assumption"],
        "robot_calibration_timestamp": calib_id,
        "robot_calibration_quality": metadata.get("calib_quality"),
        "fallback_camera_count": fallback_count,
        "calibration_tcp_z_correction_m": -tcp_height_correction,
    }
    hand_data["camera_calibration_metadata"]["extrinsics"] = extrinsics_metadata
    metadata_file = scene_dir / "hand_camera_metadata.json"
    sidecar = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
    sidecar["extrinsics"] = extrinsics_metadata
    sidecar_temp = metadata_file.with_suffix(".json.tmp")
    sidecar_temp.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
    sidecar_temp.replace(metadata_file)
    hand_temp = hand_file.with_suffix(".npy.tmp")
    with hand_temp.open("wb") as handle:
        np.save(handle, hand_data, allow_pickle=True)
    hand_temp.replace(hand_file)
    return {"scene": str(scene_dir), "status": "updated", "camera_count": len(transforms),
            "fallback_camera_count": fallback_count, "base_landmarks": valid_points,
            "calibration_tcp_z_correction_m": -tcp_height_correction}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=int, choices=(2, 3), default=3)
    parser.add_argument("--data-root", default="dataset/RH20T_subset/RH20T_cfg3")
    parser.add_argument("--raw-root", default="/mnt/d/RH20T_cfg3")
    parser.add_argument("--mapping", default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hash_mapping.json")
    parser.add_argument("--scene", action="append", help="Relative subset scene path; repeatable")
    parser.add_argument("--report", default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_base.json")
    args = parser.parse_args()
    root, raw_root = Path(args.data_root).resolve(), Path(args.raw_root).resolve()
    mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    if not mapping.get("source_index_complete"):
        raise ValueError("Raw-scene hash mapping is incomplete")
    mapped = {row["subset_scene_id"]: row["original_scene_ids"][0] for row in mapping["mappings"]
              if row["status"] == "matched" and len(row["original_scene_ids"]) == 1}
    calibration_ids = sorted({
        int(json.loads((root / scene_id / "hand_camera_metadata.json").read_text(encoding="utf-8-sig"))
            ["intrinsics_source_calibration_timestamp"])
        for scene_id in mapped
    })
    calibration_index = {
        number: load_extrinsics(raw_root / "calib" / str(number))
        for number in calibration_ids
    }
    calibration_tcp_heights = np.asarray([
        np.load(raw_root / "calib" / str(number) / "tcp.npy")[2]
        for number in calibration_index
    ], dtype=float)
    reference_tcp_z = float(np.median(calibration_tcp_heights))
    if np.count_nonzero(np.abs(calibration_tcp_heights - reference_tcp_z) < 0.01) < len(calibration_tcp_heights) * 0.8:
        raise ValueError(f"No strong cfg{args.cfg} calibration TCP height consensus")
    scenes = args.scene or sorted(mapped)
    results = []
    for scene_id in scenes:
        try:
            result = process_scene(root / scene_id, raw_root, mapped[scene_id], calibration_index,
                                   reference_tcp_z, args.cfg)
        except Exception as exc:
            result = {"scene": scene_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"[{len(results)}/{len(scenes)}] {scene_id}: {result['status']}", flush=True)
    summary = {"cfg": args.cfg, "scene_count": len(scenes), "reference_calibration_tcp_z_m": reference_tcp_z,
               "updated": sum(r["status"] == "updated" for r in results),
               "failed": sum(r["status"] == "error" for r in results),
               "cameras": sum(r.get("camera_count", 0) for r in results),
               "fallback_cameras": sum(r.get("fallback_camera_count", 0) for r in results),
               "corrected_tcp_scenes": sum(bool(r.get("calibration_tcp_z_correction_m", 0)) for r in results),
               "base_landmarks": sum(r.get("base_landmarks", 0) for r in results), "results": results}
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Report: {report}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
