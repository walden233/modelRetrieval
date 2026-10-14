"""Validate trajectory files before splitting; never substitute sample identities."""

from pathlib import Path

import numpy as np

from .scanner import SceneRecord


SCHEMA_VERSION = 1


def load_trajectory_file(path):
    return np.load(path, allow_pickle=True).item()


def validate_camera(human_frames, robot_frames):
    if not human_frames or not robot_frames:
        raise ValueError("empty human or robot sequence")
    frame_ids = np.asarray([frame["frame_index"] for frame in human_frames], dtype=float)
    if not np.isfinite(frame_ids).all() or (np.diff(frame_ids) <= 0).any():
        raise ValueError("human frame indices must be finite and strictly increasing")
    good = [frame["hands_landmarks"][0] for frame in human_frames if frame.get("hands_landmarks")]
    if not good:
        raise ValueError("no detected hand")
    poses = np.asarray(good)
    if poses.shape[1:] != (21, 3) or not np.isfinite(poses).all():
        raise ValueError("human landmarks must be finite 21x3 arrays")
    tcp = np.asarray([frame["tcp"] for frame in robot_frames])
    time = np.asarray([frame["timestamp"] for frame in robot_frames], dtype=float)
    if tcp.shape != (len(robot_frames), 7) or not np.isfinite(tcp).all():
        raise ValueError("robot TCP must be finite 7D poses")
    if (np.linalg.norm(tcp[:, 3:], axis=-1) < 1e-6).any():
        raise ValueError("zero quaternion")
    if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("robot timestamps must be finite and strictly increasing")
    return {"human_frames": len(human_frames), "detected_frames": len(good),
            "robot_frames": len(robot_frames), "coverage": len(good) / len(human_frames)}


def build_trajectory_manifest(root_dir: str) -> dict:
    entries, exclusions = [], []
    for path in sorted(Path(root_dir).glob("task_*/scene_*")):
        if not path.is_dir():
            continue
        scene = SceneRecord(scene_path=str(path), human_pose_path=str(path / "human_pose.npy"),
                            tcp_base_path=str(path / "tcp_base.npy"))
        scene_id = f"{path.parent.name}/{path.name}"
        try:
            human = load_trajectory_file(scene.human_pose_path)
            robot = load_trajectory_file(scene.tcp_base_path)
        except Exception as exc:
            exclusions.append({"scene_id": scene_id, "reason": str(exc)})
            continue
        cameras = {}
        for camera in sorted(human.keys() & robot.keys()):
            try:
                cameras[camera] = validate_camera(human[camera], robot[camera])
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                exclusions.append({"scene_id": scene_id, "camera_id": camera, "reason": str(exc)})
        if not cameras:
            exclusions.append({"scene_id": scene_id, "reason": "no valid paired camera"})
            continue
        entries.append({"scene_id": scene_id, "task_id": Path(scene.scene_path).parent.name,
                        "cameras": cameras})
    return {"schema_version": SCHEMA_VERSION, "data_root": str(Path(root_dir).resolve()),
            "human_time_unit": "frame_index", "robot_time_unit": "seconds_from_timestamp_ms",
            "task_identity": "local_subset_id_not_cross_cfg_semantic_id",
            "scenes": entries, "exclusions": exclusions}
