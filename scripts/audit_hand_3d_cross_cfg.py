"""Read-only diagnostics for selected hand tracks, cross-view geometry, and calibration provenance."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bise.data.rh20t.trajectory_manifest import load_trajectory_file


def _track_observations(camera, track_id):
    points = np.asarray(camera["landmarks_base_m"])
    clean = (np.asarray(camera["landmarks_base_valid"]).all(axis=-1)
             & np.isfinite(points).all(axis=(-1, -2))
             & ~np.asarray(camera["hand_track_ambiguous"])
             & ~np.asarray(camera["wrist_middle_span_suspect"]))
    selected = clean & (np.asarray(camera["hand_track_id"]) == track_id)
    frame, slot = np.nonzero(selected)
    return {"time_ms": np.asarray(camera["timestamp_ms"])[frame],
            "wrist_base_m": points[frame, slot, 0], "frame": frame, "slot": slot}


def _matched_wrist_residuals(first, second, tolerance_ms=100):
    a, b = first["time_ms"], second["time_ms"]
    if len(a) < 5 or len(b) < 5:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    right = np.searchsorted(b, a).clip(0, len(b) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(b[left] - a) <= np.abs(b[right] - a), left, right)
    matched = np.abs(b[nearest] - a) <= tolerance_ms
    residual = first["wrist_base_m"][matched] - second["wrist_base_m"][nearest[matched]]
    return residual, np.abs(b[nearest[matched]] - a[matched])


def _quantiles(values):
    return {"count": len(values), "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))} if values else {"count": 0}


def audit_cfg(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest["data_root"])
    cameras, pairs, examples = [], [], []
    for scene in manifest["scenes"]:
        scene_path = root / scene["scene_id"]
        human, _ = load_trajectory_file(scene_path / "human_hand_3d.npy", scene["human_sha256"])
        robot, _ = load_trajectory_file(scene_path / "tcp_base.npy", scene["robot_sha256"])
        calibration = json.loads((scene_path / "T_base_from_camera.json").read_text())
        tracks_by_camera = {}
        for camera_id, quality in scene["cameras"].items():
            camera = human["cameras"][camera_id]
            tracks = {track_id: _track_observations(camera, track_id) for track_id in (0, 1)}
            tracks_by_camera[camera_id] = tracks
            selected_id = quality["selected_track_id"]
            chosen, alternate = tracks[selected_id], tracks[1 - selected_id]
            tcp = np.asarray([row["tcp"][:3] for row in robot[camera_id]])
            tree = cKDTree(tcp)
            chosen_nearest = float(np.median(tree.query(chosen["wrist_base_m"])[0]))
            alternate_nearest = (float(np.median(tree.query(alternate["wrist_base_m"])[0]))
                                 if len(alternate["frame"]) >= 5 else None)
            sidecar = calibration["cameras"][camera_id]
            selected_depth_delta = np.asarray(camera["depth_time_delta_ms"])[chosen["frame"]]
            selected_mad = np.asarray(camera["depth_sample_mad_m"])[chosen["frame"], chosen["slot"]]
            finite_mad = selected_mad[np.isfinite(selected_mad)]
            cameras.append({
                "scene_id": scene["scene_id"], "camera_id": camera_id,
                "frames": quality["human_frames"], "clean_frames": quality["observed_frames"],
                "coverage": quality["coverage"], "selected_track_id": selected_id,
                "other_clean_frames": len(alternate["frame"]),
                "multi_hand_frames": quality["multi_hand_frames"],
                "fallback_extrinsic": bool(sidecar["fallback_camera_extrinsic"]),
                "tcp_height_corrected": bool(sidecar["source_calibration_tcp_z_correction_m"]),
                "robot_wrist_nearest_median_m": chosen_nearest,
                "other_robot_wrist_nearest_median_m": alternate_nearest,
                "depth_abs_time_delta_median_ms": float(np.median(np.abs(selected_depth_delta))),
                "depth_mad_median_m": float(np.median(finite_mad)) if len(finite_mad) else None,
                "wrist_base_median_m": np.median(chosen["wrist_base_m"], axis=0).tolist(),
                "robot_tcp_base_median_m": np.median(tcp, axis=0).tolist(),
            })
        for camera_a, camera_b in itertools.combinations(tracks_by_camera, 2):
            tracks_a, tracks_b = tracks_by_camera[camera_a], tracks_by_camera[camera_b]
            selected_a = scene["cameras"][camera_a]["selected_track_id"]
            selected_b = scene["cameras"][camera_b]["selected_track_id"]
            choices = []
            for track_a, track_b in itertools.product((0, 1), repeat=2):
                residual, time_delta_ms = _matched_wrist_residuals(tracks_a[track_a], tracks_b[track_b])
                if len(residual) >= 5:
                    choices.append((float(np.median(np.linalg.norm(residual, axis=-1))),
                                    len(residual), track_a, track_b,
                                    np.median(residual, axis=0).tolist(),
                                    float(np.median(time_delta_ms))))
            if not choices:
                continue
            best = min(choices)
            selected = next((row for row in choices if row[2:4] == (selected_a, selected_b)), None)
            row = {"scene_id": scene["scene_id"], "cameras": [camera_a, camera_b],
                   "best_median_m": best[0], "best_match_count": best[1],
                   "best_tracks": [best[2], best[3]],
                   "best_median_offset_xyz_m": best[4],
                   "selected_tracks": [selected_a, selected_b],
                   "selected_median_m": selected[0] if selected else None,
                   "selected_median_offset_xyz_m": selected[4] if selected else None,
                   "selected_median_abs_time_delta_ms": selected[5] if selected else None,
                   "selected_match_count": selected[1] if selected else 0,
                   "any_fallback": bool(calibration["cameras"][camera_a]["fallback_camera_extrinsic"]
                                        or calibration["cameras"][camera_b]["fallback_camera_extrinsic"])}
            pairs.append(row)
            if selected and selected[0] - best[0] >= .10 and best[0] <= .10:
                examples.append({**row, "improvement_m": selected[0] - best[0]})

    selected_pairs = [p["selected_median_m"] for p in pairs if p["selected_median_m"] is not None]
    if not selected_pairs:
        raise AssertionError("No selected cross-view tracks matched; check audit tuple indexing")
    dual = [c for c in cameras if c["other_clean_frames"] >= 5]
    alternate_closer = [c for c in dual if c["other_robot_wrist_nearest_median_m"] + .10
                        < c["robot_wrist_nearest_median_m"]]
    summary = {
        "scenes": len(manifest["scenes"]), "cameras": len(cameras),
        "coverage": _quantiles([c["coverage"] for c in cameras]),
        "selected_clean_frames": _quantiles([c["clean_frames"] for c in cameras]),
        "dual_clean_track_cameras": len(dual),
        "dual_alternate_10cm_closer_to_robot_workspace": len(alternate_closer),
        "fallback_cameras": sum(c["fallback_extrinsic"] for c in cameras),
        "tcp_height_corrected_cameras": sum(c["tcp_height_corrected"] for c in cameras),
        "selected_robot_wrist_nearest_m": _quantiles([c["robot_wrist_nearest_median_m"] for c in cameras]),
        "selected_depth_abs_time_delta_ms": _quantiles([c["depth_abs_time_delta_median_ms"] for c in cameras]),
        "selected_depth_mad_m": _quantiles([c["depth_mad_median_m"] for c in cameras
                                            if c["depth_mad_median_m"] is not None]),
        "crossview_pair_count": len(pairs),
        "crossview_selected_comparable_count": len(selected_pairs),
        "crossview_selected_wrist_median_m": _quantiles(selected_pairs),
        "crossview_selected_abs_time_delta_ms": _quantiles([
            p["selected_median_abs_time_delta_ms"] for p in pairs
            if p["selected_median_abs_time_delta_ms"] is not None]),
        "crossview_best_track_wrist_median_m": _quantiles([p["best_median_m"] for p in pairs]),
        "crossview_possible_track_mismatch_count": len(examples),
        "crossview_possible_track_mismatch_examples": sorted(
            examples, key=lambda x: -x["improvement_m"])[:20],
    }
    return {"summary": summary, "cameras": cameras, "crossview_pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", default="artifacts/manifests/trajectory_hand_3d_v1")
    parser.add_argument("--output", default="artifacts/audits/trajectory_hand_3d_cross_cfg.json")
    args = parser.parse_args()
    result = {cfg: audit_cfg(Path(args.manifest_dir) / f"{cfg}_valid.json")
              for cfg in ("cfg2", "cfg3")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({cfg: {key: value for key, value in values["summary"].items()
                            if key != "crossview_possible_track_mismatch_examples"}
                      for cfg, values in result.items()}, indent=2))


if __name__ == "__main__":
    main()
