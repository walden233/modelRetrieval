#!/usr/bin/env python3
"""Add persistent hand IDs to existing per-scene human_hand_3d.npy files."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.data.rh20t.hand_3d_extractor import track_hands_by_wrist, wrist_middle_span_quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/RH20T_subset/RH20T_cfg3")
    parser.add_argument("--report", default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_tracking.json")
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    scene_dirs = sorted(path for path in root.glob("task_*/scene_*") if path.is_dir())
    results = []
    for scene in scene_dirs:
        path = scene / "human_hand_3d.npy"
        if not path.is_file():
            results.append({"scene": str(scene.relative_to(root)), "status": "missing"})
            continue
        data = np.load(path, allow_pickle=True).item()
        ambiguous_count = 0
        detected_count = 0
        for camera in data["cameras"].values():
            ids, ambiguous = track_hands_by_wrist(
                camera["hand_count"], camera["handedness_label"],
                camera["landmarks_image_px"], camera["image_size_wh"],
            )
            camera["hand_track_id"] = ids
            camera["hand_track_ambiguous"] = ambiguous
            ratio, suspect = wrist_middle_span_quality(
                camera["landmarks_camera_m"], camera["landmarks_mediapipe_local_m"]
            )
            camera["wrist_middle_span_ratio"] = ratio
            camera["wrist_middle_span_suspect"] = suspect
            ambiguous_count += int(ambiguous.sum())
            detected_count += int(camera["hand_count"].sum())
        data["schema_version"] = 2
        data["coordinate_frames"]["hand_tracking"] = (
            "hand_track_id 0/1 follows 2D wrist motion and raw MediaPipe Left/Right labels; "
            "ambiguous frames are flagged"
        )
        temporary_path = path.with_suffix(".npy.tmp")
        with temporary_path.open("wb") as handle:
            np.save(handle, data, allow_pickle=True)
        temporary_path.replace(path)
        results.append({"scene": str(scene.relative_to(root)), "status": "updated",
                        "detected_hands": detected_count, "ambiguous_hands": ambiguous_count})

    counts = {"updated": sum(row["status"] == "updated" for row in results),
              "missing": sum(row["status"] == "missing" for row in results)}
    report = {"data_root": str(root), "scene_count": len(scene_dirs), "status_counts": counts,
              "detected_hands": sum(row.get("detected_hands", 0) for row in results),
              "ambiguous_hands": sum(row.get("ambiguous_hands", 0) for row in results),
              "results": results}
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    print(f"Report: {report_path}")
    return 1 if counts["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
