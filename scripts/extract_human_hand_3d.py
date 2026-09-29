#!/usr/bin/env python3
"""Extract per-camera MediaPipe hand pose and depth-backed 3D landmarks per scene."""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.data.rh20t.hand_3d_extractor import extract_scene


def run_one_scene(scene_dir: Path, overwrite: bool) -> dict:
    try:
        return extract_scene(scene_dir, overwrite=overwrite)
    except Exception as exc:
        return {"scene": str(scene_dir), "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/RH20T_subset/RH20T_cfg3")
    parser.add_argument("--scene", action="append", help="Scene path relative to data root; repeatable.")
    parser.add_argument("--limit-scenes", type=int, help="Process only the first N sorted scenes (smoke test).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing human_hand_3d.npy files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of independent scene extractors.")
    parser.add_argument(
        "--report",
        default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_3d_extract.json",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"Data root does not exist: {data_root}")
    if args.scene:
        scene_dirs = [data_root / scene for scene in args.scene]
        missing = [str(path) for path in scene_dirs if not path.is_dir()]
        if missing:
            parser.error("Scene path does not exist: " + ", ".join(missing))
    else:
        scene_dirs = sorted(path for path in data_root.glob("task_*/scene_*") if path.is_dir())
    if args.limit_scenes is not None:
        if args.limit_scenes <= 0:
            parser.error("--limit-scenes must be positive")
        scene_dirs = scene_dirs[: args.limit_scenes]
    if not scene_dirs:
        parser.error(f"No scene directories found under {data_root}")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    started = time.monotonic()
    results = []
    if args.workers == 1:
        completed = (run_one_scene(scene_dir, args.overwrite) for scene_dir in scene_dirs)
        for index, result in enumerate(completed, 1):
            results.append(result)
            print(f"[{index}/{len(scene_dirs)}] {json.dumps(result, ensure_ascii=False)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_one_scene, scene_dir, args.overwrite): scene_dir
                       for scene_dir in scene_dirs}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                print(f"[{index}/{len(scene_dirs)}] {json.dumps(result, ensure_ascii=False)}", flush=True)

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report = {
        "data_root": str(data_root),
        "scene_count": len(scene_dirs),
        "status_counts": counts,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "results": results,
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2, ensure_ascii=False))
    print(f"Report: {report_path}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
