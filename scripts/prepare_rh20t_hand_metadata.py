#!/usr/bin/env python3
"""Copy RH20T human timestamps and camera intrinsics beside subset scenes."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import numpy as np


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def wsl_unc_path(path: Path) -> str:
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        raise RuntimeError("WSL_DISTRO_NAME is required for PowerShell to access WSL files")
    return "\\\\wsl.localhost\\" + distro + str(path.resolve()).replace("/", "\\")


def build_manifest(mapping_path: Path, subset_root: Path, original_root: str) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not mapping.get("source_index_complete"):
        raise ValueError("The raw-scene hash mapping is incomplete")

    scenes = []
    for row in mapping["mappings"]:
        if row.get("status") != "matched" or len(row.get("original_scene_ids", [])) != 1:
            raise ValueError(f"Scene is not uniquely mapped: {row.get('subset_scene_id')}")
        scene_id = row["subset_scene_id"]
        scene_dir = subset_root / scene_id
        cameras = sorted(
            path.name.removesuffix("_human.mp4").removeprefix("cam_")
            for path in scene_dir.glob("cam_*_human.mp4")
        )
        if not cameras:
            raise ValueError(f"No human videos found in {scene_dir}")
        scenes.append(
            {
                "subset_scene_id": scene_id,
                "raw_robot_scene_id": row["original_scene_ids"][0],
                "destination_unc": str(scene_dir) if Path(original_root).is_dir() else wsl_unc_path(scene_dir),
                "cameras": cameras,
            }
        )
    return {
        "original_root": str(Path(original_root).resolve() if Path(original_root).is_dir() else PureWindowsPath(original_root)),
        "subset_root": str(subset_root.resolve()),
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_metadata_locally(manifest: dict, subset_root: Path, original_root: Path) -> list[dict]:
    results = []
    for row in manifest["scenes"]:
        scene_id = row["subset_scene_id"]
        raw_scene_id = row["raw_robot_scene_id"]
        scene_dir = subset_root / scene_id
        robot_scene = original_root / raw_scene_id
        human_scene = original_root / (raw_scene_id + "_human")
        try:
            robot_meta = json.loads((robot_scene / "metadata.json").read_text(encoding="utf-8"))
            human_meta = json.loads((human_scene / "metadata.json").read_text(encoding="utf-8"))
            calib_id = int(robot_meta["calib"])
            source = original_root / "calib" / str(calib_id) / "intrinsics.npy"
            target = scene_dir / "hand_camera_intrinsics.npy"
            shutil.copy2(source, target)
            if file_sha256(source) != file_sha256(target):
                raise ValueError(f"Intrinsics SHA-256 mismatch: {scene_id}")
            timestamp_results = []
            for camera_id in row["cameras"]:
                time_source = human_scene / f"cam_{camera_id}" / "timestamps.npy"
                time_target = scene_dir / f"cam_{camera_id}_human_timestamps.npy"
                if not time_source.is_file():
                    timestamp_results.append({"camera_id": camera_id, "status": "missing_timestamp_source"})
                    continue
                shutil.copy2(time_source, time_target)
                status = "copied" if file_sha256(time_source) == file_sha256(time_target) else "hash_mismatch"
                timestamp_results.append({"camera_id": camera_id, "status": status,
                                          "source_path": str(time_source)})
            metadata = {
                "schema_version": 1,
                "intrinsics_file": target.name,
                "intrinsics_source": str(source),
                "intrinsics_source_scene": raw_scene_id,
                "intrinsics_source_calibration_timestamp": calib_id,
                "intrinsics_source_calibration_quality": robot_meta.get("calib_quality"),
                "intrinsics_calibrated_resolution": [1280, 720],
                "timestamp_source_scene": raw_scene_id + "_human",
                "human_calibration_timestamp": human_meta.get("calib"),
                "human_calibration_quality": human_meta.get("calib_quality"),
                "extrinsics": {"available_for_human": False,
                               "expected_file": "T_base_from_camera.json",
                               "expected_convention": "T_base_from_camera",
                               "note": "Awaiting same-serial robot/human extrinsic alignment"},
            }
            (scene_dir / "hand_camera_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            status = "prepared" if all(x["status"] == "copied" for x in timestamp_results) else "partial"
            results.append({"subset_scene_id": scene_id, "status": status,
                            "intrinsics_sha256": file_sha256(target),
                            "timestamp_results": timestamp_results,
                            "human_calibration_quality": human_meta.get("calib_quality")})
        except Exception as exc:
            results.append({"subset_scene_id": scene_id, "status": "error", "error": str(exc)})
    return results


def copy_metadata_via_powershell(manifest_path: Path, original_root: str) -> list[dict]:
    manifest_unc = wsl_unc_path(manifest_path)
    script = f"""
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath {powershell_quote(manifest_unc)} -Raw | ConvertFrom-Json
$rawRoot = {powershell_quote(original_root)}
$results = @(foreach ($scene in $manifest.scenes) {{
    $robotScene = Join-Path $rawRoot $scene.raw_robot_scene_id
    $humanScene = Join-Path $rawRoot ($scene.raw_robot_scene_id + '_human')
    $destination = $scene.destination_unc
    try {{
        $robotMetadata = Get-Content -LiteralPath (Join-Path $robotScene 'metadata.json') -Raw | ConvertFrom-Json
        $humanMetadata = Get-Content -LiteralPath (Join-Path $humanScene 'metadata.json') -Raw | ConvertFrom-Json
        $calibrationPath = Join-Path (Join-Path $rawRoot 'calib') ([string]$robotMetadata.calib)
        $intrinsicsSource = Join-Path $calibrationPath 'intrinsics.npy'
        $intrinsicsDestination = Join-Path $destination 'hand_camera_intrinsics.npy'
        if ($robotMetadata.calib -eq -1 -or -not (Test-Path -LiteralPath $intrinsicsSource -PathType Leaf)) {{
            throw "Paired robot-scene intrinsics are unavailable (calib=$($robotMetadata.calib))"
        }}
        Copy-Item -LiteralPath $intrinsicsSource -Destination $intrinsicsDestination -Force
        $intrinsicsSourceHash = (Get-FileHash -LiteralPath $intrinsicsSource -Algorithm SHA256).Hash.ToLowerInvariant()
        $intrinsicsHash = (Get-FileHash -LiteralPath $intrinsicsDestination -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($intrinsicsSourceHash -ne $intrinsicsHash) {{ throw 'Intrinsics copy SHA-256 mismatch' }}

        $timestampResults = @(foreach ($camera in $scene.cameras) {{
            $source = Join-Path (Join-Path $humanScene ('cam_' + $camera)) 'timestamps.npy'
            $target = Join-Path $destination ('cam_' + $camera + '_human_timestamps.npy')
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {{
                [PSCustomObject]@{{ camera_id = $camera; status = 'missing_timestamp_source'; source_path = $source }}
                continue
            }}
            Copy-Item -LiteralPath $source -Destination $target -Force
            $sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
            $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
            $status = if ($sourceHash -eq $targetHash) {{ 'copied' }} else {{ 'hash_mismatch' }}
            [PSCustomObject]@{{ camera_id = $camera; status = $status; source_path = $source; source_sha256 = $sourceHash; destination_sha256 = $targetHash }}
        }})

        $metadata = [ordered]@{{
            schema_version = 1
            intrinsics_file = 'hand_camera_intrinsics.npy'
            intrinsics_source = $intrinsicsSource
            intrinsics_source_scene = $scene.raw_robot_scene_id
            intrinsics_source_calibration_timestamp = $robotMetadata.calib
            intrinsics_source_calibration_quality = $robotMetadata.calib_quality
            intrinsics_calibrated_resolution = @(1280, 720)
            timestamp_source_scene = ($scene.raw_robot_scene_id + '_human')
            human_calibration_timestamp = $humanMetadata.calib
            human_calibration_quality = $humanMetadata.calib_quality
            extrinsics = [ordered]@{{
                available_for_human = $false
                expected_file = 'T_base_from_camera.json'
                expected_convention = 'T_base_from_camera'
                note = 'Human-scene calibration is absent or unverified; paired robot extrinsics are not applied.'
            }}
        }}
        $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $destination 'hand_camera_metadata.json') -Encoding UTF8
        $failedTimestamps = @($timestampResults | Where-Object {{ $_.status -ne 'copied' }}).Count
        $sceneStatus = if ($failedTimestamps -eq 0) {{ 'prepared' }} else {{ 'partial' }}
        [PSCustomObject]@{{ subset_scene_id = $scene.subset_scene_id; status = $sceneStatus; intrinsics_sha256 = $intrinsicsHash; timestamp_results = $timestampResults; human_calibration_quality = $humanMetadata.calib_quality }}
    }} catch {{
        [PSCustomObject]@{{ subset_scene_id = $scene.subset_scene_id; status = 'error'; error = $_.Exception.Message }}
    }}
}})
ConvertTo-Json -InputObject $results -Compress -Depth 8
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=3600,
    )
    parsed = json.loads(result.stdout)
    return parsed if isinstance(parsed, list) else [parsed]


def resolve_scene_intrinsics(subset_root: Path) -> dict:
    """Fill missing camera serials from the identical cfg3 intrinsics in other scenes."""
    scene_dirs = sorted(path for path in subset_root.glob("task_*/scene_*") if path.is_dir())
    candidates: dict[str, list[dict]] = {}
    scene_sources = {}
    for scene_dir in scene_dirs:
        source_path = scene_dir / "hand_camera_intrinsics.npy"
        metadata_path = scene_dir / "hand_camera_metadata.json"
        if not source_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing prepared camera metadata in {scene_dir}")
        intrinsics = np.load(source_path, allow_pickle=True).item()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        scene_sources[scene_dir] = (intrinsics, metadata)
        for camera_id, matrix in intrinsics.items():
            matrix = np.asarray(matrix)
            if matrix.shape != (3, 4):
                raise ValueError(f"Invalid intrinsics shape for {camera_id} in {source_path}")
            candidates.setdefault(camera_id, []).append(
                {
                    "matrix": matrix,
                    "source_subset_scene": str(scene_dir.relative_to(subset_root)),
                    "source_raw_scene": metadata["intrinsics_source_scene"],
                    "source_calibration_timestamp": int(metadata["intrinsics_source_calibration_timestamp"]),
                    "source_file": metadata["intrinsics_source"],
                }
            )

    # A fallback is safe here only because all available calibrations for each
    # physical camera serial contain exactly the same intrinsic matrix.
    for camera_id, options in candidates.items():
        reference = options[0]["matrix"]
        if any(not np.array_equal(reference, option["matrix"]) for option in options[1:]):
            raise ValueError(f"Intrinsics differ across calibrations for camera {camera_id}")

    resolved = []
    for scene_dir, (intrinsics, metadata) in scene_sources.items():
        expected = sorted(
            video.name.removeprefix("cam_").removesuffix("_human.mp4")
            for video in scene_dir.glob("cam_*_human.mp4")
        )
        missing = [camera_id for camera_id in expected if camera_id not in intrinsics]
        if not missing:
            continue
        merged = dict(intrinsics)
        camera_sources = {}
        own_timestamp = int(metadata["intrinsics_source_calibration_timestamp"])
        for camera_id in expected:
            if camera_id in intrinsics:
                camera_sources[camera_id] = {
                    "fallback": False,
                    "source_raw_scene": metadata["intrinsics_source_scene"],
                    "source_calibration_timestamp": own_timestamp,
                    "source_file": metadata["intrinsics_source"],
                }
                continue
            options = candidates.get(camera_id)
            if not options:
                raise ValueError(f"No calibrated intrinsics found for camera {camera_id}")
            chosen = min(
                options,
                key=lambda row: (
                    abs(row["source_calibration_timestamp"] - own_timestamp),
                    row["source_subset_scene"],
                ),
            )
            merged[camera_id] = chosen["matrix"]
            camera_sources[camera_id] = {
                "fallback": True,
                "source_subset_scene": chosen["source_subset_scene"],
                "source_raw_scene": chosen["source_raw_scene"],
                "source_calibration_timestamp": chosen["source_calibration_timestamp"],
                "source_file": chosen["source_file"],
            }
        output_path = scene_dir / "hand_camera_intrinsics_resolved.npy"
        with output_path.open("wb") as handle:
            np.save(handle, merged, allow_pickle=True)
        metadata["resolved_intrinsics_file"] = output_path.name
        metadata["intrinsics_by_camera"] = camera_sources
        (scene_dir / "hand_camera_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        resolved.append(
            {
                "scene": str(scene_dir.relative_to(subset_root)),
                "missing_camera_ids": missing,
                "resolved_file": output_path.name,
                "sources": {camera_id: camera_sources[camera_id] for camera_id in missing},
            }
        )
    return {
        "scene_count": len(scene_dirs),
        "calibrated_camera_count": len(candidates),
        "calibration_matrices_identical_per_serial": True,
        "resolved_scene_count": len(resolved),
        "resolved_camera_count": sum(len(row["missing_camera_ids"]) for row in resolved),
        "resolved_scenes": resolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", default=r"D:\RH20T_cfg3")
    parser.add_argument("--subset-root", default="dataset/RH20T_subset/RH20T_cfg3")
    parser.add_argument(
        "--mapping",
        default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hash_mapping.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_metadata_manifest.json",
    )
    parser.add_argument(
        "--resolve-report",
        default="artifacts/audits/trajectory_20260923/rh20t_cfg3_hand_intrinsics_resolution.json",
    )
    parser.add_argument("--resolve-only", action="store_true", help="Use already-copied local intrinsics; no raw drive access.")
    args = parser.parse_args()

    subset_root = Path(args.subset_root).expanduser().resolve()
    if not subset_root.is_dir():
        parser.error(f"Subset root does not exist: {subset_root}")
    resolution_path = Path(args.resolve_report).expanduser().resolve()
    if args.resolve_only:
        resolution = resolve_scene_intrinsics(subset_root)
        resolution_path.parent.mkdir(parents=True, exist_ok=True)
        resolution_path.write_text(json.dumps(resolution, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({key: value for key, value in resolution.items() if key != "resolved_scenes"}, indent=2))
        print(f"Resolution report: {resolution_path}")
        return 0

    mapping_path = Path(args.mapping).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not mapping_path.is_file():
        parser.error(f"Mapping file does not exist: {mapping_path}")

    manifest = build_manifest(mapping_path, subset_root, args.original_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        results = (copy_metadata_locally(manifest, subset_root, Path(args.original_root))
                   if Path(args.original_root).is_dir()
                   else copy_metadata_via_powershell(output_path, args.original_root))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
        manifest["copy_error"] = str(exc)
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Metadata preparation failed: {exc}", file=sys.stderr)
        return 2

    manifest["results"] = results
    counts = {}
    for row in results:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
    manifest["status_counts"] = counts
    if counts.get("prepared") == manifest["scene_count"]:
        resolution = resolve_scene_intrinsics(subset_root)
        resolution_path.parent.mkdir(parents=True, exist_ok=True)
        resolution_path.write_text(json.dumps(resolution, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["intrinsics_resolution_report"] = str(resolution_path)
        manifest["resolved_intrinsics_scene_count"] = resolution["resolved_scene_count"]
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"scene_count": manifest["scene_count"], "status_counts": counts,
                      "manifest": str(output_path)}, indent=2, ensure_ascii=False))
    return 0 if counts.get("prepared") == manifest["scene_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
