#!/usr/bin/env python3
"""Copy matched RH20T robot and human depth videos into a subset."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath


def powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def wsl_unc_path(path: Path) -> str:
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro:
        raise RuntimeError("WSL_DISTRO_NAME is required for PowerShell to access the subset")
    return "\\\\wsl.localhost\\" + distro + str(path.resolve()).replace("/", "\\")


def build_copy_manifest(mapping_path: Path, subset_root: Path, original_root: str) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not mapping.get("source_index_complete"):
        raise ValueError("The scene hash mapping is incomplete")

    files = []
    pair_count = 0
    for row in mapping["mappings"]:
        if row.get("status") != "matched" or len(row.get("original_scene_ids", [])) != 1:
            raise ValueError(f"Scene is not uniquely mapped: {row.get('subset_scene_id')}")

        scene_id = row["subset_scene_id"]
        scene_dir = subset_root / scene_id
        robot_cameras = {
            path.name.removesuffix("_robot.mp4")
            for path in scene_dir.glob("*_robot.mp4")
        }
        human_cameras = {
            path.name.removesuffix("_human.mp4")
            for path in scene_dir.glob("*_human.mp4")
        }
        if not robot_cameras or robot_cameras != human_cameras:
            raise ValueError(f"Robot/human camera mismatch in {scene_dir}")

        original_scene_id = row["original_scene_ids"][0]
        local_source = Path(original_root).is_dir()
        original_root_path = Path(original_root) if local_source else PureWindowsPath(original_root)
        robot_source_scene = original_root_path / original_scene_id
        human_source_scene = original_root_path / (original_scene_id + "_human")
        destination_unc = None if local_source else wsl_unc_path(scene_dir)
        for camera_id in sorted(robot_cameras):
            pair_count += 1
            for role, source_scene in (
                ("robot", robot_source_scene),
                ("human", human_source_scene),
            ):
                source = source_scene / camera_id / "depth.mp4"
                destination = scene_dir / f"{camera_id}_{role}_depth.mp4"
                destination_windows = str(destination) if local_source else destination_unc + "\\" + destination.name
                files.append(
                    {
                        "subset_scene_id": scene_id,
                        "original_scene_id": original_scene_id,
                        "camera_id": camera_id,
                        "role": role,
                        "source_path": str(source),
                        "destination_path": destination_windows,
                    }
                )

    return {
        "method": "copy raw per-camera depth.mp4 for the matched robot scene and its _human partner",
        "mapping_file": str(mapping_path.resolve()),
        "subset_root": str(subset_root.resolve()),
        "scene_count": len(mapping["mappings"]),
        "camera_pair_count": pair_count,
        "file_count": len(files),
        "files": files,
    }


def run_local_copy(files: list[dict]) -> list[dict]:
    results = []
    for item in files:
        source = Path(item["source_path"])
        destination = Path(item["destination_path"])
        row = dict(item)
        if not source.is_file():
            row["status"] = "missing_source"
        elif destination.is_file():
            row["status"] = "already_exact" if source.stat().st_size == destination.stat().st_size and file_sha256(source) == file_sha256(destination) else "destination_conflict"
        else:
            shutil.copy2(source, destination)
            row["status"] = "copied" if source.stat().st_size == destination.stat().st_size and file_sha256(source) == file_sha256(destination) else "hash_mismatch"
        if row["status"] in ("copied", "already_exact"):
            row["size_bytes"] = destination.stat().st_size
            row["sha256"] = file_sha256(destination)
        results.append(row)
    return results


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_powershell_copy(manifest_unc: str) -> list[dict]:
    command = f"""
$ErrorActionPreference = 'Stop'
$manifest = Get-Content -LiteralPath {powershell_quote(manifest_unc)} -Raw | ConvertFrom-Json
$results = @(foreach ($item in $manifest.files) {{
    try {{
        if (-not (Test-Path -LiteralPath $item.source_path -PathType Leaf)) {{
            [PSCustomObject]@{{ subset_scene_id = $item.subset_scene_id; camera_id = $item.camera_id; role = $item.role; source_path = $item.source_path; destination_path = $item.destination_path; status = 'missing_source' }}
            continue
        }}
        $sourceHash = (Get-FileHash -LiteralPath $item.source_path -Algorithm SHA256).Hash.ToLowerInvariant()
        if (Test-Path -LiteralPath $item.destination_path -PathType Leaf) {{
            $destinationHash = (Get-FileHash -LiteralPath $item.destination_path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($sourceHash -ne $destinationHash) {{
                [PSCustomObject]@{{ subset_scene_id = $item.subset_scene_id; camera_id = $item.camera_id; role = $item.role; source_path = $item.source_path; destination_path = $item.destination_path; status = 'destination_conflict'; source_sha256 = $sourceHash; destination_sha256 = $destinationHash }}
                continue
            }}
            $status = 'already_exact'
        }} else {{
            Copy-Item -LiteralPath $item.source_path -Destination $item.destination_path
            $destinationHash = (Get-FileHash -LiteralPath $item.destination_path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($sourceHash -ne $destinationHash) {{
                [PSCustomObject]@{{ subset_scene_id = $item.subset_scene_id; camera_id = $item.camera_id; role = $item.role; source_path = $item.source_path; destination_path = $item.destination_path; status = 'hash_mismatch'; source_sha256 = $sourceHash; destination_sha256 = $destinationHash }}
                continue
            }}
            $status = 'copied'
        }}
        $length = (Get-Item -LiteralPath $item.destination_path).Length
        [PSCustomObject]@{{ subset_scene_id = $item.subset_scene_id; camera_id = $item.camera_id; role = $item.role; source_path = $item.source_path; destination_path = $item.destination_path; status = $status; size_bytes = $length; sha256 = $sourceHash }}
    }} catch {{
        [PSCustomObject]@{{ subset_scene_id = $item.subset_scene_id; camera_id = $item.camera_id; role = $item.role; source_path = $item.source_path; destination_path = $item.destination_path; status = 'error'; error = $_.Exception.Message }}
    }}
}})
ConvertTo-Json -InputObject $results -Compress -Depth 5
"""
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=7200,
    )
    results = json.loads(completed.stdout)
    return results if isinstance(results, list) else [results]


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
        default="artifacts/audits/trajectory_20260923/rh20t_cfg3_depth_copy_manifest.json",
    )
    args = parser.parse_args()

    subset_root = Path(args.subset_root).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not subset_root.is_dir():
        parser.error(f"Subset root does not exist: {subset_root}")
    if not mapping_path.is_file():
        parser.error(f"Mapping file does not exist: {mapping_path}")

    manifest = build_copy_manifest(mapping_path, subset_root, args.original_root)
    manifest["original_root_requested"] = args.original_root
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        results = (run_local_copy(manifest["files"]) if Path(args.original_root).is_dir()
                   else run_powershell_copy(wsl_unc_path(output_path)))
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        manifest["copy_error"] = str(exc)
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Depth copy failed: {exc}", file=sys.stderr)
        return 2

    status_counts: dict[str, int] = {}
    for result in results:
        status = result.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    manifest["copy_results"] = results
    manifest["copy_status_counts"] = status_counts
    manifest["verified_file_count"] = sum(
        status_counts.get(status, 0) for status in ("copied", "already_exact")
    )
    manifest["missing_source_count"] = status_counts.get("missing_source", 0)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key not in ("files", "copy_results")},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Depth copy manifest: {output_path}")
    resolved_count = manifest["verified_file_count"] + manifest["missing_source_count"]
    return 0 if resolved_count == manifest["file_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
