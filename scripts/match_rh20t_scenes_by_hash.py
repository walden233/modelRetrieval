#!/usr/bin/env python3
"""Map renamed RH20T subset scenes to raw scene-directory names by copied TCP hashes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath


def resolve_root(value: str) -> Path | None:
    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()
    windows = PureWindowsPath(value)
    if windows.drive and os.name != "nt":
        drive = windows.drive[0].lower()
        relative = Path(*windows.parts[1:])
        for candidate in (Path("/mnt") / drive / relative,
                          Path("/media") / drive / relative):
            if candidate.is_dir():
                return candidate.resolve()
    return None


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def index_raw_scenes(root: Path) -> tuple[dict[str, list[str]], int, list[dict]]:
    matches: dict[str, list[str]] = {}
    indexed = 0
    errors = []
    for directory, subdirs, filenames in os.walk(root):
        subdirs.sort()
        if Path(directory).name != "transformed" or "tcp_base.npy" not in filenames:
            continue
        tcp_path = Path(directory) / "tcp_base.npy"
        scene_path = Path(directory).parent
        try:
            digest = sha256(tcp_path)
        except OSError as exc:
            errors.append({"path": str(tcp_path), "error": str(exc)})
            continue
        matches.setdefault(digest, []).append(scene_path.name)
        indexed += 1
    return matches, indexed, errors


def index_raw_scenes_via_powershell(
    root: str,
) -> tuple[dict[str, list[str]], int, list[dict]]:
    """Read a Windows drive from WSL when it is not mounted under /mnt."""
    if "\n" in root or "\r" in root:
        raise ValueError("Windows raw root cannot contain newlines")
    escaped_root = root.replace("'", "''")
    command = f"""
$ErrorActionPreference = 'Stop'
$root = '{escaped_root}'
$files = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter 'tcp_base.npy' |
    Where-Object {{ $_.Directory.Name -eq 'transformed' }})
$records = @(foreach ($file in $files) {{
    try {{
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        [PSCustomObject]@{{ sha256 = $hash; scene_id = $file.Directory.Parent.Name }}
    }} catch {{
        [PSCustomObject]@{{ error = $_.Exception.Message; path = $file.FullName }}
    }}
}})
ConvertTo-Json -InputObject $records -Compress
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
        timeout=3600,
    )
    records = json.loads(result.stdout)
    if isinstance(records, dict):
        records = [records]
    matches: dict[str, list[str]] = {}
    errors = []
    indexed = 0
    for record in records:
        if "error" in record:
            errors.append({"path": record.get("path", ""),
                           "error": record["error"]})
            continue
        matches.setdefault(record["sha256"], []).append(record["scene_id"])
        indexed += 1
    return matches, indexed, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", default=r"D:\RH20T_cfg2",
                        help="Raw cfg2 root; Windows paths map to /mnt/<drive> under WSL.")
    parser.add_argument("--subset-root", default="dataset/RH20T_subset/RH20T_cfg2")
    parser.add_argument("--output", default="artifacts/audits/rh20t_cfg2_hash_mapping.json")
    args = parser.parse_args()
    started = time.monotonic()
    subset_root = Path(args.subset_root).expanduser()
    if not subset_root.is_dir():
        parser.error(f"Subset root does not exist: {subset_root}")

    subset_rows = []
    for tcp_path in sorted(subset_root.glob("task_*/scene_*/tcp_base.npy")):
        try:
            digest = sha256(tcp_path)
            error = None
        except OSError as exc:
            digest, error = None, str(exc)
        subset_rows.append({"subset_scene_id": f"{tcp_path.parent.parent.name}/{tcp_path.parent.name}",
                            "tcp_sha256": digest, "error": error})

    original_root = resolve_root(args.original_root)
    raw_index, indexed, raw_errors = ({}, 0, [])
    source_available = original_root is not None
    if original_root is not None:
        raw_index, indexed, raw_errors = index_raw_scenes(original_root)
    elif os.name != "nt" and PureWindowsPath(args.original_root).drive:
        try:
            raw_index, indexed, raw_errors = index_raw_scenes_via_powershell(
                args.original_root
            )
            source_available = True
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError,
                KeyError, ValueError) as exc:
            raw_errors = [{"path": args.original_root,
                           "error": f"Windows PowerShell indexing failed: {exc}"}]

    rows = []
    counts = {"matched": 0, "ambiguous": 0, "unmatched": 0, "unreadable": 0}
    for row in subset_rows:
        if row["error"]:
            status, candidates = "unreadable", []
        elif not source_available:
            status, candidates = "original_root_unavailable", []
        else:
            candidates = sorted(set(raw_index.get(row["tcp_sha256"], [])))
            status = "matched" if len(candidates) == 1 else (
                "ambiguous" if candidates else "unmatched")
        if status in counts:
            counts[status] += 1
        rows.append({**row, "status": status, "original_scene_ids": candidates})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "exact SHA-256 of copied tcp_base.npy",
        "subset_root": str(subset_root.resolve()),
        "original_root_requested": args.original_root,
        "original_root_resolved": (str(original_root) if original_root else
                                   (f"{args.original_root} (via PowerShell)"
                                    if source_available else None)),
        "source_index_complete": source_available and not raw_errors,
        "subset_scene_count": len(subset_rows),
        "raw_tcp_files_indexed": indexed,
        "counts": counts,
        "raw_hash_errors": raw_errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "mappings": rows,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "mappings"},
                     indent=2, ensure_ascii=False))
    print(f"Mapping details: {output}")
    if not source_available:
        print("Raw root is unavailable here; subset hashes were saved for a later rerun.",
              file=sys.stderr)
        return 2
    return 1 if raw_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
