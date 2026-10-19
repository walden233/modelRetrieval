#!/usr/bin/env python3
"""Zip retrain artifacts, retaining evaluation arrays and excluding model weights."""

import argparse
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SUFFIXES = (
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx",
    ".pt.tar", ".pth.tar", ".ckpt.tar",
)


def collect_files(source: Path) -> tuple[list[Path], int]:
    if not source.is_dir():
        raise NotADirectoryError(source)
    included, excluded = [], 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not supported: {path}")
        if not path.is_file():
            continue
        if path.name.lower().endswith(WEIGHT_SUFFIXES):
            excluded += 1
        else:
            included.append(path)
    return included, excluded


def pack_retrain(source: Path, output: Path | None = None) -> dict:
    source = source.resolve()
    files, excluded = collect_files(source)
    report = {"files": len(files), "excluded_weights": excluded,
              "bytes": sum(path.stat().st_size for path in files)}
    if output is None:
        return report
    output = output.resolve()
    if output.is_relative_to(source):
        raise ValueError("The output archive must be outside the source directory")
    if not files:
        raise ValueError(f"No non-weight files found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6,
                 allowZip64=True) as archive:
        for index, path in enumerate(files, 1):
            name = Path(source.name) / path.relative_to(source)
            archive.write(path, arcname=name.as_posix())
            if index % 100 == 0 or index == len(files):
                print(f"Pack {index}/{len(files)} ({index / len(files):.0%})", flush=True)
    report.update(output=str(output), archive_bytes=output.stat().st_size)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/runs/retrain")
    parser.add_argument("--output", type=Path, help="New ZIP path; existing archives are never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Count files and size without creating an archive")
    args = parser.parse_args()
    output = args.output or (ROOT / "artifacts/exports"
                             / f"retrain_{datetime.now():%Y%m%d_%H%M%S_%f}.zip")
    report = pack_retrain(args.source, None if args.dry_run else output)
    print(f"Include {report['files']} files ({report['bytes'] / 1024 ** 2:.1f} MiB); "
          f"exclude {report['excluded_weights']} weight files")
    if "output" in report:
        print(f"Saved {report['output']} ({report['archive_bytes'] / 1024 ** 2:.1f} MiB)")


if __name__ == "__main__":
    main()
