#!/usr/bin/env python3
"""Zip retrain artifacts; selected batches produce separate artifact/weight archives."""

import argparse
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


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


def pack_batches(source: Path, batches: list[str], output_dir: Path | None = None,
                 artifacts_only: bool = False) -> dict:
    source = source.resolve()
    files = {"artifacts": [], "weights": []}
    if artifacts_only:
        files.pop("weights")
    for batch in dict.fromkeys(batches):
        if Path(batch).name != batch or batch in (".", ".."):
            raise ValueError(f"Invalid batch name: {batch}")
        directory = source / batch
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Batch must be an existing directory: {directory}")
        included, _ = collect_files(directory)
        files["artifacts"].extend(included)
        if not artifacts_only:
            files["weights"].extend(path for path in sorted(directory.rglob("*"))
                                    if path.is_file() and path.name.lower().endswith(WEIGHT_SUFFIXES))
    report = {kind: {"files": len(paths), "bytes": sum(p.stat().st_size for p in paths)}
              for kind, paths in files.items()}
    if output_dir is None:
        return report
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(source):
        raise ValueError("The output archives must be outside the source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{'_'.join(dict.fromkeys(batches))}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    for kind, paths in files.items():
        output = output_dir / f"{prefix}_{kind}.zip"
        temporary = output.with_suffix(".zip.part")
        compression = ZIP_STORED if kind == "weights" else ZIP_DEFLATED
        try:
            with ZipFile(temporary, "x", compression=compression,
                         compresslevel=None if kind == "weights" else 1,
                         allowZip64=True) as archive:
                for index, path in enumerate(paths, 1):
                    name = Path("artifacts/runs/retrain") / path.relative_to(source)
                    archive.write(path, arcname=name.as_posix())
                    if index % 100 == 0 or index == len(paths):
                        print(f"{kind}: {index}/{len(paths)} ({index / len(paths):.0%})", flush=True)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        report[kind].update(output=str(output), archive_bytes=output.stat().st_size)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/runs/retrain")
    parser.add_argument("--output", type=Path, help="New ZIP path; existing archives are never overwritten")
    parser.add_argument("--batches", nargs="+", help="Selected batches; create two ZIPs, artifacts and weights")
    parser.add_argument("--artifacts-only", action="store_true",
                        help="With --batches, create only the non-weight archive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/exports",
                        help="Directory for the two selected-batch archives")
    parser.add_argument("--dry-run", action="store_true", help="Count files and size without creating an archive")
    args = parser.parse_args()
    if args.artifacts_only and not args.batches:
        parser.error("--artifacts-only requires --batches")
    if args.batches:
        if args.output:
            parser.error("Use --output-dir with --batches, not --output")
        report = pack_batches(args.source, args.batches,
                              None if args.dry_run else args.output_dir,
                              artifacts_only=args.artifacts_only)
        for kind, item in report.items():
            print(f"{kind}: {item['files']} files ({item['bytes'] / 1024 ** 2:.1f} MiB)")
            if "output" in item:
                print(f"Saved {item['output']} ({item['archive_bytes'] / 1024 ** 2:.1f} MiB)")
        return
    output = args.output or (ROOT / "artifacts/exports"
                             / f"retrain_{datetime.now():%Y%m%d_%H%M%S_%f}.zip")
    report = pack_retrain(args.source, None if args.dry_run else output)
    print(f"Include {report['files']} files ({report['bytes'] / 1024 ** 2:.1f} MiB); "
          f"exclude {report['excluded_weights']} weight files")
    if "output" in report:
        print(f"Saved {report['output']} ({report['archive_bytes'] / 1024 ** 2:.1f} MiB)")


if __name__ == "__main__":
    main()
