"""Collect configured evaluation outputs into one paper-ready directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_path(source: Path, target: Path, records: list[dict[str, Any]]) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        files = [item for item in target.rglob("*") if item.is_file()]
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files = [target]
    for item in files:
        records.append({
            "source": str(source),
            "target": str(item),
            "size": item.stat().st_size,
            "sha256": sha256(item),
        })


def export_bundle(manifest_path: str | Path, bundle_config: dict[str, Any] | None = None) -> Path:
    manifest_path = resolve(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = bundle_config or {}
    output_dir = resolve(config.get("output_dir", f"artifacts/paper/{manifest['experiment_id']}"))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    selected = set(config.get("include_steps") or [])
    copied_step_sources: set[Path] = set()
    for step in manifest.get("steps", []):
        if selected and step.get("name") not in selected:
            continue
        if step.get("status") == "failed":
            continue
        for output in step.get("outputs", []):
            source = resolve(output)
            # A declared output is usually metrics.json; bundle the whole
            # evaluation directory beside it so cases, matrices and PNGs are
            # retained without listing every generated file in the config.
            use_parent = str(step.get("bundle_source", "parent")).lower() != "file"
            bundle_source = source.parent if source.is_file() and use_parent else source
            if bundle_source in copied_step_sources or not bundle_source.exists():
                continue
            copied_step_sources.add(bundle_source)
            target = output_dir / "steps" / str(step["name"])
            copy_path(bundle_source, target, records)

    for item in config.get("files", []):
        source = resolve(item["source"])
        target = output_dir / str(item.get("target") or source.name)
        copy_path(source, target, records)

    for item in config.get("patterns", []):
        root = resolve(item["root"])
        target_root = output_dir / str(item.get("target") or root.name)
        patterns = item.get("patterns") or ["*"]
        for pattern in patterns:
            for source in root.glob(str(pattern)):
                relative = source.relative_to(root)
                copy_path(source, target_root / relative, records)

    copied_manifest = output_dir / "pipeline_manifest.json"
    shutil.copy2(manifest_path, copied_manifest)
    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "source_manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "files": records,
    }
    (output_dir / "bundle_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Paper bundle: {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", help="Optional pipeline config; defaults to the config recorded in the manifest.")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    manifest = json.loads(resolve(args.manifest).read_text(encoding="utf-8"))
    config = {}
    config_path = resolve(args.config or manifest.get("config", "")) if (args.config or manifest.get("config")) else None
    if config_path and config_path.exists():
        pipeline_config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(pipeline_config.get("bundle") or {})
    if args.output_dir:
        config["output_dir"] = args.output_dir
    export_bundle(args.manifest, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
