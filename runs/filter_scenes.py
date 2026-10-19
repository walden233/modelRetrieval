"""Copy a run-format split manifest, removing only explicitly reviewed exclusions."""

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from bise.common.paths import resolve_path
from bise.data.rh20t.splits import load_split_map, scene_id, validate_split


def reviewed_exclusions(review: Path, cfg: str) -> set[str]:
    exclusions, seen = set(), set()
    with review.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"cfg", "scene_id", "decision", "reason"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Review CSV requires columns: {sorted(required)}")
        for row in reader:
            key = (row["cfg"], row["scene_id"])
            if (key in seen or row["cfg"] not in {"cfg2", "cfg3"}
                    or row["decision"] not in {"keep", "exclude"}
                    or not row["reason"].strip()):
                raise ValueError(f"Invalid or duplicate review row: {key}")
            seen.add(key)
            identity = scene_id(row["scene_id"])
            if identity != row["scene_id"]:
                raise ValueError("Review scene_id must use task_xxxx/scene_x format")
            if row["cfg"] == cfg and row["decision"] == "exclude":
                exclusions.add(identity)
    return exclusions


def filter_splits(mapping: dict, exclusions: set[str]) -> dict:
    """Drop every camera of an excluded scene from every split, preserving order."""
    return {name: [identity for identity in values if scene_id(identity) not in exclusions]
            for name, values in mapping.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, choices=("cfg2", "cfg3"))
    parser.add_argument("--source", required=True, help="Run-format split_manifest.json")
    parser.add_argument("--output", help="A new file; the source and existing files are never overwritten")
    parser.add_argument("--review", default="docs/semantic/SCENE_REVIEW.csv")
    parser.add_argument("--dry-run", action="store_true", help="Show the changes without writing files")
    args = parser.parse_args()
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is used")
    source = resolve_path(args.source)
    output = resolve_path(args.output) if args.output else None
    if output is not None and output == source:
        raise ValueError("The filtered manifest must be a new file, separate from the source")
    mapping = load_split_map(source)
    exclusions = reviewed_exclusions(resolve_path(args.review), args.cfg)
    filtered = filter_splits(mapping, exclusions)
    validate_split(filtered)
    present = {scene_id(identity) for values in mapping.values() for identity in values}
    report = {
        "cfg": args.cfg,
        "excluded_scenes": sorted(present & exclusions),
        "counts": {name: {"before": len(mapping[name]), "after": len(values),
                          "removed": len(mapping[name]) - len(values)}
                   for name, values in filtered.items()},
    }
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(filtered, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        report["output"] = str(output)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
