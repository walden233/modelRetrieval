"""Add gallery_size to saved retrain evaluations without recomputing scores."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.common.fingerprints import file_hash, write_json
from bise.retrieval.metrics import calculate_label_retrieval_metrics


def labels(metadata, domain, level):
    key = f"{domain}_{level}_ids"
    return np.asarray(metadata[key] if key in metadata else metadata[f"{level}_ids"])


def backfill(root: Path, *, apply: bool = False, evaluator_hash_migration: dict | None = None) -> dict:
    plans = []
    paths = sorted(p for p in root.rglob("metrics.json") if p.parent.parent.name == "evaluations")
    # Validate the entire batch before writing any original result.
    for path in paths:
        original = json.loads(path.read_text(encoding="utf-8"))
        updated = copy.deepcopy(original)
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        matrix = np.load(path.parent / "similarity_matrix.npy", mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError(f"Similarity matrix must be 2D: {path}")
        additions = []
        counts = []
        for direction, query_domain, gallery_domain, scores in (
            ("human_to_robot", "human", "robot", matrix),
            ("robot_to_human", "robot", "human", matrix.T),
        ):
            query_size, gallery_size = scores.shape
            for level, metrics in updated[direction].items():
                queries = labels(metadata, query_domain, level)
                gallery = labels(metadata, gallery_domain, level)
                if len(queries) != query_size or len(gallery) != gallery_size:
                    raise ValueError(f"Metadata count disagrees with matrix: {path}:{direction}:{level}")
                valid_queries = int(np.isin(queries, gallery).sum())
                if metrics["valid_queries"] != valid_queries:
                    raise ValueError(f"Recorded valid_queries disagrees with labels: {path}:{direction}:{level}")
                if "gallery_size" in metrics and metrics["gallery_size"] != gallery_size:
                    raise ValueError(f"Recorded gallery_size disagrees with matrix: {path}:{direction}:{level}")
                reproduced = calculate_label_retrieval_metrics(scores, queries, gallery)
                if any(not np.isclose(value, reproduced[key], rtol=1e-10, atol=1e-12, equal_nan=True)
                       for key, value in metrics.items()):
                    raise ValueError(f"Recorded scores disagree with matrix: {path}:{direction}:{level}")
                if "gallery_size" not in metrics:
                    metrics["gallery_size"] = int(gallery_size)
                    additions.append(f"{direction}.{level}.gallery_size")
                counts.append({"direction": direction, "level": level,
                               "gallery_size": int(gallery_size), "valid_queries": valid_queries})
        info_path = path.parent / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else None
        before_hash = file_hash(path)
        artifact_hash = (info or {}).get("evaluation", {}).get("artifacts", {}).get("metrics.json")
        if artifact_hash is not None and artifact_hash != before_hash:
            raise ValueError(f"Existing metrics fingerprint is already inconsistent: {path}")
        old_evaluator = (info or {}).get("evaluation", {}).get("evaluator_sha256")
        new_evaluator = (evaluator_hash_migration or {}).get(old_evaluator)
        plans.append({"path": path, "original": original, "updated": updated,
                      "before_hash": before_hash, "info_path": info_path,
                      "info": info, "info_hash": file_hash(info_path) if info is not None else None,
                      "artifact_hash_present": artifact_hash is not None,
                      "old_evaluator": old_evaluator, "new_evaluator": new_evaluator,
                      "additions": additions, "counts": counts})
    entries = []
    for plan in plans:
        if apply and plan["additions"]:
            path = plan["path"]
            if file_hash(path) != plan["before_hash"]:
                raise ValueError(f"Metrics changed during backfill: {path}")
            if plan["info"] is not None and file_hash(plan["info_path"]) != plan["info_hash"]:
                raise ValueError(f"Run info changed during backfill: {plan['info_path']}")
            write_json(path, plan["updated"])
            if plan["artifact_hash_present"]:
                info = copy.deepcopy(plan["info"])
                info["evaluation"]["artifacts"]["metrics.json"] = file_hash(path)
                if plan["new_evaluator"]:
                    evaluation = info["evaluation"]
                    evaluation["evaluator_sha256"] = plan["new_evaluator"]
                    evaluation.setdefault("metric_schema_migrations", []).append({
                        "added_fields": ["gallery_size"], "source": "similarity_matrix.npy.shape",
                        "evaluator_sha256_before": plan["old_evaluator"],
                        "evaluator_sha256_after": plan["new_evaluator"],
                        "existing_metrics_reproduced": True})
                write_json(plan["info_path"], info)
        entries.append({"path": str(plan["path"]), "added_fields": plan["additions"],
                        "counts": plan["counts"], "metrics_sha256_before": plan["before_hash"],
                        "metrics_sha256_after": file_hash(plan["path"]),
                        "run_info_sha256_before": plan["info_hash"],
                        "run_info_sha256_after": file_hash(plan["info_path"]) if plan["info"] is not None else None,
                        "evaluator_sha256_before": plan["old_evaluator"],
                        "evaluator_sha256_after": (plan["new_evaluator"] if apply and plan["additions"]
                                                    and plan["artifact_hash_present"] and plan["new_evaluator"]
                                                    else plan["old_evaluator"]),
                        "artifact_fingerprint_updated": bool(apply and plan["additions"] and plan["artifact_hash_present"])})
    return {"applied": apply, "evaluation_files_checked": len(plans),
            "evaluation_files_needing_additions": sum(bool(p["additions"]) for p in plans),
            "fields_added" if apply else "fields_to_add": sum(len(p["additions"]) for p in plans),
            "evaluator_fingerprints_migrated" if apply else "evaluator_fingerprints_to_migrate": sum(
                bool(p["additions"] and p["artifact_hash_present"] and p["new_evaluator"]) for p in plans),
            "historical_evaluator_fingerprints_preserved": True, "files": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/runs/retrain")
    parser.add_argument("--apply", action="store_true", help="Apply the validated additions; default is a dry run.")
    parser.add_argument("--report", type=Path, help="Write the detailed audit JSON to this path.")
    parser.add_argument("--evaluator-hash-migration", type=Path,
                        help="Optional JSON mapping of a verified previous evaluator fingerprint to its additive-schema successor.")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Evaluation root does not exist: {args.root}")
    migration = json.loads(args.evaluator_hash_migration.read_text()) if args.evaluator_hash_migration else None
    report = backfill(args.root, apply=args.apply, evaluator_hash_migration=migration)
    if args.report:
        write_json(args.report, report)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
