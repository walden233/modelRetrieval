"""Search cfg2 fusion weights and evaluate the fixed choice on cfg3 features."""

import argparse
import csv
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.common.paths import resolve_path
from bise.retrieval.system import evaluate_retrieval_system, load_retrieval_library
from bise.retrieval.system.io import load_json, save_json


MODALITIES = ("video", "trajectory", "semantic_text")
METRICS = ("R@1", "R@5", "R@10", "MRR", "NDCG@10", "valid_queries")


def build_weight_grid(step=0.1):
    if not np.isfinite(step) or not 0 < step <= 1:
        raise ValueError("step must be finite and in (0, 1].")
    divisions = round(1.0 / step)
    if not np.isclose(divisions * step, 1.0, rtol=0, atol=1e-9):
        raise ValueError("step must divide 1 exactly, for example 0.1 or 0.05.")
    grid = [
        (i / divisions, j / divisions, (divisions - i - j) / divisions)
        for i in range(divisions + 1)
        for j in range(divisions - i + 1)
    ]
    uniform = (1 / 3, 1 / 3, 1 / 3)
    if uniform not in grid:
        grid.append(uniform)
    return [dict(zip(MODALITIES, values)) for values in grid]


def split_scene_id(sample_id):
    # Existing video splits use rh20t::task::scene::camera; trajectories use task/scene.
    parts = str(sample_id).split("::")
    if len(parts) == 4 and parts[1].startswith("task_"):
        return f"{parts[1]}/{parts[2]}"
    if len(parts) == 1 and len(parts[0].split("/")) == 2:
        return parts[0]
    raise ValueError(f"Unsupported scene/sample ID in split manifest: {sample_id!r}")


def filter_search_queries(library):
    split_sets = {}
    sources = {}
    for modality in ("video", "trajectory"):
        checkpoint = library.config.get(modality, {}).get("checkpoint")
        if not checkpoint:
            raise ValueError(f"Library config has no {modality} checkpoint.")
        path = resolve_path(checkpoint).parent / "split_manifest.json"
        manifest = load_json(path)
        if not manifest or not manifest.get("train"):
            raise ValueError(f"Missing or empty training split: {path}")
        train = {split_scene_id(value) for value in manifest["train"]}
        known = {
            split_scene_id(value)
            for name in ("train", "val", "test")
            for value in manifest.get(name, [])
        }
        split_sets[modality] = (train, known)
        sources[modality] = str(path)

    queries = []
    excluded = {"missing_modalities": [], "training_scene": [], "unknown_scene": []}
    for query in library.query_items:
        if not set(MODALITIES).issubset(library.item_feature_map(query)):
            excluded["missing_modalities"].append(query.query_id)
        elif any(query.scene_id in train for train, _ in split_sets.values()):
            excluded["training_scene"].append(query.query_id)
        elif any(query.scene_id not in known for _, known in split_sets.values()):
            excluded["unknown_scene"].append(query.query_id)
        else:
            queries.append(query)
    if not queries:
        raise ValueError("No complete, non-training queries remain in the search library.")
    audit = {
        "split_manifests": sources,
        "original_query_count": len(library.query_items),
        "selected_query_count": len(queries),
        "selected_scene_count": len({query.scene_id for query in queries}),
        "query_ids": [query.query_id for query in queries],
        "excluded_query_ids": excluded,
        "excluded_counts": {reason: len(ids) for reason, ids in excluded.items()},
        "policy": "Complete queries outside both model training splits; gallery unchanged.",
    }
    return replace(library, query_items=queries), audit


def weighted_config(config, weights, library_dir):
    result = deepcopy(config)
    result["library_dir"] = str(library_dir)
    result["level"] = "task"
    result["modalities"] = list(MODALITIES)
    result.setdefault("fusion", {})["weights"] = dict(weights)
    return result


def evaluate_weights(library, config, weights):
    # Reuse the exact camera scoring -> calibration -> fusion -> scene aggregation path.
    result = evaluate_retrieval_system(
        library,
        config=weighted_config(config, weights, library.root_dir),
        level="task",
        require_modalities=list(MODALITIES),
        top_k=10,
    )
    metrics = result["metrics"]["task"]
    if not metrics["valid_queries"] or not np.isfinite(metrics["MRR"]):
        raise ValueError(f"No valid task retrieval queries in {library.root_dir}.")
    return result


def selection_key(row):
    distance = sum((row[name] - 1 / 3) ** 2 for name in MODALITIES)
    return (-row["MRR"], -row["NDCG@10"], distance, *(row[name] for name in MODALITIES))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation(output_dir, result, config):
    save_json(output_dir / "metrics.json", result["metrics"])
    save_json(output_dir / "cases.json", result["cases"])
    save_json(output_dir / "ordered_ids.json", {
        "query_ids": result["query_ids"], "gallery_ids": result["gallery_ids"],
    })
    save_json(output_dir / "config.json", config)
    np.save(output_dir / "fused_similarity_matrix.npy", result["similarity_matrix"])


def plot_search(rows, best, output_dir):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(
        [row["video"] for row in rows], [row["trajectory"] for row in rows],
        c=[row["MRR"] for row in rows], cmap="viridis", s=95,
    )
    ax.scatter(best["video"], best["trajectory"], marker="*", s=320,
               facecolors="none", edgecolors="red", linewidths=1.8, label="Selected weights")
    ax.set(xlabel="Video weight", ylabel="Trajectory weight",
           title="Fusion weight search (semantic weight = 1 - video - trajectory)")
    ax.legend()
    fig.colorbar(points, ax=ax, label="cfg2 tuning task MRR")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_search.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-library", default="artifacts/retrieval/rh20t_cfg2_v1")
    parser.add_argument("--test-library", default="artifacts/retrieval/rh20t_cfg3_v1")
    parser.add_argument("--config", default="configs/retrieval/system_default.json")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--output-dir", default="artifacts/retrieval/weight_search_cfg2_to_cfg3")
    args = parser.parse_args()
    grid = build_weight_grid(args.step)
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty; choose a new --output-dir: {output_dir}")

    config = load_json_config(args.config)
    search_library = load_retrieval_library(resolve_path(args.search_library))
    test_library = load_retrieval_library(resolve_path(args.test_library))
    if not search_library.config.get("cfg") or not test_library.config.get("cfg"):
        parser.error("Both libraries must declare cfg in library_config.json.")
    if search_library.config["cfg"] == test_library.config["cfg"]:
        parser.error("Search and final-test libraries must use different configurations.")
    for modality in ("video", "trajectory"):
        search_checkpoint = search_library.config.get(modality, {}).get("checkpoint")
        test_checkpoint = test_library.config.get(modality, {}).get("checkpoint")
        if not search_checkpoint or not test_checkpoint:
            parser.error(f"Both libraries must declare the {modality} checkpoint.")
        if resolve_path(search_checkpoint) != resolve_path(test_checkpoint):
            parser.error(f"Libraries use different {modality} checkpoints; hold models fixed.")

    search_library, audit = filter_search_queries(search_library)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "query_filter.json", audit)
    save_json(output_dir / "run_info.json", {
        "search_library": str(search_library.root_dir),
        "test_library": str(test_library.root_dir),
        "search_library_config": search_library.config,
        "test_library_config": test_library.config,
        "config": config, "step": args.step, "weight_count": len(grid),
        "selection": "task MRR, then NDCG@10, then nearest uniform, then lexicographic weights",
    })
    print(f"Tuning: {len(search_library.query_items)} queries, "
          f"{audit['selected_scene_count']} scenes; {len(grid)} weight combinations.", flush=True)

    rows = []
    best_row = best_result = uniform_result = None
    uniform = dict.fromkeys(MODALITIES, 1 / 3)
    for index, weights in enumerate(grid, 1):
        result = evaluate_weights(search_library, config, weights)
        row = {**weights, **{key: result["metrics"]["task"][key] for key in METRICS}}
        rows.append(row)
        if best_row is None or selection_key(row) < selection_key(best_row):
            best_row, best_result = row, result
        if weights == uniform:
            uniform_result = result
        print(f"[{index}/{len(grid)}] {weights} MRR={row['MRR']:.6f}", flush=True)
    best_weights = {name: best_row[name] for name in MODALITIES}
    write_csv(output_dir / "weight_search.csv", rows)
    save_json(output_dir / "weight_search.json", rows)
    save_json(output_dir / "best_weights.json", {
        "weights": best_weights, "selection_metric": "task.MRR",
        "search_metrics": best_result["metrics"]["task"],
    })
    best_config = weighted_config(config, best_weights, test_library.root_dir)
    save_json(output_dir / "best_config.json", best_config)

    comparison = []
    for name, weights, search_result in (
        ("uniform", uniform, uniform_result), ("selected", best_weights, best_result),
    ):
        save_evaluation(output_dir / "cfg2_tuning" / name, search_result,
                        weighted_config(config, weights, search_library.root_dir))
        final_result = evaluate_weights(test_library, config, weights)
        save_evaluation(output_dir / "cfg3_test" / name, final_result,
                        weighted_config(config, weights, test_library.root_dir))
        comparison.append({
            "method": name, **weights,
            **{key: final_result["metrics"]["task"][key] for key in METRICS},
            "query_count": final_result["query_count"],
            "gallery_count": final_result["gallery_count"],
        })
    write_csv(output_dir / "final_comparison.csv", comparison)
    save_json(output_dir / "final_comparison.json", comparison)
    plot_search(rows, best_weights, output_dir)
    print(f"Selected weights: {best_weights}\nResults: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
