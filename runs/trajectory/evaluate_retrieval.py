import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.data import RH20TTrajectoryDataset, collate_trajectories
from bise.modalities.trajectory import CrossModalTrajectoryModel, build_trajectory_retrieval_cases, evaluate_trajectory_retrieval
from bise.modalities.trajectory.factory import split_trajectory_dataset
from bise.modalities.video.figures import keep_first_camera_per_scene, plot_similarity_heatmap, plot_sorted_similarity_heatmap


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics from a saved checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--split-manifest", help="Optional split_manifest.json produced by train_trajectory.py.")
    parser.add_argument("--output-dir", help="Optional directory for metrics, cases, embeddings, and similarity matrix.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cases to export in stdout.")
    parser.add_argument("--data-root", help="Optional trajectory dataset root override, for example dataset/RH20T_subset/RH20T_cfg3.")
    parser.add_argument("--all-as-test", action="store_true", help="Evaluate the entire dataset as the test split.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    split_config, split_manifest_path = _resolve_split_config(config, args.checkpoint, args.split_manifest, args.all_as_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_source = RH20TTrajectoryDataset(
        root_dir=config["data_root"],
        use_6_keypoints=config.get("use_6_keypoints", False),
    )
    split_datasets = split_trajectory_dataset(dataset_source, split_config)
    dataset = split_datasets.get(args.split)
    if dataset is None or len(dataset) == 0:
        raise ValueError(f"No samples available for split={args.split!r}.")

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_trajectories,
    )
    model = CrossModalTrajectoryModel(**config["model_params"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    result = evaluate_trajectory_retrieval(model, dataloader, device)
    cases = build_trajectory_retrieval_cases(result, top_k=args.top_k)
    if args.output_dir:
        _save_evaluation_outputs(Path(args.output_dir), result, cases, split_manifest_path, split_config, config["data_root"])
    print(json.dumps({"metrics": result["metrics"], "cases": cases[:5]}, indent=2, ensure_ascii=False))


def _resolve_split_config(config: dict, checkpoint: str, explicit_manifest: str | None, all_as_test: bool):
    if all_as_test and explicit_manifest:
        raise ValueError("--all-as-test cannot be combined with --split-manifest.")

    split_config = dict(config.get("split") or {"unit": "scene", "seed": config.get("seed", 42), "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}})
    if all_as_test:
        return {"all_as_test": True}, None
    if split_config.get("all_as_test") or str(split_config.get("unit", "")).strip().lower() == "all_test":
        return split_config, None

    manifest_path = Path(explicit_manifest) if explicit_manifest else Path(checkpoint).resolve().parent / "split_manifest.json"
    if explicit_manifest and not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if manifest_path.exists():
        split_config["manifest_path"] = str(manifest_path)
        return split_config, manifest_path
    return split_config, None


def _save_evaluation_outputs(
    output_dir: Path,
    result: dict,
    cases: list[dict],
    split_manifest_path: Path | None,
    split_config: dict,
    data_root: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = _json_ready_metadata(result["metadata"])
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / "similarity_matrix.npy", result["similarity_matrix"])
    np.save(output_dir / "human_embeddings.npy", result["human_embeddings"])
    np.save(output_dir / "robot_embeddings.npy", result["robot_embeddings"])
    _save_heatmaps(output_dir, result["similarity_matrix"], metadata)
    run_info = {"split_config": split_config, "data_root": data_root}
    if split_manifest_path is not None:
        run_info["split_manifest"] = str(split_manifest_path)
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_heatmaps(output_dir: Path, similarity_matrix, metadata: dict) -> None:
    heatmap_metadata = {
        "sample_ids": [f"{scene_id}/{camera_id}" for scene_id, camera_id in zip(metadata["human_scene_ids"], metadata["human_camera_ids"])],
        "scene_ids": metadata["human_scene_ids"],
        "task_ids": metadata["human_task_ids"],
        "camera_ids": metadata["human_camera_ids"],
    }
    try:
        matrix, filtered_metadata = keep_first_camera_per_scene(similarity_matrix, heatmap_metadata)
        plot_similarity_heatmap(matrix, output_dir / "similarity_heatmap.png", max_items=120, modality_label="Trajectory")
        plot_sorted_similarity_heatmap(
            matrix,
            filtered_metadata,
            output_dir / "task_scene_sorted_similarity_heatmap.png",
            max_items=120,
            modality_label="Trajectory",
        )
    except ValueError:
        plot_similarity_heatmap(similarity_matrix, output_dir / "similarity_heatmap.png", max_items=120, modality_label="Trajectory")


def _json_ready_metadata(metadata: dict) -> dict:
    serializable = {}
    for key, value in metadata.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif isinstance(value, list):
            serializable[key] = [item.tolist() if hasattr(item, "tolist") else item for item in value]
        else:
            serializable[key] = value
    return serializable


if __name__ == "__main__":
    main()
