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
from bise.modalities.video import build_retrieval_cases, collate_video_pairs, evaluate_video_retrieval
from bise.modalities.video.figures import keep_first_camera_per_scene, plot_similarity_heatmap, plot_sorted_similarity_heatmap
from bise.modalities.video.factory import build_video_dataset, build_video_model, split_video_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained video retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cases to export in stdout.")
    parser.add_argument("--output-dir", help="Optional directory for metrics, cases, embeddings, and similarity matrix.")
    parser.add_argument("--split-manifest", help="Optional split_manifest.json produced by train_video.py.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    split_config, split_manifest_path = _resolve_split_config(config, args.checkpoint, args.split_manifest)
    processor, model = build_video_model(config["model"])
    eval_source = build_video_dataset(config["dataset"], processor=processor, is_train=False)
    split_datasets = split_video_dataset(eval_source, split_config)
    dataset = split_datasets.get(args.split)
    if dataset is None or len(dataset) == 0:
        raise ValueError(f"No samples available for split={args.split!r}.")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        collate_fn=collate_video_pairs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    result = evaluate_video_retrieval(model, dataloader, device)
    cases = build_retrieval_cases(result, top_k=args.top_k)
    if args.output_dir:
        _save_evaluation_outputs(Path(args.output_dir), result, cases, split_manifest_path)
    print(json.dumps({"metrics": result["metrics"], "cases": cases[:5]}, indent=2, ensure_ascii=False))


def _resolve_split_config(config: dict, checkpoint: str, explicit_manifest: str | None):
    split_config = dict(config.get("split") or {})
    manifest_path = Path(explicit_manifest) if explicit_manifest else Path(checkpoint).resolve().parent / "split_manifest.json"
    if explicit_manifest and not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if manifest_path.exists():
        split_config["manifest_path"] = str(manifest_path)
        return split_config, manifest_path
    return split_config, None


def _save_evaluation_outputs(output_dir: Path, result: dict, cases: list[dict], split_manifest_path: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(_json_ready_metadata(result["metadata"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / "similarity_matrix.npy", result["similarity_matrix"])
    np.save(output_dir / "human_embeddings.npy", result["human_embeddings"])
    np.save(output_dir / "robot_embeddings.npy", result["robot_embeddings"])
    _save_heatmaps(output_dir, result)
    if split_manifest_path is not None:
        (output_dir / "run_info.json").write_text(
            json.dumps({"split_manifest": str(split_manifest_path)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _save_heatmaps(output_dir: Path, result: dict) -> None:
    matrix, metadata = keep_first_camera_per_scene(
        result["similarity_matrix"],
        _json_ready_metadata(result["metadata"]),
    )
    plot_similarity_heatmap(matrix, output_dir / "similarity_heatmap.png", max_items=120)
    plot_sorted_similarity_heatmap(matrix, metadata, output_dir / "task_scene_sorted_similarity_heatmap.png", max_items=120)


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
