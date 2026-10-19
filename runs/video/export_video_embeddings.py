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

from bise.common import EmbeddingSample, load_json_config
from bise.retrieval.feature_store import FeatureStore
from bise.modalities.video import build_retrieval_cases, evaluate_video_retrieval
from bise.modalities.video.factory import build_video_dataset, build_video_model, split_video_dataset
from bise.modalities.video.loading import video_loader_options


def parse_args():
    parser = argparse.ArgumentParser(description="Export video embeddings and retrieval cases.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint path.")
    parser.add_argument("--output", required=True, help="Output JSON path for embedding samples.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to export.")
    parser.add_argument("--split-manifest", help="Optional split_manifest.json produced by train_video.py.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    from bise.common.device import experiment_device
    device = experiment_device(config)
    split_config = _resolve_split_config(config, args.checkpoint, args.split_manifest)
    if config.get("protocol") == "retrain_v2":
        from bise.common.evaluation import evaluation_inputs
        evaluation_inputs(config, args.checkpoint, "video", split_config["manifest_path"], args.split)
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
        **video_loader_options(config["training"], device),
    )
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    result = evaluate_video_retrieval(model, dataloader, device)

    samples = []
    for index, sample_id in enumerate(result["metadata"]["sample_ids"]):
        samples.append(
            EmbeddingSample(
                sample_id=sample_id,
                task_id=result["metadata"]["task_ids"][index],
                scene_id=result["metadata"]["scene_ids"][index],
                video_embedding=result["robot_embeddings"][index].tolist(),
                metadata={
                    "human_video_path": result["metadata"]["human_video_paths"][index],
                    "robot_video_path": result["metadata"]["robot_video_paths"][index],
                    "camera_id": result["metadata"]["camera_ids"][index],
                },
            )
        )
    FeatureStore(args.output).save(samples)
    np.save(f"{args.output}.similarity.npy", result["similarity_matrix"])
    retrieval_cases = build_retrieval_cases(result, top_k=5)
    with open(f"{args.output}.cases.json", "w", encoding="utf-8") as handle:
        json.dump(retrieval_cases, handle, indent=2, ensure_ascii=False)


def _resolve_split_config(config: dict, checkpoint: str, explicit_manifest: str | None):
    from bise.common.evaluation import resolve_eval_split
    return resolve_eval_split(config, checkpoint, explicit_manifest, False)[0]


if __name__ == "__main__":
    main()
