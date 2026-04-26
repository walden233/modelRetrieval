import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.modalities.video import build_retrieval_cases, collate_video_pairs, evaluate_video_retrieval
from bise.modalities.video.factory import build_video_dataset, build_video_model, split_video_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained video retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cases to export in stdout.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    processor, model = build_video_model(config["model"])
    eval_source = build_video_dataset(config["dataset"], processor=processor, is_train=False)
    split_datasets = split_video_dataset(eval_source, config.get("split"))
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
    print(json.dumps({"metrics": result["metrics"], "cases": cases[:5]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
