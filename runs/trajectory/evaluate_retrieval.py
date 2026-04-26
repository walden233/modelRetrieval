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
from bise.data import RH20TTrajectoryDataset, collate_trajectories
from bise.modalities.trajectory import CrossModalTrajectoryModel, evaluate_retrieval_grouped


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics from a saved checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RH20TTrajectoryDataset(
        root_dir=config["data_root"],
        use_6_keypoints=config.get("use_6_keypoints", False),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_trajectories,
    )
    model = CrossModalTrajectoryModel(**config["model_params"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    metrics = evaluate_retrieval_grouped(
        model,
        dataloader,
        device,
        group_by_task=config.get("evaluate_task_positives", False),
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
