import argparse
import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import ensure_directory, load_json_config, merge_overrides, save_run_artifacts
from bise.data import RH20TTrajectoryDataset, collate_trajectories
from bise.modalities.trajectory import (
    CrossModalTrajectoryModel,
    evaluate_retrieval_grouped,
    pretrain_intra_modal_epoch,
    train_augmented_trajectory_epoch,
    train_trajectory_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the trajectory retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--data-root", help="Optional dataset root override.")
    parser.add_argument("--output-dir", help="Optional output dir override.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    config = merge_overrides(config, {"data_root": args.data_root, "output_dir": args.output_dir})

    dataset = RH20TTrajectoryDataset(
        root_dir=config["data_root"],
        use_6_keypoints=config.get("use_6_keypoints", False),
    )
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_trajectories,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalTrajectoryModel(**config["model_params"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    run_name = f'{config["experiment_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = ensure_directory(Path(config["output_dir"]) / run_name)
    history = {"train_loss": [], "train_loss_inter": [], "train_loss_intra": [], "val_mean_p_rank": []}
    best_result = None
    best_score = float("inf")
    mode = config.get("mode", "standard")

    if mode == "two_stage":
        for epoch in range(config["pretrain_epochs"]):
            intra_loss = pretrain_intra_modal_epoch(model, train_loader, optimizer, device)
            history["train_loss_intra"].append(intra_loss)
            history["train_loss"].append(None)
            history["train_loss_inter"].append(None)
            history["val_mean_p_rank"].append(None)
            print(f"Pretrain Epoch {epoch + 1}/{config['pretrain_epochs']}: intra_loss={intra_loss:.4f}")

    finetune_epochs = config.get("finetune_epochs", config.get("num_epochs", 0))
    loop_epochs = finetune_epochs if mode == "two_stage" else config["num_epochs"]

    for epoch in range(loop_epochs):
        if mode == "augment" or mode == "two_stage":
            train_loss, inter_loss, intra_loss = train_augmented_trajectory_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config.get("intra_loss_weight", 1.0),
                use_task_labels=config.get("train_task_positives", False),
            )
            history["train_loss_inter"].append(inter_loss)
            history["train_loss_intra"].append(intra_loss)
        else:
            train_loss = train_trajectory_epoch(
                model,
                train_loader,
                optimizer,
                device,
                use_task_labels=config.get("train_task_positives", False),
            )

        result = evaluate_retrieval_grouped(
            model,
            val_loader,
            device,
            group_by_task=config.get("evaluate_task_positives", False),
        )
        history["train_loss"].append(train_loss)
        history["val_mean_p_rank"].append(result["mean_percentage_rank"])

        if result["mean_percentage_rank"] < best_score:
            best_score = result["mean_percentage_rank"]
            best_result = result
            torch.save(model.state_dict(), run_dir / "best_model.pth")

        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "metrics": result}, ensure_ascii=False))

    torch.save(model.state_dict(), run_dir / "last_model.pth")
    save_run_artifacts(run_dir, config, history, best_result)


if __name__ == "__main__":
    main()
