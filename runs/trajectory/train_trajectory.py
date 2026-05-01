import argparse
import datetime
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import ensure_directory, load_json_config, merge_overrides, save_run_artifacts
from bise.data import RH20TTrajectoryDataset, collate_trajectories
from bise.modalities.trajectory import (
    CrossModalTrajectoryModel,
    build_trajectory_retrieval_cases,
    evaluate_trajectory_retrieval,
    pretrain_intra_modal_epoch,
    train_augmented_trajectory_epoch,
    train_trajectory_epoch,
)
from bise.modalities.trajectory.factory import build_split_manifest, split_trajectory_dataset


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
    _set_seed(config.get("seed", 42))

    run_name = f'{config["experiment_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = ensure_directory(Path(config["output_dir"]) / run_name)

    dataset = RH20TTrajectoryDataset(
        root_dir=config["data_root"],
        use_6_keypoints=config.get("use_6_keypoints", False),
    )
    split_config = config.get("split") or {"unit": "scene", "seed": config.get("seed", 42), "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}}
    split_datasets = split_trajectory_dataset(dataset, split_config)
    split_manifest = build_split_manifest(split_datasets)
    split_manifest_path = run_dir / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)
    config["split"] = dict(split_config)
    config["split"]["manifest_path"] = str(split_manifest_path)

    train_loader = DataLoader(
        split_datasets["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = None
    if split_datasets.get("val") is not None and len(split_datasets["val"]) > 0:
        val_loader = DataLoader(
            split_datasets["val"],
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_trajectories,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalTrajectoryModel(**config["model_params"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    config.setdefault("checkpoint_selection", _default_checkpoint_selection())

    history = {"train_loss": [], "train_loss_inter": [], "train_loss_intra": [], "val_mean_p_rank": [], "val_mrr": [], "val_ndcg": []}
    best_result = None
    best_score = float("-inf")
    mode = config.get("mode", "standard")

    if mode == "two_stage":
        for epoch in range(config["pretrain_epochs"]):
            intra_loss = pretrain_intra_modal_epoch(
                model,
                train_loader,
                optimizer,
                device,
                intra_task_positive_weight=config.get("intra_task_positive_weight", 0.0),
                augmentation_noise_std=config.get("augmentation_noise_std", 0.005),
                augmentation_max_rotation_degrees=config.get("augmentation_max_rotation_degrees", 10.0),
            )
            history["train_loss_intra"].append(intra_loss)
            history["train_loss"].append(None)
            history["train_loss_inter"].append(None)
            history["val_mean_p_rank"].append(None)
            history["val_mrr"].append(None)
            history["val_ndcg"].append(None)
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
                intra_task_positive_weight=config.get("intra_task_positive_weight", 0.0),
                augmentation_noise_std=config.get("augmentation_noise_std", 0.005),
                augmentation_max_rotation_degrees=config.get("augmentation_max_rotation_degrees", 10.0),
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

        result = evaluate_trajectory_retrieval(model, val_loader, device) if val_loader is not None else None
        primary_metrics = _extract_primary_metrics(result)
        history["train_loss"].append(train_loss)
        history["val_mean_p_rank"].append(primary_metrics["mean_percentage_rank"])
        history["val_mrr"].append(primary_metrics["mrr"])
        history["val_ndcg"].append(primary_metrics["ndcg"])

        current_score = _compute_checkpoint_score(train_loss, primary_metrics, config)
        if current_score > best_score:
            best_score = current_score
            best_result = result["metrics"] if result is not None else {}
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            if result is not None:
                _save_eval_artifacts(run_dir, "best_val", result)

        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "metrics": result["metrics"] if result is not None else None}, ensure_ascii=False))
    save_run_artifacts(run_dir, config, history, best_result)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_primary_metrics(result):
    if result is None:
        return {"mrr": None, "ndcg": None, "mean_percentage_rank": None}
    primary = result["metrics"]["human_to_robot"]["task"]
    return {
        "mrr": primary["MRR"],
        "ndcg": primary["NDCG@10"],
        "mean_percentage_rank": primary["Mean Percentage Rank"],
    }


def _default_checkpoint_selection() -> dict:
    return {"loss_weight": 0.1, "mrr_weight": 1.0, "ndcg_weight": 1.0}


def _compute_checkpoint_score(train_loss: float, primary_metrics: dict, config: dict) -> float:
    selection = config.get("checkpoint_selection") or {}
    loss_weight = float(selection.get("loss_weight", 0.1))
    mrr_weight = float(selection.get("mrr_weight", 1.0))
    ndcg_weight = float(selection.get("ndcg_weight", 1.0))
    mrr = primary_metrics.get("mrr")
    ndcg = primary_metrics.get("ndcg")

    if mrr is None and ndcg is None:
        return -loss_weight * float(train_loss)

    score = -loss_weight * float(train_loss)
    if mrr is not None:
        score += mrr_weight * float(mrr)
    if ndcg is not None:
        score += ndcg_weight * float(ndcg)
    return score


def _save_eval_artifacts(run_dir: Path, prefix: str, result: dict):
    np.save(run_dir / f"{prefix}_similarity_matrix.npy", result["similarity_matrix"])
    np.save(run_dir / f"{prefix}_human_embeddings.npy", result["human_embeddings"])
    np.save(run_dir / f"{prefix}_robot_embeddings.npy", result["robot_embeddings"])
    with (run_dir / f"{prefix}_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(build_trajectory_retrieval_cases(result), handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
