import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from _bootstrap import bootstrap

bootstrap()

from bise.common import ensure_directory, load_json_config, merge_overrides, save_run_artifacts
from bise.modalities.video import build_retrieval_cases, collate_video_pairs, evaluate_video_retrieval, train_video_epoch
from bise.modalities.video.factory import build_split_manifest, build_video_dataset, build_video_model, split_video_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the video retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--csv-path", help="Optional CSV manifest override.")
    parser.add_argument("--data-root", help="Optional RH20T root override.")
    parser.add_argument("--output-dir", help="Optional output dir override.")
    parser.add_argument("--resume", help="Optional checkpoint path for resuming.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    config = merge_overrides(config, {"output_dir": args.output_dir})
    if args.csv_path:
        config.setdefault("dataset", {})["csv_path"] = args.csv_path
    if args.data_root:
        config.setdefault("dataset", {})["root_dir"] = args.data_root

    # 拿到预训练模型和对应 processor。
    processor, model = build_video_model(config["model"])

    run_name = f'{config["experiment_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = ensure_directory(Path(config["output_dir"]) / run_name)
    
    train_dataset = build_video_dataset(config["dataset"], processor=processor, is_train=True)
    split_datasets = split_video_dataset(train_dataset, config.get("split"))
    split_manifest = build_split_manifest(split_datasets)
    split_manifest_path = run_dir / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)
    config.setdefault("split", {})["manifest_path"] = str(split_manifest_path)

    val_source = build_video_dataset(config["dataset"], processor=processor, is_train=False)
    val_dataset = _mirror_subset(val_source, split_datasets.get("val"))

    train_loader = DataLoader(
        split_datasets["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        collate_fn=collate_video_pairs,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["eval_batch_size"],
            shuffle=False,
            num_workers=config["training"].get("num_workers", 0),
            collate_fn=collate_video_pairs,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["optimization"]["learning_rate"],
        weight_decay=config["optimization"].get("weight_decay", 0.0),
    )
    total_steps = config["training"]["num_epochs"] * max(len(train_loader), 1)
    warmup_steps = int(total_steps * float(config["optimization"].get("warmup_ratio", 0.0)))
    scheduler = get_scheduler(
        config["optimization"].get("scheduler", "linear"),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )
    scaler = GradScaler(device.type, enabled=config["training"].get("amp", False) and device.type == "cuda")
    config.setdefault("checkpoint_selection", _default_checkpoint_selection())

    history = {"train_loss": [], "train_loss_inter": [], "train_loss_intra": [], "val_mean_p_rank": [], "val_mrr": [], "val_ndcg": []}
    best_result = None
    best_score = float("-inf")
    record_intra_curves = _uses_intra_loss(config)

    for epoch in range(config["training"]["num_epochs"]):
        train_result = train_video_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            loss_config=config["loss"],
            amp=config["training"].get("amp", False),
            scaler=scaler,
            grad_clip_norm=config["optimization"].get("grad_clip_norm"),
            gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
            intra_transform_config=config["dataset"].get("intra_train_augmentations"),
        )
        history["train_loss"].append(train_result["loss"])
        history["train_loss_inter"].append(train_result["inter_loss"] if record_intra_curves else None)
        history["train_loss_intra"].append(train_result["intra_loss"] if record_intra_curves else None)

        val_result = evaluate_video_retrieval(model, val_loader, device) if val_loader is not None else None
        primary_metrics = _extract_primary_metrics(val_result)
        history["val_mean_p_rank"].append(primary_metrics["mean_percentage_rank"])
        history["val_mrr"].append(primary_metrics["mrr"])
        history["val_ndcg"].append(primary_metrics["ndcg"])

        current_score = _compute_checkpoint_score(train_result["loss"], primary_metrics, config)
        if current_score > best_score:
            best_score = current_score
            best_result = {"train": train_result, "validation": val_result["metrics"] if val_result is not None else {}}
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            if val_result is not None:
                _save_eval_artifacts(run_dir, "best_val", val_result)

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train": train_result,
                    "validation": val_result["metrics"] if val_result is not None else None,
                },
                ensure_ascii=False,
            )
        )
    save_run_artifacts(run_dir, config, history, best_result)


def _mirror_subset(source_dataset, subset):
    if subset is None:
        return None
    if hasattr(subset, "indices"):
        return torch.utils.data.Subset(source_dataset, list(subset.indices))
    return subset


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


def _uses_intra_loss(config: dict) -> bool:
    loss_weight = float(config.get("loss", {}).get("lambda_intra", 0.0))
    transform_config = config.get("dataset", {}).get("intra_train_augmentations") or config.get("loss", {}).get("intra_transform")
    return loss_weight > 0.0 and bool(transform_config)


def _save_eval_artifacts(run_dir: Path, prefix: str, result: dict):
    import numpy as np

    np.save(run_dir / f"{prefix}_similarity_matrix.npy", result["similarity_matrix"])
    np.save(run_dir / f"{prefix}_human_embeddings.npy", result["human_embeddings"])
    np.save(run_dir / f"{prefix}_robot_embeddings.npy", result["robot_embeddings"])
    with (run_dir / f"{prefix}_cases.json").open("w", encoding="utf-8") as handle:
        json.dump(build_retrieval_cases(result), handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
