import argparse
import datetime
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config, merge_overrides, save_run_artifacts
from bise.common.paths import resolve_path
from bise.data import collate_trajectories
from bise.modalities.trajectory import (
    build_trajectory_retrieval_cases,
    evaluate_trajectory_retrieval,
    pretrain_intra_modal_epoch,
    train_augmented_trajectory_epoch,
    train_trajectory_epoch,
)
from bise.modalities.trajectory.factory import (build_split_manifest, split_trajectory_dataset,
                                              build_trajectory_dataset, build_trajectory_model)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the trajectory retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--data-root", help="Optional dataset root override.")
    parser.add_argument("--output-dir", help="Optional output dir override.")
    parser.add_argument("--epochs", type=int, help="Override epochs for a closed-loop short run.")
    parser.add_argument("--run-dir", help="Exact new run directory (must not exist).")
    parser.add_argument("--split-manifest", help="Existing split_manifest.json to reuse.")
    parser.add_argument("--seed", type=int, help="Override training and dynamic split seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    config = merge_overrides(config, {"data_root": args.data_root, "output_dir": args.output_dir})
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config["num_epochs"] = args.epochs
        if config.get("mode") == "two_stage":
            if args.epochs < 2:
                raise ValueError("A two-stage run needs at least two total epochs")
            total = config["pretrain_epochs"] + config["finetune_epochs"]
            pretrain = max(1, min(args.epochs - 1,
                                 round(args.epochs * config["pretrain_epochs"] / total)))
            config["pretrain_epochs"] = pretrain
            config["finetune_epochs"] = args.epochs - pretrain
    from bise.common.device import experiment_device
    device = experiment_device(config)
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    _set_seed(config.get("seed", 42))

    run_name = f'{config["experiment_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = Path(args.run_dir) if args.run_dir else Path(config["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    if args.split_manifest:
        manifest = resolve_path(args.split_manifest)
        (run_dir / "split_manifest.json").write_bytes(manifest.read_bytes())
        config.setdefault("split", {})["manifest_path"] = str(run_dir / "split_manifest.json")

    if config["model_params"].get("grasp_geometry"):
        from bise.data.rh20t.grasp_geometry import bind_geometry
        bind_geometry(config)
    dataset = build_trajectory_dataset(config)
    (run_dir / "data_manifest.json").write_text(json.dumps(dataset.manifest, indent=2))
    split_config = dict(config.get("split") or {"unit": "scene", "seed": config.get("seed", 42), "ratios": {"train": 0.8, "val": 0.1, "test": 0.1}})
    if (split_config.get("fixed_test_manifest_path") or split_config.get("shared_manifest_path")) and "seed" not in split_config:
        split_config["seed"] = int(config.get("seed", 42))
    if split_config.get("scene_counts") and not split_config.get("manifest_path"):
        from bise.data.rh20t.task_split import split_task_dataset
        split_datasets = split_task_dataset(dataset, split_config, "trajectory")
    else:
        split_datasets = split_trajectory_dataset(dataset, split_config)
    split_manifest = build_split_manifest(split_datasets)
    split_manifest_path = run_dir / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)
    if config.get("protocol") == "retrain_v2":
        from bise.common.fingerprints import write_json
        from bise.data.rh20t.splits import validate_split
        write_json(run_dir / "split_info.json", {
            "protocol": config["protocol"], "unit": split_config.get("unit", "scene"),
            "counts": validate_split(split_manifest, split_config.get("unit", "scene")),
            "use_fixed_val": (split_config.get("unit") != "task"
                              and split_config.get("use_fixed_val", True)),
            **({"scene_counts": split_config["scene_counts"]}
               if split_config.get("scene_counts") else {}),
        })
    print(json.dumps({
        "split_counts": {name: len(scene_ids) for name, scene_ids in split_manifest.items()},
        "split_seed": split_config.get("seed"),
        "fixed_test": bool(split_config.get("fixed_test_manifest_path")
                           or split_config.get("shared_manifest_path")),
    }, ensure_ascii=False), flush=True)
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

    model = build_trajectory_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    mode = config.get("mode", "standard")
    total_epochs = (config["pretrain_epochs"] + config["finetune_epochs"]
                    if mode == "two_stage" else config["num_epochs"])
    scheduler = _build_lr_scheduler(optimizer, config, total_epochs)
    config.setdefault("checkpoint_selection", _default_checkpoint_selection())
    (run_dir / "params.json").write_text(json.dumps(config, indent=2))
    if config.get("protocol") == "retrain_v2":
        from bise.common.evaluation import bind_protocols
        bind_protocols(run_dir, config)
        from bise.common.fingerprints import write_json
        write_json(run_dir / "status.json", {"state": "training"})
    print(f"RUN_DIR={run_dir.resolve()}", flush=True)

    history = {"train_loss": [], "train_loss_inter": [], "train_loss_intra": [], "val_mean_p_rank": [], "val_mrr": [], "val_ndcg": []}
    history["learning_rate"] = []
    best_result = None
    best_score = float("-inf")

    if mode == "two_stage":
        for epoch in range(config["pretrain_epochs"]):
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
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
            print(f"Pretrain Epoch {epoch + 1}/{config['pretrain_epochs']}: "
                  f"intra_loss={intra_loss:.4f}, lr={history['learning_rate'][-1]:.6g}")
            if scheduler is not None:
                scheduler.step()

    finetune_epochs = config.get("finetune_epochs", config.get("num_epochs", 0))
    loop_epochs = finetune_epochs if mode == "two_stage" else config["num_epochs"]

    for epoch in range(loop_epochs):
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
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
            (run_dir / "checkpoint_info.json").write_text(json.dumps({
                "epoch": epoch + 1 + (config["pretrain_epochs"] if mode == "two_stage" else 0),
                "score": current_score,
                "selection": config["checkpoint_selection"],
                "validation_only": config["checkpoint_selection"].get("loss_weight", .1) == 0,
            }, indent=2))
            if result is not None:
                _save_eval_artifacts(run_dir, "best_val", result)

        print(json.dumps({
            "epoch": len(history["train_loss"]), "train_loss": train_loss,
            "learning_rate": history["learning_rate"][-1],
            "metrics": result["metrics"] if result is not None else None,
        }, ensure_ascii=False))
        if scheduler is not None:
            scheduler.step()
    torch.save(model.state_dict(), run_dir / "last_model.pth")
    save_run_artifacts(run_dir, config, history, best_result)
    if config.get("protocol") == "retrain_v2":
        write_json(run_dir / "status.json", {"state": "complete", "epochs": len(history["train_loss"])})


def _build_lr_scheduler(optimizer, config: dict, total_epochs: int):
    schedule = config.get("lr_schedule")
    if not schedule:
        return None
    if schedule["name"] != "cosine":
        raise ValueError("Trajectory lr_schedule.name must be cosine")
    initial_lr = float(config["learning_rate"])
    min_lr = float(schedule["min_lr"])
    hold_epochs = int(schedule["hold_epochs"])
    if initial_lr <= 0 or not 0 <= min_lr <= initial_lr:
        raise ValueError("Learning rates must satisfy 0 <= min_lr <= learning_rate > 0")
    if total_epochs < 1 or hold_epochs < 1:
        raise ValueError("Total and hold epochs must be positive")
    decay_epochs = max(total_epochs - hold_epochs, 0)

    def multiplier(epoch: int) -> float:
        # Index zero is epoch one; the final training epoch uses min_lr.
        if epoch < hold_epochs or decay_epochs == 0:
            return 1.0
        progress = min((epoch - hold_epochs + 1) / decay_epochs, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr / initial_lr + (1 - min_lr / initial_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


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
