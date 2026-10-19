import argparse
import datetime
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from _bootstrap import bootstrap

bootstrap()

from bise.common import ensure_directory, load_json_config, merge_overrides, save_run_artifacts
from bise.common.paths import resolve_path
from bise.modalities.video import build_retrieval_cases, collate_video_pairs, evaluate_video_retrieval, train_video_epoch
from bise.modalities.video.factory import build_split_manifest, build_video_dataset, build_video_model, split_video_dataset
from bise.modalities.video.sampling import build_training_sampler
from bise.modalities.video.semantic import prepare_semantic_supervision


def parse_args():
    parser = argparse.ArgumentParser(description="Train the video retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--csv-path", help="Optional CSV manifest override.")
    parser.add_argument("--data-root", help="Optional RH20T root override.")
    parser.add_argument("--output-dir", help="Optional output dir override.")
    parser.add_argument("--resume", help="Optional checkpoint path for resuming.")
    parser.add_argument("--seed", type=int, help="Override training and dynamic split seed.")
    parser.add_argument("--run-dir", help="Exact new run directory (must not exist).")
    parser.add_argument("--prepared", action="store_true",
                        help="Train an existing run whose metadata is prepared by the pipeline.")
    parser.add_argument("--split-manifest", help="Existing split_manifest.json to reuse.")
    parser.add_argument("--epochs", type=int, help="Override the training epoch budget.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    config = merge_overrides(config, {"output_dir": args.output_dir})
    if args.seed is not None:
        from bise.common.config import override_seed
        override_seed(config, args.seed)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config["training"]["num_epochs"] = args.epochs
    if args.csv_path:
        config.setdefault("dataset", {})["csv_path"] = args.csv_path
    if args.data_root:
        config.setdefault("dataset", {})["root_dir"] = args.data_root
    from bise.common.device import experiment_device
    device = experiment_device(config)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_name = f'{config["experiment_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = Path(args.run_dir) if args.run_dir else Path(config["output_dir"]) / run_name
    if args.prepared:
        from bise.common.training import validate_prepared_run
        validate_prepared_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=args.prepared)
    if args.split_manifest:
        manifest = resolve_path(args.split_manifest)
        (run_dir / "split_manifest.json").write_bytes(manifest.read_bytes())
        config.setdefault("split", {})["manifest_path"] = str(run_dir / "split_manifest.json")
    processor, model = build_video_model(config["model"])
    (run_dir / "trainable.json").write_text(json.dumps(_trainable_info(model, config), indent=2))

    train_dataset = build_video_dataset(config["dataset"], processor=processor, is_train=True)
    split_config = dict(config.get("split") or {})
    if (split_config.get("fixed_test_manifest_path") or split_config.get("shared_manifest_path")) and "seed" not in split_config:
        split_config["seed"] = seed
    if split_config.get("scene_counts") and not split_config.get("manifest_path"):
        from bise.data.rh20t.task_split import split_task_dataset
        split_datasets = split_task_dataset(train_dataset, split_config, "video",
                                           config.get("eval_manifests", {}).get("cfg2"))
    else:
        split_datasets = split_video_dataset(train_dataset, split_config)
    split_manifest = build_split_manifest(split_datasets)
    split_manifest_path = run_dir / "split_manifest.json"
    with split_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(split_manifest, handle, indent=2, ensure_ascii=False)
    if config.get("protocol") == "retrain_v2":
        from bise.common.fingerprints import write_json
        from bise.data.rh20t.splits import validate_split
        write_json(run_dir / "data_manifest.json", train_dataset.manifest)
        write_json(run_dir / "split_info.json", {
            "protocol": config["protocol"], "unit": split_config.get("unit", "scene"),
            "counts": validate_split(
                split_manifest, split_config.get("unit", "scene"),
                require_disjoint=split_config.get("require_disjoint", False)),
            "use_fixed_val": (split_config.get("unit") != "task"
                              and split_config.get("use_fixed_val", True)),
            **({"scene_counts": split_config["scene_counts"]}
               if split_config.get("scene_counts") else {}),
        })
    print(json.dumps({
        "split_sample_counts": {name: len(sample_ids) for name, sample_ids in split_manifest.items()},
        "split_scene_counts": {
            name: _count_subset_scenes(train_dataset, subset)
            for name, subset in split_datasets.items()
        },
        "split_seed": split_config.get("seed"),
        "fixed_test": bool(split_config.get("fixed_test_manifest_path")
                           or split_config.get("shared_manifest_path")),
    }, ensure_ascii=False), flush=True)
    config["split"] = split_config
    config["split"]["manifest_path"] = str(split_manifest_path)

    val_source = build_video_dataset(config["dataset"], processor=processor, is_train=False)
    val_dataset = _mirror_subset(val_source, split_datasets.get("val"))

    sampler = build_training_sampler(split_datasets["train"], config["training"], seed)
    loader_options = {
        "num_workers": config["training"].get("num_workers", 0),
        "collate_fn": collate_video_pairs,
    }
    if sampler is not None:
        loader_options["batch_sampler"] = sampler
    else:
        loader_options.update(batch_size=config["training"]["batch_size"], shuffle=True)
    train_loader = DataLoader(split_datasets["train"], **loader_options)
    semantic = prepare_semantic_supervision(config, split_datasets["train"], run_dir)
    if semantic is not None:
        semantic.to(device)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["eval_batch_size"],
            shuffle=False,
            num_workers=config["training"].get("num_workers", 0),
            collate_fn=collate_video_pairs,
        )

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
    if config.get("protocol") == "retrain_v2":
        write_json(run_dir / "params.json", config)
        from bise.common.evaluation import bind_protocols
        bind_protocols(run_dir, config)
        write_json(run_dir / "status.json", {"state": "training"})
    print(f"RUN_DIR={run_dir.resolve()}", flush=True)

    history = {"train_loss": [], "train_loss_inter": [], "train_loss_intra": [], "val_mean_p_rank": [], "val_mrr": [], "val_ndcg": []}
    if config["loss"].get("negative_separation", {}).get("weight", 0):
        history.update(train_loss_negative=[], negative_weighted_loss=[])
    if semantic is not None:
        history.update(train_loss_semantic=[], semantic_coverage=[], semantic_entropy=[],
                       semantic_weight=[])
        if semantic.mode == "queue":
            history.update(semantic_candidates=[], semantic_weighted_loss=[],
                           semantic_gradient_ratio=[])
    best_result = None
    best_score = float("-inf")
    record_intra_curves = _uses_intra_loss(config)

    for epoch in range(config["training"]["num_epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
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
            semantic_supervisor=semantic,
            epoch=epoch,
            dataset_cfg=Path(config["dataset"]["root_dir"]).name.removeprefix("RH20T_"),
        )
        history["train_loss"].append(train_result["loss"])
        history["train_loss_inter"].append(train_result["inter_loss"] if record_intra_curves else None)
        history["train_loss_intra"].append(train_result["intra_loss"] if record_intra_curves else None)
        if "train_loss_negative" in history:
            history["train_loss_negative"].append(train_result["negative_loss"])
            history["negative_weighted_loss"].append(train_result["negative_weighted_loss"])
        if semantic is not None:
            history["train_loss_semantic"].append(train_result["semantic_loss"])
            history["semantic_coverage"].append(train_result["semantic_coverage"])
            history["semantic_entropy"].append(train_result["semantic_entropy"])
            history["semantic_weight"].append(train_result["semantic_weight"])
            if semantic.mode == "queue":
                for key in ("semantic_candidates", "semantic_weighted_loss", "semantic_gradient_ratio"):
                    history[key].append(train_result[key])

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
            if config.get("protocol") == "retrain_v2":
                write_json(run_dir / "checkpoint_info.json", {
                    "epoch": epoch + 1, "score": current_score,
                    "selection": config["checkpoint_selection"], "validation_only": True,
                })
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
    if config.get("protocol") == "retrain_v2":
        write_json(run_dir / "status.json", {"state": "complete", "epochs": len(history["train_loss"])})


def _trainable_info(model, config: dict) -> dict:
    def counts(module):
        return {
            "total": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(parameter.numel() for parameter in module.parameters()
                             if parameter.requires_grad),
        }

    backbones = ([("shared", model.shared_backbone)] if model.shared_backbone is not None
                 else [("human", model.human_backbone), ("robot", model.robot_backbone)])
    heads = [("human", model.human_adapter), ("robot", model.robot_adapter),
             ("projector", model.projector)]
    if getattr(model, "semantic_projector", None) is not None:
        heads.append(("semantic", model.semantic_projector))
    return {
        "encoder_mode": model.encoder_mode,
        "batch_size": config["training"]["batch_size"],
        "unfreeze_last_n_blocks": config["model"].get("unfreeze_last_n_blocks", 0),
        "parameters": counts(model),
        "heads": {name: counts(module) for name, module in heads},
        "backbones": [{
            "name": name, "parameters": counts(backbone),
            "trainable_blocks": [
                index + 1 for index, block in enumerate(backbone.get_transformer_blocks())
                if any(parameter.requires_grad for parameter in block.parameters())
            ] if hasattr(backbone, "get_transformer_blocks") else [],
        } for name, backbone in backbones],
    }


def _count_subset_scenes(source_dataset, subset) -> int:
    if subset is None:
        return 0
    indices = subset.indices if hasattr(subset, "indices") else range(len(source_dataset))
    return len({source_dataset.samples[index].scene_id for index in indices})


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
