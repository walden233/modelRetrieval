"""Re-encode changed human coordinates against one unchanged robot candidate library."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import bootstrap
bootstrap()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bise.common import load_json_config
from bise.data.rh20t.collate import collate_trajectories, trajectory_context_kwargs
from bise.modalities.trajectory.evaluator import evaluate_trajectory_retrieval
from bise.modalities.trajectory.factory import build_trajectory_dataset, build_trajectory_model, split_trajectory_dataset
from bise.modalities.trajectory.human_robustness import fixed_gallery_comparison
from bise.modalities.trajectory.provenance import save_evaluation_provenance, validate_checkpoint_config


def _rigid_transform(yaw_degrees: float = 0.0, translation_m: float = 0.0) -> np.ndarray:
    angle = math.radians(yaw_degrees)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[math.cos(angle), -math.sin(angle), 0],
                         [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    transform[0, 3] = translation_m
    return transform


def _human_embeddings(model, dataloader, device, expected_ids):
    model.eval()
    embeddings, identities = [], []
    with torch.no_grad():
        for batch in dataloader:
            context = trajectory_context_kwargs(batch, device)["human_context"]
            value = model.forward_human(batch["human_poses"].to(device),
                                        batch["human_mask"].to(device), context)
            embeddings.append(value.cpu().numpy())
            identities.extend(zip(batch["human_scene_ids"], batch["human_camera_ids"]))
    if identities != expected_ids:
        raise ValueError("Transformed human query order differs from the baseline")
    return np.concatenate(embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Saved run params.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Optional cfg3 root; requires --all-as-test")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split-manifest", help="Defaults to the checkpoint run split")
    parser.add_argument("--all-as-test", action="store_true")
    parser.add_argument("--translation-m", type=float, default=0.1)
    parser.add_argument("--yaw-degrees", type=float, default=90.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.translation_m) or not np.isfinite(args.yaw_degrees):
        raise ValueError("Transform parameters must be finite")
    if args.data_root and not args.all_as_test:
        raise ValueError("A different data root requires --all-as-test or an explicit matching split")
    if args.all_as_test and args.split_manifest:
        raise ValueError("--all-as-test cannot be combined with --split-manifest")
    config = load_json_config(args.config)
    if args.data_root:
        config["data_root"] = args.data_root
    if (config.get("data_params") or {}).get("human_source") != "hand_3d_base":
        raise ValueError("Human robustness evaluation requires hand_3d_base")
    validate_checkpoint_config(config, args.checkpoint)
    if args.all_as_test:
        split_config = {"all_as_test": True}
    else:
        manifest_path = (Path(args.split_manifest) if args.split_manifest else
                         Path(args.checkpoint).resolve().parent / "split_manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        split_config = {"manifest_path": str(manifest_path)}
    torch.set_num_threads(int(config.get("torch_num_threads", 4)))
    source = build_trajectory_dataset(config, args.checkpoint)
    subset = split_trajectory_dataset(source, split_config)[args.split]
    if subset is None or not len(subset):
        raise ValueError(f"No samples in split {args.split}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_trajectory_model(config, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    loader = DataLoader(subset, batch_size=config["batch_size"], collate_fn=collate_trajectories)
    baseline = evaluate_trajectory_retrieval(model, loader, device)
    expected_ids = list(zip(baseline["metadata"]["human_scene_ids"],
                            baseline["metadata"]["human_camera_ids"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "fixed_robot_embeddings.npy", baseline["robot_embeddings"])
    np.save(output / "baseline_human_embeddings.npy", baseline["human_embeddings"])
    transforms = {"translation_x": _rigid_transform(translation_m=args.translation_m),
                  "yaw": _rigid_transform(yaw_degrees=args.yaw_degrees),
                  "yaw_and_translation": _rigid_transform(args.yaw_degrees, args.translation_m)}
    results = {}
    for name, transform in transforms.items():
        variant_source = source.with_human_transform(transform)
        variant_subset = Subset(variant_source, subset.indices) if isinstance(subset, Subset) else variant_source
        variant_loader = DataLoader(variant_subset, batch_size=config["batch_size"], collate_fn=collate_trajectories)
        embeddings = _human_embeddings(model, variant_loader, device, expected_ids)
        np.save(output / f"{name}_human_embeddings.npy", embeddings)
        results[name] = {"T_new_from_original": transform.tolist(),
                         **fixed_gallery_comparison(baseline, embeddings)}
    summary = {"human_representation": source.human_representation,
               "split": args.split, "data_root": config["data_root"],
               "fixed_gallery": "baseline robot embeddings reused for every human variant",
               "baseline_metrics": baseline["metrics"]["human_to_robot"],
               "variants": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "query_ids.json").write_text(json.dumps(expected_ids, indent=2), encoding="utf-8")
    save_evaluation_provenance(output, config, args.checkpoint, source.manifest)
    print(json.dumps({"output_dir": str(output.resolve()),
                      "query_count": len(expected_ids),
                      "gallery_count": len(baseline["robot_embeddings"]),
                      "variants": {name: {key: value for key, value in result.items()
                                           if key in ("max_abs_embedding_delta", "top1_same_candidate_rate")}
                                   for name, result in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
