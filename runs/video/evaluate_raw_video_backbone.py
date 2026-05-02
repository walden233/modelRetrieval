import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import ensure_directory, load_json_config
from bise.modalities.video import build_retrieval_cases, collate_video_pairs, evaluate_video_retrieval
from bise.modalities.video.factory import build_video_dataset, split_video_dataset
from bise.modalities.video.models.backbone_registry import build_video_backbone
from evaluate_video import _resolve_split_config, _save_evaluation_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an untrained raw video backbone without projection heads.")
    parser.add_argument("--config", required=True, help="Video config whose dataset and backbone will be used.")
    parser.add_argument("--output-dir", required=True, help="Run-like output directory. final_test will be written inside it.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--split-manifest", help="Optional split_manifest.json from a trained run.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cases to export.")
    parser.add_argument("--data-root", help="Optional RH20T dataset root override, for example dataset/RH20T_subset/RH20T_cfg3.")
    parser.add_argument("--all-as-test", action="store_true", help="Evaluate the entire dataset as the test split.")
    parser.add_argument("--final-subdir", default="final_test", help="Subdirectory inside output-dir for evaluation artifacts.")
    parser.add_argument("--skip-save-params", action="store_true", help="Do not overwrite params.json in output-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_json_config(args.config)
    if args.data_root:
        config.setdefault("dataset", {})["root_dir"] = args.data_root
    run_dir = ensure_directory(args.output_dir)
    final_test_dir = ensure_directory(Path(run_dir) / args.final_subdir)

    processor, backbone_adapter = build_video_backbone(
        backbone_type=config["model"]["backbone_type"],
        model_name=config["model"]["backbone_name"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    model = RawBackboneRetrievalModel(backbone_adapter)

    split_config, split_manifest_path = _resolve_split_config(
        config,
        str(Path(run_dir) / "raw_backbone.pth"),
        args.split_manifest,
        all_as_test=args.all_as_test,
    )
    dataset_source = build_video_dataset(config["dataset"], processor=processor, is_train=False)
    split_datasets = split_video_dataset(dataset_source, split_config)
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
    result = evaluate_video_retrieval(model, dataloader, device)
    cases = build_retrieval_cases(result, top_k=args.top_k)
    _save_evaluation_outputs(
        final_test_dir,
        result,
        cases,
        split_manifest_path,
        split_config,
        config["dataset"].get("root_dir"),
    )
    if not args.skip_save_params:
        _save_raw_params(run_dir, config, args, split_manifest_path)
    print(json.dumps({"metrics": result["metrics"], "cases": cases[:5]}, indent=2, ensure_ascii=False))


class RawBackboneRetrievalModel(torch.nn.Module):
    def __init__(self, backbone_adapter: torch.nn.Module):
        super().__init__()
        self.backbone_adapter = backbone_adapter

    def forward(self, human_pixel_values: torch.Tensor, robot_pixel_values: torch.Tensor):
        human_features = self.backbone_adapter.encode_features(human_pixel_values)
        robot_features = self.backbone_adapter.encode_features(robot_pixel_values)
        return {
            "human_embeddings": F.normalize(human_features, p=2, dim=-1),
            "robot_embeddings": F.normalize(robot_features, p=2, dim=-1),
        }


def _save_raw_params(run_dir: Path, config: dict, args, split_manifest_path: Path | None) -> None:
    params = dict(config)
    params["raw_backbone"] = True
    params["source_config"] = args.config
    params.setdefault("split", {})
    if args.all_as_test:
        params["split"] = {"all_as_test": True}
    if split_manifest_path is not None:
        params["split"]["manifest_path"] = str(split_manifest_path)
    with (run_dir / "params.json").open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
