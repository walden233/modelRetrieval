import argparse
import datetime
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoVideoProcessor, VideoMAEImageProcessor, get_scheduler

from _bootstrap import bootstrap

bootstrap()

from bise.common import ensure_directory, load_json_config, merge_overrides, save_run_artifacts
from bise.modalities.video import InfoNCELoss, VJEPAAdapter, VideoMAEAdapter, train_video_epoch
from bise.data.whirl.video_pair_dataset import VideoPairDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the video retrieval model.")
    parser.add_argument("--config", required=True, help="Path to the JSON config file.")
    parser.add_argument("--csv-path", help="Optional CSV manifest override.")
    parser.add_argument("--output-dir", help="Optional output dir override.")
    return parser.parse_args()


def build_video_model(config):
    model_name = config["model_name"]
    feature_dim = config["feature_dim"]
    if config["model_type"] == "videomae":
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = getattr(backbone.config, "hidden_size", None) or backbone.config.hidden_size
        model = VideoMAEAdapter(backbone, hidden_size=hidden_size, feature_dim=feature_dim)
    elif config["model_type"] == "vjepa":
        processor = AutoVideoProcessor.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = backbone.config.hidden_size
        model = VJEPAAdapter(backbone, hidden_size=hidden_size, feature_dim=feature_dim)
    else:
        raise ValueError(f'Unsupported model_type: {config["model_type"]}')
    return processor, model


def main():
    args = parse_args()
    config = load_json_config(args.config)
    config = merge_overrides(config, {"csv_path": args.csv_path, "output_dir": args.output_dir})
    processor, model = build_video_model(config)

    dataset = VideoPairDataset(config["csv_path"], processor=processor, num_frames=config.get("num_frames", 16))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=config["num_epochs"] * len(dataloader),
    )
    loss_fn = InfoNCELoss(temperature=config.get("temperature", 0.07))

    run_name = f'{config["experiment_name"]}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    run_dir = ensure_directory(Path(config["output_dir"]) / run_name)
    history = {"train_loss": [], "val_mean_p_rank": []}
    best_result = None
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        train_loss = train_video_epoch(model, dataloader, optimizer, scheduler, loss_fn, device)
        history["train_loss"].append(train_loss)
        if train_loss < best_loss:
            best_loss = train_loss
            best_result = {"train_loss": train_loss}
            torch.save(model.state_dict(), run_dir / "best_model.pth")
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss}, ensure_ascii=False))

    torch.save(model.state_dict(), run_dir / "last_model.pth")
    save_run_artifacts(run_dir, config, history, best_result)


if __name__ == "__main__":
    main()
