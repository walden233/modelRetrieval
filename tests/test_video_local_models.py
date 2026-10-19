"""Load complete local video backbones and processors without a Hub connection."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bise.modalities.video.batch import extract_pixel_values
from bise.modalities.video.factory import build_video_model
from bise.modalities.video.models.backbone_registry import backbone_source, build_video_backbone


@pytest.mark.parametrize("kind", ("vjepa", "videomae"))
def test_local_bundle_loads_identical_weights_and_runs_without_hub(tmp_path, monkeypatch, kind):
    from huggingface_hub import file_download
    from transformers import (
        VideoMAEConfig, VideoMAEImageProcessor, VideoMAEModel,
        VJEPA2Config, VJEPA2Model, VJEPA2VideoProcessor,
    )

    if kind == "vjepa":
        original = VJEPA2Model(VJEPA2Config(
            crop_size=16, patch_size=8, frames_per_clip=4, tubelet_size=2,
            hidden_size=64, num_attention_heads=4, num_hidden_layers=2,
            pred_hidden_size=64, pred_num_attention_heads=4, pred_num_hidden_layers=1,
        ))
        processor = VJEPA2VideoProcessor(do_resize=False, do_center_crop=False)
    else:
        original = VideoMAEModel(VideoMAEConfig(
            image_size=16, patch_size=8, num_frames=4, tubelet_size=2,
            hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64,
        ))
        processor = VideoMAEImageProcessor(do_resize=False, do_center_crop=False)
    directory = tmp_path / kind
    original.save_pretrained(directory)
    processor.save_pretrained(directory)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("Local model loading must not request Hub files")

    monkeypatch.setattr(file_download, "_request_wrapper", network_forbidden)
    config = {"backbone_type": kind, "backbone_name": "unavailable/remote-model",
              "backbone_dir": str(directory), "feature_dim": 8, "encoder_mode": "dual_head",
              "dropout": 0.0, "freeze_backbone": False, "unfreeze_last_n_blocks": 1}
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        loaded_processor, model = build_video_model(config)
        backbone = model.shared_backbone.backbone
        for name, value in original.state_dict().items():
            torch.testing.assert_close(backbone.state_dict()[name], value, rtol=0, atol=0)
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
        videos = extract_pixel_values(loaded_processor([frames, frames], return_tensors="pt"))
        result = model(videos, videos)
        assert result["human_embeddings"].shape == result["robot_embeddings"].shape == (2, 8)
        assert torch.isfinite(result["human_embeddings"]).all()
    finally:
        torch.set_num_threads(threads)


def test_missing_local_directory_fails_without_using_the_remote_name(tmp_path):
    config = {"backbone_name": "facebook/vjepa2-vitl-fpc64-256",
              "backbone_dir": str(tmp_path / "missing")}
    with pytest.raises(FileNotFoundError, match="Local video backbone directory"):
        build_video_backbone("vjepa", backbone_source(config))
    assert backbone_source({"backbone_name": config["backbone_name"]}) == config["backbone_name"]


def test_video32_configs_select_portable_model_directories():
    directory = Path(__file__).resolve().parents[1] / "configs/retrain/video32"
    for path in directory.glob("*.json"):
        config = json.loads(path.read_text())
        if "model" not in config:
            continue
        model = config["model"]
        assert model["backbone_dir"] == (
            "artifacts/models/video/" + model["backbone_name"].replace("/", "--"))
