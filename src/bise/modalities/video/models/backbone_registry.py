from pathlib import Path

from transformers import AutoModel, AutoVideoProcessor, VideoMAEImageProcessor

from bise.common.paths import resolve_path

from .videomae_adapter import VideoMAEAdapter
from .vjepa_adapter import VJEPAAdapter


def backbone_source(model_config: dict) -> str:
    directory = model_config.get("backbone_dir")
    return str(resolve_path(directory).resolve()) if directory else model_config["backbone_name"]


def build_video_backbone(backbone_type: str, model_name: str, trust_remote_code: bool = True):
    # 这个文件只解决一件事：根据配置字符串，创建对应的 processor 和 backbone adapter。
    normalized_type = str(backbone_type).strip().lower()
    options = {}
    if Path(model_name).is_absolute():
        if not Path(model_name).is_dir():
            raise FileNotFoundError(f"Local video backbone directory is missing: {model_name}")
        options["local_files_only"] = True
    if normalized_type == "videomae":
        processor = VideoMAEImageProcessor.from_pretrained(model_name, **options)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code,
                                             **options)
        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("Unable to determine VideoMAE hidden size from model config.")
        return processor, VideoMAEAdapter(backbone, hidden_size=hidden_size)

    if normalized_type == "vjepa":
        processor = AutoVideoProcessor.from_pretrained(model_name, **options)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code,
                                             **options)
        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("Unable to determine V-JEPA hidden size from model config.")
        return processor, VJEPAAdapter(backbone, hidden_size=hidden_size)

    raise ValueError(f"Unsupported backbone_type: {backbone_type}")
