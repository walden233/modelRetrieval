from transformers import AutoModel, AutoVideoProcessor, VideoMAEImageProcessor

from .videomae_adapter import VideoMAEAdapter
from .vjepa_adapter import VJEPAAdapter


def build_video_backbone(backbone_type: str, model_name: str, trust_remote_code: bool = True):
    # 这个文件只解决一件事：根据配置字符串，创建对应的 processor 和 backbone adapter。
    normalized_type = str(backbone_type).strip().lower()
    if normalized_type == "videomae":
        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("Unable to determine VideoMAE hidden size from model config.")
        return processor, VideoMAEAdapter(backbone, hidden_size=hidden_size)

    if normalized_type == "vjepa":
        processor = AutoVideoProcessor.from_pretrained(model_name)
        backbone = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim", None)
        if hidden_size is None:
            raise ValueError("Unable to determine V-JEPA hidden size from model config.")
        return processor, VJEPAAdapter(backbone, hidden_size=hidden_size)

    raise ValueError(f"Unsupported backbone_type: {backbone_type}")
