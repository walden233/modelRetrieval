import torch
import torch.nn as nn


class VJEPAAdapter(nn.Module):
    # V-JEPA 的输出接口和 VideoMAE 不同，所以也单独包一层适配器。
    def __init__(self, backbone: nn.Module, hidden_size: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.output_dim = hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_features(pixel_values)

    def encode_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # V-JEPA 先输出 patch/token 级视觉特征，这里简单做均值池化得到视频级表示。
        vision_features = self.backbone.get_vision_features(pixel_values)
        return torch.mean(vision_features, dim=1)

    def get_transformer_blocks(self):
        # 统一暴露 Transformer blocks，便于做 partial finetune。
        if hasattr(self.backbone, "blocks"):
            return list(self.backbone.blocks)
        if hasattr(self.backbone, "vision_model") and hasattr(self.backbone.vision_model, "blocks"):
            return list(self.backbone.vision_model.blocks)
        return []

    def patch_embedding_modules(self):
        modules = []
        if hasattr(self.backbone, "patch_embed"):
            modules.append(self.backbone.patch_embed)
        if hasattr(self.backbone, "vision_model") and hasattr(self.backbone.vision_model, "patch_embed"):
            modules.append(self.backbone.vision_model.patch_embed)
        return modules

    def norm_modules(self):
        modules = []
        for name, module in self.backbone.named_modules():
            if "norm" in name.lower():
                modules.append(module)
        return modules
