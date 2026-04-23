import torch
import torch.nn as nn


class VideoMAEAdapter(nn.Module):
    # 这个类不负责最终投影，只负责把 VideoMAE backbone 的输出整理成统一特征接口。
    def __init__(self, backbone: nn.Module, hidden_size: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.output_dim = hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_features(pixel_values)

    def encode_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # VideoMAE 期望输入布局是 [B, C, T, H, W]，因此这里先做维度变换。
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        outputs = self.backbone(pixel_values)
        if hasattr(outputs, "last_hidden_state"):
            # 取 CLS token 作为整段视频的全局表示。
            return outputs.last_hidden_state[:, 0]
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def get_transformer_blocks(self):
        # 为冻结 / 解冻最后几层提供统一访问入口。
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return list(self.backbone.encoder.layer)
        if hasattr(self.backbone, "videomae") and hasattr(self.backbone.videomae, "encoder"):
            encoder = self.backbone.videomae.encoder
            if hasattr(encoder, "layer"):
                return list(encoder.layer)
        return []

    def patch_embedding_modules(self):
        # patch embedding 往往参数量大、底层性强，单独暴露便于冻结。
        modules = []
        if hasattr(self.backbone, "embeddings"):
            modules.append(self.backbone.embeddings)
        if hasattr(self.backbone, "videomae") and hasattr(self.backbone.videomae, "embeddings"):
            modules.append(self.backbone.videomae.embeddings)
        return modules

    def norm_modules(self):
        # 归一化层有时在小数据微调中需要冻结，这里统一收集。
        modules = []
        for name, module in self.backbone.named_modules():
            if "norm" in name.lower():
                modules.append(module)
        return modules
