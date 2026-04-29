import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    # 把 backbone 的高维特征映射到检索空间，并做归一化。
    def __init__(self, hidden_size: int, feature_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features), p=2, dim=-1)


class ResidualDomainAdapter(nn.Module):
    # 轻量域适配层：只做小幅残差校正，而不是完全重写 backbone 表示。
    def __init__(self, hidden_size: int, bottleneck_dim: int | None = None):
        super().__init__()
        inner_dim = bottleneck_dim or max(hidden_size // 4, 64)
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.layers(features)


class CrossDomainVideoEncoder(nn.Module):
    def __init__(
        self,
        backbone_adapter: nn.Module,
        feature_dim: int = 128,
        encoder_mode: str = "shared",
        adapter_bottleneck_dim: int | None = None,
        dropout: float = 0.1,
        temperature: float = 0.07,
        intra_temperature: float | None = None,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = 0,
        freeze_patch_embed: bool = False,
        freeze_norm_layers: bool = False,
    ):
        super().__init__()
        # encoder_mode 决定 human / robot 两个域共享多少参数。
        self.encoder_mode = str(encoder_mode).strip().lower()
        if self.encoder_mode not in {"shared", "dual_head", "dual_encoder"}:
            raise ValueError(f"Unsupported encoder_mode: {encoder_mode}")

        if self.encoder_mode == "dual_encoder":
            # dual_encoder: 两个域各自一套 backbone，参数完全分开。
            self.shared_backbone = None
            self.human_backbone = backbone_adapter
            self.robot_backbone = copy.deepcopy(backbone_adapter)
        else:
            # shared / dual_head: 先共享 backbone，再决定是否额外加域适配头。
            self.shared_backbone = backbone_adapter
            self.human_backbone = self.shared_backbone
            self.robot_backbone = self.shared_backbone

        hidden_size = getattr(backbone_adapter, "output_dim")
        if self.encoder_mode == "shared":
            self.human_adapter = nn.Identity()
            self.robot_adapter = nn.Identity()
        else:
            # dual_head / dual_encoder 都允许 human 和 robot 经过不同的 adapter。
            self.human_adapter = ResidualDomainAdapter(hidden_size, adapter_bottleneck_dim)
            self.robot_adapter = ResidualDomainAdapter(hidden_size, adapter_bottleneck_dim)

        self.projector = ProjectionHead(hidden_size, feature_dim=feature_dim, dropout=dropout)
        # 这里不直接存 temperature，而是存 logit_scale 的对数形式，训练时更稳定。
        self.logit_scale_inter = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        intra_temp = intra_temperature or temperature
        self.logit_scale_intra = nn.Parameter(torch.tensor(math.log(1.0 / intra_temp)))

        self._apply_freeze_policy(
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            freeze_patch_embed=freeze_patch_embed,
            freeze_norm_layers=freeze_norm_layers,
        )

    def encode_human(self, pixel_values: torch.Tensor, return_features: bool = False):
        # human 支路：backbone 提特征 -> adapter 做域校正 -> projector 投影到检索空间。
        features = self.human_adapter(self.human_backbone.encode_features(pixel_values))
        embeddings = self.projector(features)
        return (embeddings, features) if return_features else embeddings

    def encode_robot(self, pixel_values: torch.Tensor, return_features: bool = False):
        backbone = self.robot_backbone
        features = self.robot_adapter(backbone.encode_features(pixel_values))
        embeddings = self.projector(features)
        return (embeddings, features) if return_features else embeddings

    def forward(self, human_pixel_values: torch.Tensor, robot_pixel_values: torch.Tensor):
        # 前向同时输出 embedding 和中间 feature，后者主要给分析/可视化用。
        human_embeddings, human_features = self.encode_human(human_pixel_values, return_features=True)
        robot_embeddings, robot_features = self.encode_robot(robot_pixel_values, return_features=True)
        return {
            "human_embeddings": human_embeddings,
            "robot_embeddings": robot_embeddings,
            "human_features": human_features,
            "robot_features": robot_features,
            "logit_scale_inter": self.logit_scale_inter.exp(),
            "logit_scale_intra": self.logit_scale_intra.exp(),
        }

    def _apply_freeze_policy(
        self,
        freeze_backbone: bool,
        unfreeze_last_n_blocks: int,
        freeze_patch_embed: bool,
        freeze_norm_layers: bool,
    ):
        # 统一处理“冻结 backbone / 只解冻最后几层 / 冻结 patch embed / 冻结 norm”等策略。
        backbones = self._backbone_modules()
        if freeze_backbone or unfreeze_last_n_blocks > 0:
            for backbone in backbones:
                for parameter in backbone.parameters():
                    parameter.requires_grad = False

        if unfreeze_last_n_blocks > 0:
            # 常见微调策略：只解冻最后若干个 Transformer block。
            for backbone in backbones:
                blocks = backbone.get_transformer_blocks()
                for block in blocks[-unfreeze_last_n_blocks:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True

        if freeze_patch_embed:
            for backbone in backbones:
                for module in backbone.patch_embedding_modules():
                    for parameter in module.parameters():
                        parameter.requires_grad = False

        if freeze_norm_layers:
            for backbone in backbones:
                for module in backbone.norm_modules():
                    for parameter in module.parameters():
                        parameter.requires_grad = False

    def _backbone_modules(self):
        # 返回当前真正参与训练的 backbone 模块列表，供冻结策略统一处理。
        modules = []
        if self.shared_backbone is not None:
            modules.append(self.shared_backbone)
        if self.encoder_mode == "dual_encoder":
            modules.extend([self.human_backbone, self.robot_backbone])
        return modules
