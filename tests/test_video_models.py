import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
from transformers import VJEPA2Config, VJEPA2Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video.models import CrossDomainVideoEncoder, VideoMAEAdapter, VJEPAAdapter


class _DummyVideoMAEBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_channels=3)
        self.received_shape = None

    def forward(self, pixel_values):
        self.received_shape = tuple(pixel_values.shape)
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(last_hidden_state=torch.ones(batch_size, 2, 4))


def test_videomae_adapter_keeps_btchw_layout():
    backbone = _DummyVideoMAEBackbone()
    adapter = VideoMAEAdapter(backbone, hidden_size=4)
    output = adapter(torch.randn(2, 8, 3, 16, 16))
    assert backbone.received_shape == (2, 8, 3, 16, 16)
    assert output.shape == (2, 4)


def test_videomae_adapter_converts_bcthw_layout():
    backbone = _DummyVideoMAEBackbone()
    adapter = VideoMAEAdapter(backbone, hidden_size=4)
    output = adapter(torch.randn(2, 3, 8, 16, 16))
    assert backbone.received_shape == (2, 8, 3, 16, 16)
    assert output.shape == (2, 4)


def tiny_vjepa():
    config = VJEPA2Config(
        hidden_size=48, num_attention_heads=4, num_hidden_layers=24, mlp_ratio=2,
        crop_size=16, patch_size=8, frames_per_clip=4, tubelet_size=2,
        pred_hidden_size=48, pred_num_attention_heads=4, pred_num_hidden_layers=1,
    )
    return VJEPAAdapter(VJEPA2Model(config), hidden_size=48)


@pytest.mark.parametrize("mode,unfreeze", [("dual_head", 0), ("robot_head", 0),
                                          ("dual_head", 2)])
def test_vjepa_training_structures_update_only_intended_parameters(mode, unfreeze):
    adapter = tiny_vjepa()
    model = CrossDomainVideoEncoder(
        adapter, feature_dim=8, encoder_mode=mode, freeze_backbone=not unfreeze,
        unfreeze_last_n_blocks=unfreeze, dropout=0.,
    )
    assert model.human_backbone is model.robot_backbone
    blocks = adapter.get_transformer_blocks()
    assert len(blocks) == 24
    assert [index + 1 for index, block in enumerate(blocks)
            if any(p.requires_grad for p in block.parameters())] == ([23, 24] if unfreeze else [])
    assert isinstance(model.human_adapter, torch.nn.Identity) == (mode == "robot_head")
    assert not any(p.requires_grad for module in adapter.patch_embedding_modules()
                   for p in module.parameters())
    before = {name: parameter.detach().clone() for name, parameter in adapter.named_parameters()}
    output = model(torch.randn(2, 4, 3, 16, 16), torch.randn(2, 4, 3, 16, 16))
    loss = output["human_embeddings"][:, 0].sum() + output["robot_embeddings"][:, 1].sum()
    loss.backward()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    if unfreeze:
        for block in blocks[-2:]:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
        assert any(not torch.equal(parameter, before[name])
                   for name, parameter in adapter.named_parameters() if parameter.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.robot_adapter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.projector.parameters())


@pytest.mark.parametrize("count,message", [(-1, "non-negative"), (25, "exposes 24")])
def test_partial_finetune_rejects_invalid_block_counts(count, message):
    with pytest.raises(ValueError, match=message):
        CrossDomainVideoEncoder(tiny_vjepa(), unfreeze_last_n_blocks=count)


def test_partial_finetune_rejects_backbone_without_exposed_blocks():
    with pytest.raises(ValueError, match="exposes 0"):
        CrossDomainVideoEncoder(
            VideoMAEAdapter(_DummyVideoMAEBackbone(), hidden_size=4), unfreeze_last_n_blocks=2,
        )
