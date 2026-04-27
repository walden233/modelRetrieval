import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video.models import VideoMAEAdapter


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
