"""Lossless CPU pixels for repeated deterministic video sampling."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch


class VideoPixelCache:
    def __init__(self, directory, processor, num_frames, strategy, stride):
        recipe = {
            "format": 1,
            "processor_class": type(processor).__module__ + "." + type(processor).__name__,
            "processor": processor.to_dict() if hasattr(processor, "to_dict") else None,
            "num_frames": num_frames, "strategy": strategy, "stride": stride,
        }
        self.strategy = strategy
        self.root = Path(directory) / self._digest(recipe)

    @staticmethod
    def _digest(value):
        encoded = json.dumps(value, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]

    def path(self, paths, seed):
        sources = []
        for source in paths:
            source = Path(source).resolve()
            stat = source.stat()
            sources.append((str(source), stat.st_size, stat.st_mtime_ns))
        key = {"sources": sources}
        if self.strategy not in {"uniform", "center_clip", "stride"}:
            key["seed"] = seed
        return self.root / (self._digest(key) + ".npz")

    def load(self, path):
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as saved:
                pixels = saved["pixel_values"]
            if pixels.ndim != 5 or pixels.shape[0] != 2:
                return None
            return torch.from_numpy(pixels)
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
            return None

    def save(self, path, pixels):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".", suffix=".npz",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, pixel_values=pixels.detach().cpu().numpy())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
