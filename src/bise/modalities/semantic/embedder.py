from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List

import numpy as np


class TextEmbedder:
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class HashTextEmbedder(TextEmbedder):
    def __init__(self, dimension: int = 32):
        self.dimension = dimension

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        matrix = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for index in range(self.dimension):
                digest = hashlib.sha256(f"{text}|{index}".encode("utf-8")).digest()
                vector[index] = int.from_bytes(digest[:4], "big") / 2**32
            norm = np.linalg.norm(vector)
            matrix.append(vector if norm == 0 else vector / norm)
        return np.vstack(matrix) if matrix else np.zeros((0, self.dimension), dtype=np.float32)


class TransformersTextEmbedder(TextEmbedder):
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 256):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.max_length = max_length

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.zeros((0, hidden_size), dtype=np.float32)
        with self.torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            pooled = self.torch.nn.functional.normalize(pooled, dim=1)
        return pooled.cpu().numpy().astype("float32")


def build_text_embedding(embedder: TextEmbedder, task_description: str) -> List[float]:
    return embedder.encode_texts([task_description])[0].tolist()


def build_label_embedding(embedder: TextEmbedder, label_canonical_text: str) -> List[float]:
    return embedder.encode_texts([label_canonical_text])[0].tolist()


def build_text_embedder(config: Dict[str, Any]) -> TextEmbedder:
    provider_name = str(config.get("provider_name", "hash"))
    if provider_name == "hash":
        return HashTextEmbedder(dimension=int(config.get("dimension", 32)))
    if provider_name == "transformers":
        return TransformersTextEmbedder(
            model_name=str(config["model_name"]),
            device=str(config.get("device", "cpu")),
            max_length=int(config.get("max_length", 256)),
        )
    raise ValueError(f"Unsupported text embedder provider: {provider_name}")
