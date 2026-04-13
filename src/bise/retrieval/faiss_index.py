from typing import Iterable, Tuple

import numpy as np


class FaissIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._index = None

    def build(self, embeddings: Iterable[Iterable[float]]) -> None:
        import faiss

        matrix = np.asarray(list(embeddings), dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("Embedding matrix shape does not match FAISS index dimension.")
        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(matrix)

    def search(self, queries: Iterable[Iterable[float]], top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built.")
        matrix = np.asarray(list(queries), dtype="float32")
        return self._index.search(matrix, top_k)

    def save(self, output_path: str) -> None:
        import faiss

        if self._index is None:
            raise RuntimeError("Index has not been built.")
        faiss.write_index(self._index, output_path)

    def load(self, input_path: str) -> None:
        import faiss

        self._index = faiss.read_index(input_path)
