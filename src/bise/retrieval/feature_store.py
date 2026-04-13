import json
from pathlib import Path
from typing import Iterable

from bise.common.schemas import EmbeddingSample


class FeatureStore:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

    def save(self, samples: Iterable[EmbeddingSample]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [sample.to_dict() for sample in samples]
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self):
        return json.loads(self.output_path.read_text(encoding="utf-8"))
