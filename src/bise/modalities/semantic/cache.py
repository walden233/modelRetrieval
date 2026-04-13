import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        else:
            self._cache = {}

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False), encoding="utf-8")
