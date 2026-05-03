from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_json(path: str | Path, default=None):
    candidate = Path(path)
    if not candidate.exists():
        return default
    return json.loads(candidate.read_text(encoding="utf-8"))


def save_json(path: str | Path, payload) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict]:
    candidate = Path(path)
    if not candidate.exists():
        return []
    with candidate.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    candidate = Path(path)
    candidate.parent.mkdir(parents=True, exist_ok=True)
    with candidate.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
