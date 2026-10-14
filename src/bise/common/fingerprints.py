"""Small, streaming fingerprints for frozen experiment inputs."""

import hashlib
import json
from pathlib import Path


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def without_sha256(value):
    """Remove obsolete checksum fields while preserving manifest identities and order."""
    if isinstance(value, dict):
        return {key: without_sha256(item) for key, item in value.items()
                if "sha256" not in key.lower()}
    if isinstance(value, list):
        return [without_sha256(item) for item in value]
    return value


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
