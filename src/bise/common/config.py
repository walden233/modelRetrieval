import json
from pathlib import Path
from typing import Any, Dict


def load_json_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def override_seed(config: Dict[str, Any], seed: int) -> None:
    """Apply one effective seed to training, dynamic splits and dataset sampling."""
    config["seed"] = seed
    if "split" in config:
        config["split"]["seed"] = seed
    if "dataset" in config:
        config["dataset"]["seed"] = seed
