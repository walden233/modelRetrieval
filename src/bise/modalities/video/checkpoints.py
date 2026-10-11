"""Locate raw video backbone weights shared across experiment batches."""

from pathlib import Path

from bise.common.paths import resolve_path


def raw_backbone_path(config: dict, run_dir: str | Path) -> Path:
    """Prefer a historical run snapshot, then use the shared model checkpoint."""
    local = Path(run_dir) / "backbone.pth"
    if local.is_file():
        return local
    name = config["model"]["backbone_name"].replace("/", "--")
    return resolve_path(Path("artifacts/models/video") / f"{name}.pth")
