from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def ensure_directory(path: str | Path) -> Path:
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
