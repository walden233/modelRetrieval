from __future__ import annotations

from typing import Any, Dict, Iterable, List


def normalize_capability_tags(tags: Iterable[str], taxonomy: Dict[str, Any]) -> List[str]:
    capability_config = taxonomy.get("capability_tags", {})
    aliases = {str(key).strip().lower(): str(value).strip() for key, value in capability_config.get("tag_aliases", {}).items()}
    allowed_tags = {str(tag).strip() for tag in capability_config.get("allowed_tags", [])}
    normalized: List[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if not value:
            continue
        canonical = aliases.get(value, value)
        if canonical in allowed_tags and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def normalize_environment_tags(tags: Iterable[str], taxonomy: Dict[str, Any]) -> List[str]:
    environment_config = taxonomy.get("environment_tags", {})
    aliases = {str(key).strip().lower(): str(value).strip() for key, value in environment_config.get("tag_aliases", {}).items()}
    allowed_tags = {str(tag).strip() for tag in environment_config.get("allowed_tags", [])}
    normalized: List[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if not value:
            continue
        canonical = aliases.get(value, value)
        if canonical in allowed_tags and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def normalize_task_complexity(value: str, taxonomy: Dict[str, Any]) -> str:
    options = {str(item).strip() for item in taxonomy.get("task_complexity_options", [])}
    cleaned = str(value).strip()
    return cleaned if cleaned in options else "unknown"


def normalize_scene_category(value: str, taxonomy: Dict[str, Any]) -> str:
    options = {str(item).strip() for item in taxonomy.get("scene_category_options", [])}
    cleaned = str(value).strip()
    return cleaned if cleaned in options else "unknown"


def build_label_canonical_text(
    capability_tags: Iterable[str],
    task_complexity: str,
    environment_tags: Iterable[str],
    scene_category: str,
) -> str:
    capability_segment = ", ".join(sorted({str(tag).strip() for tag in capability_tags if str(tag).strip()}))
    environment_segment = ", ".join(sorted({str(tag).strip() for tag in environment_tags if str(tag).strip()}))
    parts = [
        f"capabilities: {capability_segment}" if capability_segment else "capabilities: none",
        f"task_complexity: {task_complexity}",
        f"environment: {environment_segment}" if environment_segment else "environment: none",
        f"scene_category: {scene_category}",
    ]
    return "; ".join(parts)


def validate_annotation(
    task_description: str,
    capability_tags: List[str],
    task_complexity: str,
    environment_tags: List[str],
    scene_category: str,
) -> None:
    if not str(task_description).strip():
        raise ValueError("task_description is required.")
    if not capability_tags:
        raise ValueError("capability_tags must not be empty.")
    if not environment_tags:
        raise ValueError("environment_tags must not be empty.")
