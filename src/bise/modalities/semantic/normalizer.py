from __future__ import annotations

from typing import Any, Dict, Iterable, List

from bise.modalities.semantic.schemas import ActionSlots


def normalize_capability_tags(tags: Iterable[str], taxonomy: Dict[str, Any]) -> List[str]:
    aliases = {str(key).strip().lower(): str(value).strip() for key, value in taxonomy.get("tag_aliases", {}).items()}
    allowed_tags = {str(tag).strip() for tag in taxonomy.get("allowed_tags", [])}
    normalized: List[str] = []
    for tag in tags:
        value = str(tag).strip().lower()
        if not value:
            continue
        canonical = aliases.get(value, value)
        if canonical in allowed_tags and canonical not in normalized:
            normalized.append(canonical)
    return normalized


def normalize_action_slots(payload: ActionSlots | Dict[str, Any]) -> ActionSlots:
    if isinstance(payload, ActionSlots):
        slots = payload
    else:
        slots = ActionSlots.from_dict(payload)
    return ActionSlots(
        object=str(slots.object).strip(),
        target=str(slots.target).strip(),
        verb=str(slots.verb).strip().lower(),
        tool=str(slots.tool).strip().lower(),
    )


def build_label_canonical_text(capability_tags: Iterable[str], action_slots: ActionSlots) -> str:
    tag_segment = ", ".join(sorted({str(tag).strip() for tag in capability_tags if str(tag).strip()}))
    parts = [
        f"capabilities: {tag_segment}" if tag_segment else "capabilities: none",
        f"object: {action_slots.object}",
        f"target: {action_slots.target}",
    ]
    if action_slots.verb:
        parts.append(f"verb: {action_slots.verb}")
    if action_slots.tool:
        parts.append(f"tool: {action_slots.tool}")
    return "; ".join(parts)


def validate_annotation(task_description: str, capability_tags: List[str], action_slots: ActionSlots) -> None:
    if not str(task_description).strip():
        raise ValueError("task_description is required.")
    if not capability_tags:
        raise ValueError("capability_tags must not be empty.")
    if not action_slots.object or not action_slots.target:
        raise ValueError("action_slots.object and action_slots.target are required.")
