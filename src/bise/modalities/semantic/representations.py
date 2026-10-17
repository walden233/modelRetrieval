"""Task-only response schemas and independent retrieval representations."""

import json
import re
from pathlib import Path

from bise.common.paths import resolve_path
from bise.modalities.semantic.parser import _extract_json_object


REPRESENTATIONS = ("S0", "S1", "S2", "S3", "S4", "S5")
# "Orange" also names a manipulated fruit; do not reject that task noun lexically.
COLORS = (r'\b(red|green|blue|yellow|white|black|purple|pink|brown|grey|gray|'
          r'colorful|multicolored)\b')
LABEL_FIELDS = ("capability_tags", "task_complexity", "environment_tags", "scene_category")
VIDEO_SCHEMA = {
    "baseline_description": "string: one concise task sentence without incidental appearance",
    "description": "string",
    "labels": {"capability_tags": ["enum"], "task_complexity": "enum",
               "environment_tags": ["enum"], "scene_category": "enum"},
    "object_action_text": "string: Objects: ... Operations: ... [Result: ...]",
    "action_focus_text": "string: Objects: ... Operations: ... [Result: ...]",
    "structured": {
        "objects": [{"id": "o1", "name": "string", "role": "string"}],
        "operations": [{"action": "string", "object": "object ID or null",
                        "actor": "primary/other/joint/unknown",
                        "target": "null or {object_id: ID} or {location: string}"}],
        "result": "string or null",
    },
}


def prompt_parts(path: str | Path, text: str | None = None) -> tuple[str, dict[str, str]]:
    text = text if text is not None else resolve_path(path).read_text(encoding="utf-8")
    sections = text.split("\n## ")
    return sections[0], {
        section.split("\n", 1)[0]: section.split("\n", 1)[1].strip()
        for section in sections[1:]
    }


def video_prompt(config: dict, role: str) -> str:
    if role not in ("human", "robot"):
        raise ValueError(f"Invalid video role: {role}")
    common, sections = prompt_parts(config['video_prompt'], config.get('_video_prompt_text'))
    parts = [common, sections[f"{role.title()} focus"]]
    parts.extend([sections["Output instructions"],
                  "Allowed label values:\n" + json.dumps(config["taxonomy"]),
                  "Required response shape:\n" + json.dumps(VIDEO_SCHEMA)])
    return "\n\n".join(parts)


def _text(value, field: str, nullable: bool = False):
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _strings(value, field: str) -> list:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{field} must be a string list")
    return list(dict.fromkeys(x.strip() for x in value))


def task_text(value, field: str, nullable: bool = False):
    text = _text(value, field, nullable)
    if text and re.search(
            r'\b(uncertain|uncertainty|unclear|unobservable|not visible|not observable|'
            r'cannot (?:determine|confirm|see)|confidence|visibility|insufficient evidence)\b',
            text, re.I):
        raise ValueError(f"{field}: omit observation commentary; retain task content only")
    return text


def validate_video(content: str | dict, taxonomy: dict, role: str | None = None) -> dict:
    data = _extract_json_object(content) if isinstance(content, str) else content
    if not isinstance(data, dict) or set(data) != set(VIDEO_SCHEMA):
        raise ValueError(f"Response requires exactly {list(VIDEO_SCHEMA)}")
    data['baseline_description'] = validate_description(
        {'description': data['baseline_description']}, no_colors=True)['description']
    for key in ("description", "object_action_text", "action_focus_text"):
        data[key] = task_text(data[key], key)
    for key in ("object_action_text", "action_focus_text"):
        if not all(section in data[key].lower() for section in ("objects:", "operations:")):
            raise ValueError(f"{key} needs Objects: and Operations: sections")
    labels = data["labels"]
    normalization_notes = []
    if not isinstance(labels, dict) or set(labels) != set(LABEL_FIELDS):
        raise ValueError("labels requires the original four dimensions")
    for key in LABEL_FIELDS:
        values = _strings(labels[key], key) if key.endswith("_tags") else [labels[key]]
        unsupported = [value for value in values if value not in taxonomy[key]]
        if unsupported:
            normalization_notes.append({"field": key, "unsupported": unsupported})
            values = [value for value in values if value in taxonomy[key]]
            if not key.endswith('_tags'):
                values = ['unknown']
        labels[key] = values if key.endswith("_tags") else values[0]
    structure = data["structured"]
    if not isinstance(structure, dict) or set(structure) != {"objects", "operations", "result"}:
        raise ValueError("structured requires objects, operations and result")
    objects, operations = structure["objects"], structure["operations"]
    if not isinstance(objects, list) or not isinstance(operations, list):
        raise ValueError("objects and operations must be lists")
    ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {"id", "name", "role"}:
            raise ValueError("An object requires id/name/role")
        for key in obj:
            obj[key] = task_text(obj[key], f"object.{key}")
        if obj["id"] in ids:
            raise ValueError("Duplicate local object ID")
        ids.add(obj["id"])
    for operation in operations:
        if (not isinstance(operation, dict) or
                set(operation) != {"action", "object", "actor", "target"}):
            raise ValueError("An operation requires action/object/actor/target")
        operation["action"] = task_text(operation["action"], "operation.action")
        if operation["object"] is not None and operation["object"] not in ids:
            raise ValueError("Operation references a missing object")
        if operation["actor"] not in ("primary", "other", "joint", "unknown"):
            raise ValueError("Invalid actor attribution")
        target = operation["target"]
        if target is not None:
            if not isinstance(target, dict) or set(target) not in ({"object_id"}, {"location"}):
                raise ValueError("target must be null, object_id or location")
            if "object_id" in target and target["object_id"] not in ids:
                raise ValueError("Target references a missing object")
            if "location" in target:
                task_text(target["location"], "target.location")
    structure["result"] = task_text(structure["result"], "result", nullable=True)
    if role == 'robot' and re.search(r'\b(human|person|people|man|woman)\b',
                                    data['description'], re.I):
        interaction = any(re.search(r'\b(handover|hand over|give|receive|pass|cooperate|assist)\b',
                                    op['action'], re.I) for op in operations)
        if not interaction and not any(op['actor'] in ('other', 'joint') for op in operations):
            raise ValueError('Remove unrelated people from the robot task description')
    if normalization_notes:
        data['normalization_notes'] = normalization_notes
    return data


def representation_texts(annotation: dict, taxonomy: dict) -> dict[str, str]:
    if 'representation' in annotation:
        return {annotation['representation']: annotation['text']}
    labels = annotation["labels"]
    parts = []
    for key in LABEL_FIELDS:
        value = labels[key]
        if isinstance(value, list):
            value = ", ".join(x for x in taxonomy[key] if x in value) or "unknown"
        parts.append(f"{key}: {value}")
    structure = annotation["structured"]
    ordered = {
        "objects": [{key: obj[key] for key in ("id", "name", "role")}
                    for obj in structure["objects"]],
        "operations": [{key: op[key] for key in ("action", "object", "actor", "target")}
                       for op in structure["operations"]],
        "result": structure["result"],
    }
    return {
        "S0": annotation["baseline_description"],
        "S1": annotation["description"],
        "S2": annotation["description"] + "\nLabels: " + "; ".join(parts),
        "S3": annotation["object_action_text"],
        "S4": annotation["action_focus_text"],
        "S5": json.dumps(ordered, ensure_ascii=False, separators=(",", ":")),
    }


def validate_description(content: str | dict, nullable: bool = False,
                         no_colors: bool = False) -> dict:
    data = _extract_json_object(content) if isinstance(content, str) else content
    if not isinstance(data, dict) or set(data) != {'description'}:
        raise ValueError('Response requires description only')
    data['description'] = task_text(data['description'], 'description', nullable)
    if no_colors and re.search(COLORS, data['description'], re.I):
        raise ValueError('description contains color; return the concise task without color words')
    return data


def validate_scene(content: str | dict, observations: dict) -> dict:
    return validate_description(content, nullable=True)
