from __future__ import annotations

import json
import re
from typing import Any, Dict

from bise.modalities.semantic.schemas import ParsedLabelResult
from bise.modalities.semantic.vlm_client import VLMResponse


class SemanticParseError(ValueError):
    pass


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise SemanticParseError(f"Could not locate JSON object in response: {text}")
        fragment = candidate[start : end + 1]
        try:
            return json.loads(fragment)
        except json.JSONDecodeError as exc:
            raise SemanticParseError(f"Failed to parse semantic JSON: {fragment}") from exc


def parse_description_response(raw_response: VLMResponse | Dict[str, Any] | str) -> str:
    payload = _coerce_payload(raw_response)
    task_description = str(payload.get("task_description", "")).strip()
    if not task_description:
        raise SemanticParseError("Missing task_description in description response.")
    return task_description


def parse_label_response(raw_response: VLMResponse | Dict[str, Any] | str) -> ParsedLabelResult:
    payload = _coerce_payload(raw_response)
    capability_tags = payload.get("capability_tags", [])
    environment_tags = payload.get("environment_tags", [])
    if not isinstance(capability_tags, list):
        raise SemanticParseError("capability_tags must be a list.")
    if not isinstance(environment_tags, list):
        raise SemanticParseError("environment_tags must be a list.")
    return ParsedLabelResult(
        capability_tags=[str(tag).strip() for tag in capability_tags if str(tag).strip()],
        task_complexity=str(payload.get("task_complexity", "unknown")).strip(),
        environment_tags=[str(tag).strip() for tag in environment_tags if str(tag).strip()],
        scene_category=str(payload.get("scene_category", "unknown")).strip(),
        raw_payload=payload,
    )


def _coerce_payload(raw_response: VLMResponse | Dict[str, Any] | str) -> Dict[str, Any]:
    if isinstance(raw_response, VLMResponse):
        return _extract_json_object(raw_response.content)
    if isinstance(raw_response, dict):
        return raw_response
    if isinstance(raw_response, str):
        return _extract_json_object(raw_response)
    raise SemanticParseError(f"Unsupported response type: {type(raw_response)!r}")
