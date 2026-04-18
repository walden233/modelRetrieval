import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.parser import SemanticParseError, parse_description_response, parse_label_response


def test_parse_description_from_json_string():
    payload = '{"task_description": "the robot places a cup on a shelf"}'
    assert parse_description_response(payload) == "the robot places a cup on a shelf"


def test_parse_label_response_from_code_fence():
    payload = """```json
    {
      "capability_tags": ["grasp", "place"],
      "action_slots": {"object": "cup", "target": "shelf"}
    }
    ```"""
    parsed = parse_label_response(payload)
    assert parsed.capability_tags == ["grasp", "place"]
    assert parsed.action_slots.target == "shelf"


def test_parse_invalid_json_raises():
    with pytest.raises(SemanticParseError):
        parse_description_response("not-json")
