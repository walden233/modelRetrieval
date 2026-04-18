import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.normalizer import (
    build_label_canonical_text,
    normalize_action_slots,
    normalize_capability_tags,
    validate_annotation,
)
from bise.modalities.semantic.schemas import ActionSlots


def test_normalize_capability_tags_applies_alias_and_filters():
    taxonomy = {"allowed_tags": ["grasp", "place"], "tag_aliases": {"pick": "grasp"}}
    assert normalize_capability_tags(["pick", "invalid", "place"], taxonomy) == ["grasp", "place"]


def test_build_label_canonical_text_contains_object_target():
    slots = ActionSlots(object="cup", target="shelf", verb="place")
    text = build_label_canonical_text(["grasp", "place"], slots)
    assert "object: cup" in text
    assert "target: shelf" in text


def test_validate_annotation_rejects_empty_tags():
    with pytest.raises(ValueError):
        validate_annotation("move the cup", [], ActionSlots(object="cup", target="shelf"))


def test_normalize_action_slots_strips_values():
    slots = normalize_action_slots({"object": " cup ", "target": " shelf ", "verb": " Place "})
    assert slots.object == "cup"
    assert slots.verb == "place"
