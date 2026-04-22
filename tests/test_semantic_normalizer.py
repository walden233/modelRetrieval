import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.normalizer import (
    build_label_canonical_text,
    normalize_capability_tags,
    normalize_environment_tags,
    normalize_scene_category,
    normalize_task_complexity,
    validate_annotation,
)


def test_normalize_capability_tags_applies_alias_and_filters():
    taxonomy = {"capability_tags": {"allowed_tags": ["grasp", "place"], "tag_aliases": {"pick": "grasp"}}}
    assert normalize_capability_tags(["pick", "invalid", "place"], taxonomy) == ["grasp", "place"]


def test_normalize_environment_tags_filters():
    taxonomy = {"environment_tags": {"allowed_tags": ["无障碍物", "动态环境"], "tag_aliases": {}}}
    assert normalize_environment_tags(["无障碍物", "invalid"], taxonomy) == ["无障碍物"]


def test_scalar_normalizers_fallback_to_unknown():
    taxonomy = {"task_complexity_options": ["高", "中", "低", "unknown"], "scene_category_options": ["工业", "家庭", "unknown"]}
    assert normalize_task_complexity("超高", taxonomy) == "unknown"
    assert normalize_scene_category("实验室", taxonomy) == "unknown"


def test_build_label_canonical_text_contains_new_dimensions():
    text = build_label_canonical_text(["grasp", "place"], "中", ["无障碍物"], "家庭")
    assert "task_complexity: 中" in text
    assert "scene_category: 家庭" in text


def test_validate_annotation_allows_unknown_scalar_labels():
    validate_annotation("move the cup", ["transport"], "unknown", ["无障碍物"], "工业")
