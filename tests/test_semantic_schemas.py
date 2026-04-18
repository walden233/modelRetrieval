import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.schemas import ActionSlots, SemanticAnnotation


def test_action_slots_requires_object_and_target():
    with pytest.raises(ValueError):
        ActionSlots(object="", target="shelf")
    with pytest.raises(ValueError):
        ActionSlots(object="cup", target="")


def test_semantic_annotation_to_dict_roundtrip():
    annotation = SemanticAnnotation(
        sample_id="sample_1",
        pair_id="pair_1",
        task_id="task_1",
        scene_id="scene_1",
        dataset_name="RH20T",
        video_role="robot",
        video_path="/tmp/robot.mp4",
        paired_video_path="/tmp/human.mp4",
        cam_id="cam_0",
        task_description="the robot grasps a cup",
        capability_tags=["grasp"],
        action_slots=ActionSlots(object="cup", target="table"),
        label_canonical_text="capabilities: grasp; object: cup; target: table",
        metadata={"text_embedding": [0.1], "label_embedding": [0.2]},
    )
    restored = SemanticAnnotation.from_dict(annotation.to_dict())
    assert restored.sample_id == "sample_1"
    assert restored.pair_id == "pair_1"
    assert restored.action_slots.object == "cup"
