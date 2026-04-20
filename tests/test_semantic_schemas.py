import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.schemas import SemanticAnnotation


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
        task_description="the robot grasps a cup and places it on a shelf",
        capability_tags=["grasp", "place"],
        task_complexity="低",
        environment_tags=["无障碍物"],
        scene_category="工业",
        label_canonical_text="capabilities: grasp, place; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
        metadata={"text_embedding": [0.1], "label_embedding": [0.2]},
    )
    restored = SemanticAnnotation.from_dict(annotation.to_dict())
    assert restored.sample_id == "sample_1"
    assert restored.pair_id == "pair_1"
    assert restored.task_complexity == "低"
    assert restored.scene_category == "工业"
