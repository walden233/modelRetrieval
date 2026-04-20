import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.evaluator import evaluate_semantic_retrieval_by_task, split_cross_role_annotations
from bise.modalities.semantic.schemas import SemanticAnnotation


def test_semantic_retrieval_by_task():
    query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metrics = evaluate_semantic_retrieval_by_task(
        query_embeddings,
        ["task_a", "task_b"],
        gallery_embeddings,
        ["task_a", "task_b"],
    )
    assert metrics["R@1"] == 1.0
    assert metrics["MRR"] == 1.0


def test_split_default_cross_role_sets_prefers_human_to_robot():
    annotations = [
        SemanticAnnotation(
            sample_id="pair_1_robot",
            pair_id="pair_1",
            task_id="task_a",
            scene_id="scene_1",
            dataset_name="RH20T",
            video_role="robot",
            video_path="/tmp/robot.mp4",
            paired_video_path="/tmp/human.mp4",
            cam_id="cam_0",
            task_description="robot does task a",
            capability_tags=["grasp"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: grasp; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [1.0, 0.0], "label_embedding": [1.0, 0.0]},
        ),
        SemanticAnnotation(
            sample_id="pair_1_human",
            pair_id="pair_1",
            task_id="task_a",
            scene_id="scene_1",
            dataset_name="RH20T",
            video_role="human",
            video_path="/tmp/human.mp4",
            paired_video_path="/tmp/robot.mp4",
            cam_id="cam_0",
            task_description="human does task a",
            capability_tags=["grasp"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: grasp; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [1.0, 0.0], "label_embedding": [1.0, 0.0]},
        ),
    ]
    queries, gallery = split_cross_role_annotations(annotations)
    assert len(queries) == 1
    assert len(gallery) == 1
    assert queries[0].video_role == "human"
    assert gallery[0].video_role == "robot"
