import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.evaluator import (
    evaluate_annotation_retrieval,
    evaluate_semantic_retrieval_by_key,
    extract_annotation_embeddings,
    extract_positive_keys,
    split_annotations_by_role,
)
from bise.modalities.semantic.schemas import SemanticAnnotation


def test_evaluate_semantic_retrieval_by_pair_key():
    query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    gallery_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metrics = evaluate_semantic_retrieval_by_key(
        query_embeddings=query_embeddings,
        query_keys=["pair_1", "pair_2"],
        gallery_embeddings=gallery_embeddings,
        gallery_keys=["pair_1", "pair_2"],
        query_sample_ids=["pair_1_human", "pair_2_human"],
        gallery_sample_ids=["pair_1_robot", "pair_2_robot"],
    )
    assert metrics["valid_query_count"] == 2
    assert metrics["R@1"] == 1.0
    assert metrics["MRR"] == 1.0
    assert metrics["NDCG@10"] == 1.0
    assert metrics["Mean Percentage Rank"] == 0.5


def test_evaluate_semantic_retrieval_returns_finite_values_without_positives():
    query_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery_embeddings = np.asarray([[0.0, 1.0]], dtype=np.float32)
    metrics = evaluate_semantic_retrieval_by_key(
        query_embeddings=query_embeddings,
        query_keys=["pair_1"],
        gallery_embeddings=gallery_embeddings,
        gallery_keys=["pair_2"],
    )
    assert metrics["valid_queries"] == 0
    assert metrics["MRR"] == 0.0
    assert metrics["Mean Rank"] == 0.0
    assert metrics["NDCG@10"] == 0.0


def test_split_annotations_by_role_prefers_human_to_robot():
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
    queries, gallery = split_annotations_by_role(annotations, query_role="human", gallery_role="robot")
    assert len(queries) == 1
    assert len(gallery) == 1
    assert queries[0].video_role == "human"
    assert gallery[0].video_role == "robot"


def test_extract_annotation_embeddings_supports_combined_mode():
    annotations = [
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
            metadata={"text_embedding": [3.0, 0.0], "label_embedding": [0.0, 4.0]},
        ),
    ]
    embeddings, keys, sample_ids = extract_annotation_embeddings(annotations, mode="combined")
    assert embeddings.shape == (1, 2)
    assert np.allclose(np.linalg.norm(embeddings[0]), 1.0)
    assert keys == ["pair_1"]
    assert sample_ids == ["pair_1_human"]


def test_evaluate_annotation_retrieval_uses_pair_id_by_default():
    query_annotations = [
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
        )
    ]
    gallery_annotations = [
        SemanticAnnotation(
            sample_id="pair_1_robot",
            pair_id="pair_1",
            task_id="task_b",
            scene_id="scene_9",
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
        )
    ]
    metrics = evaluate_annotation_retrieval(
        query_annotations=query_annotations,
        gallery_annotations=gallery_annotations,
        mode="text",
    )
    assert metrics["R@1"] == 1.0
    assert metrics["NDCG@10"] == 1.0


def test_evaluate_annotation_retrieval_supports_scene_level_keys():
    query_annotations = [
        SemanticAnnotation(
            sample_id="task_a_scene_1_human",
            pair_id="task_a_scene_1",
            task_id="task_a",
            scene_id="task_a/scene_1",
            dataset_name="RH20T",
            video_role="human",
            video_path="/tmp/human_a.mp4",
            paired_video_path="/tmp/robot_a.mp4",
            cam_id="cam_0",
            task_description="human does task a",
            capability_tags=["grasp"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: grasp; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [1.0, 0.0], "label_embedding": [1.0, 0.0]},
        ),
        SemanticAnnotation(
            sample_id="task_b_scene_1_human",
            pair_id="task_b_scene_1",
            task_id="task_b",
            scene_id="task_b/scene_1",
            dataset_name="RH20T",
            video_role="human",
            video_path="/tmp/human_b.mp4",
            paired_video_path="/tmp/robot_b.mp4",
            cam_id="cam_0",
            task_description="human does task b",
            capability_tags=["place"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: place; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [0.0, 1.0], "label_embedding": [0.0, 1.0]},
        ),
    ]
    gallery_annotations = [
        SemanticAnnotation(
            sample_id="task_a_scene_1_robot",
            pair_id="task_a_scene_1",
            task_id="task_a",
            scene_id="task_a/scene_1",
            dataset_name="RH20T",
            video_role="robot",
            video_path="/tmp/robot_a.mp4",
            paired_video_path="/tmp/human_a.mp4",
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
            sample_id="task_b_scene_1_robot",
            pair_id="task_b_scene_1",
            task_id="task_b",
            scene_id="task_b/scene_1",
            dataset_name="RH20T",
            video_role="robot",
            video_path="/tmp/robot_b.mp4",
            paired_video_path="/tmp/human_b.mp4",
            cam_id="cam_0",
            task_description="robot does task b",
            capability_tags=["place"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: place; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [0.0, 1.0], "label_embedding": [0.0, 1.0]},
        ),
    ]
    metrics = evaluate_annotation_retrieval(
        query_annotations=query_annotations,
        gallery_annotations=gallery_annotations,
        mode="text",
        positive_key="scene_id",
    )
    assert metrics["R@1"] == 1.0
    assert metrics["valid_queries"] == 2


def test_extract_positive_keys_task_scopes_legacy_scene_ids():
    annotations = [
        SemanticAnnotation(
            sample_id="task_a_scene_1_human",
            pair_id="task_a_scene_1",
            task_id="task_a",
            scene_id="scene_1",
            dataset_name="RH20T",
            video_role="human",
            video_path="/tmp/human_a.mp4",
            paired_video_path="/tmp/robot_a.mp4",
            cam_id="cam_0",
            task_description="human does task a",
            capability_tags=["grasp"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: grasp; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [1.0, 0.0], "label_embedding": [1.0, 0.0]},
        ),
        SemanticAnnotation(
            sample_id="task_b_scene_1_human",
            pair_id="task_b_scene_1",
            task_id="task_b",
            scene_id="scene_1",
            dataset_name="RH20T",
            video_role="human",
            video_path="/tmp/human_b.mp4",
            paired_video_path="/tmp/robot_b.mp4",
            cam_id="cam_0",
            task_description="human does task b",
            capability_tags=["place"],
            task_complexity="低",
            environment_tags=["无障碍物"],
            scene_category="工业",
            label_canonical_text="capabilities: place; task_complexity: 低; environment: 无障碍物; scene_category: 工业",
            metadata={"text_embedding": [0.0, 1.0], "label_embedding": [0.0, 1.0]},
        ),
    ]

    assert extract_positive_keys(annotations, "scene_id") == ["task_a/scene_1", "task_b/scene_1"]
