import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.retrieval.system import RetrievalQuery, evaluate_retrieval_system, load_retrieval_library, retrieve_top_k
from bise.retrieval.system.io import save_json, save_jsonl
from bise.retrieval.system.keys import make_entity_key, normalize_camera_id, normalize_scene_id


def test_retrieval_key_normalization():
    assert normalize_scene_id("task_0001", "scene_1") == "task_0001/scene_1"
    assert normalize_scene_id("task_0001", "task_0001/scene_1") == "task_0001/scene_1"
    assert normalize_camera_id("cam_037") == "037"
    assert make_entity_key("RH20T", "cfg2", "task_0001", "scene_1") == "rh20t::cfg2::task_0001::task_0001/scene_1"


def test_retrieve_top_k_with_missing_query_modalities(tmp_path: Path):
    library_dir = _build_tiny_library(tmp_path)
    library = load_retrieval_library(library_dir)
    query = RetrievalQuery(query_id="q", video_embedding=[1.0, 0.0])
    results = retrieve_top_k(
        library,
        query,
        config={
            "modalities": ["video", "trajectory", "semantic_text"],
            "fusion": {"calibration": "none", "weights": {"video": 1.0}, "missing_policy": "renormalize"},
        },
        top_k=2,
    )
    assert results[0].scene_id == "task_0001/scene_1"
    assert results[0].modality_scores["video"] > results[1].modality_scores["video"]


def test_evaluate_retrieval_system_scene_and_mixed_metrics(tmp_path: Path):
    library_dir = _build_tiny_library(tmp_path)
    library = load_retrieval_library(library_dir)
    result = evaluate_retrieval_system(
        library,
        config={
            "modalities": ["video", "semantic_text"],
            "fusion": {"calibration": "none", "weights": {"video": 0.7, "semantic_text": 0.3}},
            "aggregation": {"camera_to_scene": "max"},
            "scene_task_mixed": {"scene_gain": 1.0, "task_gain": 0.3},
        },
        level="mixed",
        require_modalities=["video", "semantic_text"],
        top_k=2,
    )
    assert result["query_count"] == 2
    assert result["metrics"]["scene"]["R@1"] == 1.0
    assert result["metrics"]["mixed"]["MixedNDCG@10"] == 1.0
    assert result["cases"][0]["retrieved"][0]["is_scene_positive"]


def _build_tiny_library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    (root / "features").mkdir(parents=True)
    (root / "manifests").mkdir()
    np.save(root / "features" / "video_robot.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "features" / "video_human.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "features" / "semantic_text_robot.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "features" / "semantic_text_human.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    save_json(root / "library_config.json", {})
    save_json(root / "build_info.json", {})
    save_json(root / "coverage.json", {})
    records = []
    gallery = []
    queries = []
    scenes = []
    for index, task_id in enumerate(("task_0001", "task_0002")):
        scene_id = f"{task_id}/scene_1"
        entity_key = f"rh20t::cfg2::{task_id}::{scene_id}"
        scenes.append(
            {
                "entity_key": entity_key,
                "dataset_name": "rh20t",
                "cfg": "cfg2",
                "task_id": task_id,
                "scene_id": scene_id,
                "scene_name": "scene_1",
                "camera_ids": ["001"],
                "available_modalities": {"video": True, "semantic_text": True},
            }
        )
        for domain in ("robot", "human"):
            item_feature_ids = {}
            for modality in ("video", "semantic_text"):
                array_path = f"features/{modality}_{domain}.npy"
                feature_id = f"{entity_key}::001::{domain}::{modality}"
                records.append(
                    {
                        "feature_id": feature_id,
                        "entity_key": entity_key,
                        "domain": domain,
                        "modality": modality,
                        "array_path": array_path,
                        "row_index": index,
                        "task_id": task_id,
                        "scene_id": scene_id,
                        "camera_id": "001",
                        "source_path": "",
                        "metadata": {},
                    }
                )
                item_feature_ids[modality] = feature_id
            if domain == "robot":
                gallery.append(
                    {
                        "gallery_id": f"{entity_key}::001::robot",
                        "entity_key": entity_key,
                        "domain": "robot",
                        "task_id": task_id,
                        "scene_id": scene_id,
                        "camera_id": "001",
                        "feature_ids": item_feature_ids,
                        "metadata": {},
                    }
                )
            else:
                queries.append(
                    {
                        "query_id": f"{entity_key}::001::human",
                        "entity_key": entity_key,
                        "domain": "human",
                        "task_id": task_id,
                        "scene_id": scene_id,
                        "camera_id": "001",
                        "feature_ids": item_feature_ids,
                        "metadata": {},
                    }
                )
    save_jsonl(root / "manifests" / "feature_records.jsonl", records)
    save_jsonl(root / "manifests" / "gallery_robot.jsonl", gallery)
    save_jsonl(root / "manifests" / "query_human_eval.jsonl", queries)
    save_jsonl(root / "manifests" / "scenes.jsonl", scenes)
    return root
