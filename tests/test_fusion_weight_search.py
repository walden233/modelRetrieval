import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs" / "retrieval"))

from search_fusion_weights import (
    MODALITIES, build_weight_grid, evaluate_weights, filter_search_queries,
    main, selection_key, split_scene_id,
)
from bise.retrieval.system import load_retrieval_library
from bise.retrieval.system.io import save_json
from bise.retrieval.system.library import RetrievalLibrary
from bise.retrieval.system.schemas import FeatureRecord, GalleryItem, QueryItem
from test_retrieval_system import _build_tiny_library


def test_grid_and_selection():
    grid = build_weight_grid()
    assert len(grid) == 67
    assert all(sum(weights.values()) == pytest.approx(1) for weights in grid)
    assert dict(zip(MODALITIES, (0, 0, 1))) in grid
    assert dict.fromkeys(MODALITIES, 1 / 3) in grid
    rows = [{**weights, "MRR": 0.5, "NDCG@10": 0.6} for weights in grid]
    assert min(rows, key=selection_key)["video"] == pytest.approx(1 / 3)
    rows[0]["MRR"] = 0.7
    assert min(rows, key=selection_key) is rows[0]
    for invalid in (0, -1, 0.3, float("nan")):
        with pytest.raises(ValueError):
            build_weight_grid(invalid)


def test_search_filter_uses_both_training_splits(tmp_path):
    library = load_retrieval_library(_build_tiny_library(tmp_path))
    for item in library.query_items:
        item.feature_ids["trajectory"] = item.feature_ids["video"]
    for modality, train in (
        ("video", ["rh20t::task_0001::scene_1::cam_001"]),
        ("trajectory", ["task_9999/scene_1"]),
    ):
        root = tmp_path / modality
        save_json(root / "split_manifest.json", {
            "train": train, "val": ["task_0001/scene_1", "task_0002/scene_1"],
        })
        library.config[modality] = {"checkpoint": str(root / "best_model.pth")}
    filtered, audit = filter_search_queries(library)
    assert [query.task_id for query in filtered.query_items] == ["task_0002"]
    assert len(library.query_items) == 2
    assert audit["excluded_counts"]["training_scene"] == 1
    save_json(tmp_path / "trajectory" / "split_manifest.json", {
        "train": ["task_0002/scene_1"], "val": ["task_0001/scene_1"],
    })
    with pytest.raises(ValueError, match="No complete"):
        filter_search_queries(library)
    with pytest.raises(ValueError, match="Unsupported"):
        split_scene_id("unrecognized_id")


def test_fusion_happens_before_camera_max(tmp_path):
    records, arrays, gallery = {}, {}, []
    specs = [("task_0001", "a", (1, 0, 0)),
             ("task_0001", "b", (0, 1, 0)),
             ("task_0002", "a", (0.6, 0.6, 0.6))]
    for task, camera, scores in specs:
        scene = f"{task}/scene_1"
        features = {}
        for modality, score in zip(MODALITIES, scores):
            key = f"{scene}/{camera}/{modality}"
            arrays[key] = np.array([[score, np.sqrt(1 - score**2)]], dtype=np.float32)
            records[key] = FeatureRecord(key, scene, "robot", modality, key, 0, task, scene)
            features[modality] = key
        gallery.append(GalleryItem(f"{scene}/{camera}", scene, "robot", task, scene,
                                   camera_id=camera, feature_ids=features))
    features = {}
    for modality in MODALITIES:
        key = f"query/{modality}"
        arrays[key] = np.array([[1, 0]], dtype=np.float32)
        records[key] = FeatureRecord(key, "q", "human", modality, key, 0,
                                     "task_0001", "task_0001/scene_1")
        features[modality] = key
    query = QueryItem("q", "q", "human", "task_0001", "task_0001/scene_1",
                      feature_ids=features)
    library = RetrievalLibrary(tmp_path, {}, {}, {}, records, gallery, [query], arrays)
    result = evaluate_weights(library, {"fusion": {"calibration": "none"}},
                              dict(zip(MODALITIES, (0.5, 0.5, 0))))
    assert result["metrics"]["task"]["MRR"] == pytest.approx(0.5)
    np.testing.assert_allclose(result["similarity_matrix"], [[0.5, 0.6]])


def test_cli_saves_reusable_config_and_does_not_filter_cfg3_by_cfg2_scene(tmp_path, monkeypatch):
    roots = []
    checkpoints = {}
    for modality in ("video", "trajectory"):
        root = tmp_path / modality
        save_json(root / "split_manifest.json", {
            "train": ["task_0001/scene_1"], "val": ["task_0002/scene_1"],
        })
        checkpoints[modality] = {"checkpoint": str(root / "best_model.pth")}
    for cfg in ("cfg2", "cfg3"):
        root = _build_tiny_library(tmp_path / cfg)
        save_json(root / "library_config.json", {"cfg": cfg, **checkpoints})
        for name in ("query_human_eval", "gallery_robot"):
            path = root / "manifests" / f"{name}.jsonl"
            items = [json.loads(line) for line in path.read_text().splitlines()]
            for item in items:
                item["feature_ids"]["trajectory"] = item["feature_ids"]["video"]
            path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
        roots.append(root)
    config = tmp_path / "config.json"
    save_json(config, {"fusion": {"calibration": "none"}})
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", [
        "search", "--search-library", str(roots[0]), "--test-library", str(roots[1]),
        "--config", str(config), "--step", "1", "--output-dir", str(output),
    ])
    main()
    comparison = json.loads((output / "final_comparison.json").read_text())
    assert all(row["query_count"] == 2 for row in comparison)
    assert json.loads((output / "query_filter.json").read_text())["selected_query_count"] == 1
    assert json.loads((output / "best_config.json").read_text())["library_dir"] == str(roots[1])
    assert (output / "weight_search.png").exists()
    assert len(json.loads((output / "weight_search.json").read_text())) == 4
    with pytest.raises(SystemExit):
        main()


def test_cli_can_tune_on_all_complete_queries_in_one_library(tmp_path, monkeypatch):
    checkpoint_config = {}
    for modality in ("video", "trajectory"):
        root = tmp_path / modality
        save_json(root / "split_manifest.json", {"train": ["task_0001/scene_1"]})
        checkpoint_config[modality] = {"checkpoint": str(root / "best_model.pth")}
    library_root = _build_tiny_library(tmp_path / "cfg3")
    save_json(library_root / "library_config.json", {"cfg": "cfg3", **checkpoint_config})
    for name in ("query_human_eval", "gallery_robot"):
        path = library_root / "manifests" / f"{name}.jsonl"
        items = [json.loads(line) for line in path.read_text().splitlines()]
        for item in items:
            item["feature_ids"]["trajectory"] = item["feature_ids"]["video"]
        path.write_text("\n".join(json.dumps(item) for item in items) + "\n")
    config = tmp_path / "config.json"
    save_json(config, {"fusion": {"calibration": "none"}})
    output = tmp_path / "cfg3_weight_search"
    monkeypatch.setattr(sys, "argv", [
        "search", "--search-library", str(library_root),
        "--test-library", str(library_root),
        "--search-query-policy", "all_complete", "--config", str(config),
        "--step", "1", "--output-dir", str(output),
    ])

    main()

    audit = json.loads((output / "query_filter.json").read_text())
    assert audit["selected_query_count"] == 2
    assert audit["split_manifests"] == {}
    assert (output / "cfg3_tuning" / "selected" / "metrics.json").is_file()
    assert not (output / "cfg3_test").exists()
