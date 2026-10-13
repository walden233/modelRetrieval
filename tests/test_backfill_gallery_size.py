import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bise.common.fingerprints import file_hash, write_json
from bise.retrieval.metrics import calculate_label_retrieval_metrics

SPEC = importlib.util.spec_from_file_location("backfill_gallery_size", ROOT / "runs/backfill_gallery_size.py")
BACKFILL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKFILL)


def legacy_evaluation(root):
    output = root / "trajectory/N2/evaluations/cfg2"
    output.mkdir(parents=True)
    matrix = np.array([[.9, .8, .1], [.1, .2, .3]])
    metadata = {"human_scene_ids": ["a", "missing"], "robot_scene_ids": ["a", "a", "b"],
                "human_task_ids": ["x", "missing"], "robot_task_ids": ["x", "x", "y"]}
    metrics = {}
    for direction, scores, qdomain, gdomain in (
        ("human_to_robot", matrix, "human", "robot"),
        ("robot_to_human", matrix.T, "robot", "human"),
    ):
        metrics[direction] = {}
        for level in ("scene", "task"):
            result = calculate_label_retrieval_metrics(scores, metadata[f"{qdomain}_{level}_ids"],
                                                       metadata[f"{gdomain}_{level}_ids"])
            result.pop("gallery_size")
            metrics[direction][level] = result
    metrics["human_to_robot"]["scene"].pop("MPR-all")
    write_json(output / "metrics.json", metrics)
    write_json(output / "metadata.json", metadata)
    np.save(output / "similarity_matrix.npy", matrix)
    info = {"custom": "preserve", "evaluation": {"evaluator_sha256": "historical-code",
            "artifacts": {"metrics.json": file_hash(output / "metrics.json"), "metadata.json": "keep"}}}
    write_json(output / "run_info.json", info)
    return output, metrics, info


def test_backfill_preserves_scores_and_refreshes_only_metrics_fingerprint(tmp_path):
    output, original, original_info = legacy_evaluation(tmp_path)
    matrix_hash = file_hash(output / "similarity_matrix.npy")
    report = BACKFILL.backfill(tmp_path, apply=True)
    result = json.loads((output / "metrics.json").read_text())
    stripped = copy.deepcopy(result)
    for direction, levels in stripped.items():
        for metrics in levels.values():
            assert metrics.pop("gallery_size") == (3 if direction == "human_to_robot" else 2)
    assert stripped == original
    assert result["human_to_robot"]["scene"]["valid_queries"] == 1
    assert result["robot_to_human"]["scene"]["valid_queries"] == 2
    info = json.loads((output / "run_info.json").read_text())
    expected_info = copy.deepcopy(original_info)
    expected_info["evaluation"]["artifacts"]["metrics.json"] = file_hash(output / "metrics.json")
    assert info == expected_info
    assert file_hash(output / "similarity_matrix.npy") == matrix_hash
    assert report["fields_added"] == 4


def test_backfill_dry_run_and_repeated_application_do_not_rewrite(tmp_path):
    output, _, _ = legacy_evaluation(tmp_path)
    before = {name: (output / name).read_bytes() for name in ("metrics.json", "run_info.json")}
    assert BACKFILL.backfill(tmp_path)["fields_to_add"] == 4
    assert before == {name: (output / name).read_bytes() for name in before}
    BACKFILL.backfill(tmp_path, apply=True)
    after = {name: (output / name).read_bytes() for name in before}
    assert BACKFILL.backfill(tmp_path, apply=True)["fields_added"] == 0
    assert after == {name: (output / name).read_bytes() for name in after}


def test_backfill_preflights_every_file_before_mutating_results(tmp_path):
    good, _, _ = legacy_evaluation(tmp_path / "a")
    bad, metrics, _ = legacy_evaluation(tmp_path / "z")
    before = (good / "metrics.json").read_bytes()
    metrics["human_to_robot"]["scene"]["valid_queries"] = 99
    write_json(bad / "metrics.json", metrics)
    with pytest.raises(ValueError, match="valid_queries"):
        BACKFILL.backfill(tmp_path, apply=True)
    assert (good / "metrics.json").read_bytes() == before


def test_backfill_rejects_preexisting_fingerprint_mismatch(tmp_path):
    output, _, _ = legacy_evaluation(tmp_path)
    path = output / "metrics.json"
    path.write_bytes(path.read_bytes() + b"\n")
    before = (output / "metrics.json").read_bytes()
    with pytest.raises(ValueError, match="fingerprint"):
        BACKFILL.backfill(tmp_path, apply=True)
    assert (output / "metrics.json").read_bytes() == before


def test_backfill_records_an_explicit_additive_evaluator_migration(tmp_path):
    output, _, _ = legacy_evaluation(tmp_path)
    report = BACKFILL.backfill(tmp_path, apply=True,
                               evaluator_hash_migration={"historical-code": "gallery-size-schema"})
    evaluation = json.loads((output / "run_info.json").read_text())["evaluation"]
    assert report["evaluator_fingerprints_migrated"] == 1
    assert evaluation["evaluator_sha256"] == "gallery-size-schema"
    assert evaluation["metric_schema_migrations"][0]["evaluator_sha256_before"] == "historical-code"
    assert evaluation["metric_schema_migrations"][0]["existing_metrics_reproduced"] is True
