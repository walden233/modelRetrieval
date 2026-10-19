"""Check explicit scene exclusions without changing legacy datasets or protocols."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs"))

from filter_scenes import filter_splits, reviewed_exclusions


def write_review(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("cfg", "scene_id", "decision", "reason"))
        writer.writerows(rows)


def test_only_explicit_exclusions_apply_to_the_requested_cfg(tmp_path):
    review = tmp_path / "review.csv"
    write_review(review, [
        ("cfg2", "task_0001/scene_1", "exclude", "Verified unrelated demonstration"),
        ("cfg2", "task_0001/scene_2", "keep", "Teacher is null; not a confirmed mismatch"),
        ("cfg3", "task_0001/scene_1", "keep", "Different dataset; valid pairing"),
    ])
    assert reviewed_exclusions(review, "cfg2") == {"task_0001/scene_1"}
    assert reviewed_exclusions(review, "cfg3") == set()
    mapping = {
        "train": ["task_0001/scene_1", "task_0001/scene_2"],
        "val": ["rh20t::task_0001::scene_1::camera_a"],
        "test": ["rh20t::task_0002::scene_1::camera_a",
                 "rh20t::task_0001::scene_1::camera_b",
                 "rh20t::task_0001::scene_2::camera_a"],
    }
    original = json.dumps(mapping)
    result = filter_splits(mapping, reviewed_exclusions(review, "cfg2"))
    assert result == {"train": ["task_0001/scene_2"], "val": [],
                      "test": ["rh20t::task_0002::scene_1::camera_a",
                               "rh20t::task_0001::scene_2::camera_a"]}
    assert json.dumps(mapping) == original


def test_review_requires_an_unambiguous_decision_and_reason(tmp_path):
    review = tmp_path / "review.csv"
    for decision, reason in (("null", "Null teacher"), ("exclude", "")):
        write_review(review, [("cfg2", "task_0001/scene_1", decision, reason)])
        with pytest.raises(ValueError, match="Invalid or duplicate"):
            reviewed_exclusions(review, "cfg2")
    write_review(review, [("cfg2", "task_0001/scene_1", "keep", "uncertain")] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        reviewed_exclusions(review, "cfg2")


def test_cli_writes_only_a_new_manifest_and_leaves_sources_untouched(tmp_path):
    source, output = tmp_path / "split.json", tmp_path / "filtered.json"
    mapping = {"train": ["task_0001/scene_1"], "val": [], "test": ["task_0002/scene_1"]}
    source.write_text(json.dumps(mapping))
    before = source.read_bytes()
    review = tmp_path / "review.csv"
    write_review(review, [("cfg2", "task_0001/scene_1", "exclude", "Confirmed wrong scene")])
    command = [sys.executable, str(ROOT / "runs/filter_scenes.py"), "--cfg", "cfg2",
               "--source", str(source), "--review", str(review)]
    result = subprocess.run(command + ["--dry-run"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["counts"]["train"]["removed"] == 1
    assert not output.exists()
    subprocess.run(command + ["--output", str(output)], capture_output=True, check=True)
    assert json.loads(output.read_text()) == {**mapping, "train": []}
    saved = output.read_bytes()
    assert subprocess.run(command + ["--output", str(output)], capture_output=True).returncode
    assert subprocess.run(command + ["--output", str(source)], capture_output=True).returncode
    assert source.read_bytes() == before and output.read_bytes() == saved


def test_current_visual_review_retains_all_27_scenes_without_automatic_null_filtering():
    review = ROOT / "docs/semantic/SCENE_REVIEW.csv"
    assert reviewed_exclusions(review, "cfg2") == reviewed_exclusions(review, "cfg3") == set()
    with review.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 27
    # This scene was explicitly accepted despite the different power-strip operations.
    assert next(row for row in rows if row["scene_id"] == "task_0052/scene_3")["decision"] == "keep"
