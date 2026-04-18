import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.pipeline import run_semantic_annotation_pipeline


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_semantic_pipeline_runs_end_to_end(tmp_path: Path):
    robot_video = tmp_path / "cam_0_robot.mp4"
    human_video = tmp_path / "cam_0_human.mp4"
    robot_video.write_bytes(b"")
    human_video.write_bytes(b"")

    manifest_path = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "sample_id": "task_1_scene_1_robot",
                "pair_id": "task_1_scene_1",
                "task_id": "task_1",
                "scene_id": "scene_1",
                "dataset_name": "RH20T",
                "video_role": "robot",
                "video_path": str(robot_video),
                "paired_video_path": str(human_video),
                "cam_id": "cam_0",
            }
        ],
    )

    provider_path = tmp_path / "provider.json"
    _write_json(provider_path, {"provider_name": "stub"})
    embedder_path = tmp_path / "embedder.json"
    _write_json(embedder_path, {"provider_name": "hash", "dimension": 8})
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_json(
        taxonomy_path,
        {"allowed_tags": ["transport"], "tag_aliases": {"move": "transport"}},
    )
    desc_prompt_path = tmp_path / "prompt_description.json"
    _write_json(
        desc_prompt_path,
        {
            "version": "description_prompt_v1",
            "system_prompt": "Return JSON",
            "user_template": "Describe sample {sample_id}",
            "output_schema": {"task_description": "string"},
        },
    )
    label_prompt_path = tmp_path / "prompt_label.json"
    _write_json(
        label_prompt_path,
        {
            "version": "label_prompt_v1",
            "system_prompt": "Return JSON",
            "user_template": "Label sample {sample_id}",
            "output_schema": {"capability_tags": ["string"], "action_slots": {"object": "string", "target": "string"}},
        },
    )

    config = {
        "dataset_type": "rh20t",
        "dataset_variant": "test",
        "output_prefix": str(tmp_path / "semantic"),
        "manifest_path": str(manifest_path),
        "raw_response_path": str(tmp_path / "raw.jsonl"),
        "normalized_annotation_path": str(tmp_path / "annotations.jsonl"),
        "sample_ids_path": str(tmp_path / "sample_ids.json"),
        "text_embedding_path": str(tmp_path / "text.npy"),
        "label_embedding_path": str(tmp_path / "label.npy"),
        "errors_path": str(tmp_path / "errors.jsonl"),
        "feature_store_path": str(tmp_path / "feature_store.json"),
        "cache_path": str(tmp_path / "cache.json"),
        "provider_config_path": str(provider_path),
        "embedder_config_path": str(embedder_path),
        "taxonomy_config_path": str(taxonomy_path),
        "description_prompt_config_path": str(desc_prompt_path),
        "label_prompt_config_path": str(label_prompt_path),
        "frame_count": 0,
        "skip_completed": True,
    }

    summary = run_semantic_annotation_pipeline(config)
    assert summary["count"] == 1
    assert np.load(tmp_path / "text.npy").shape == (1, 8)
    assert np.load(tmp_path / "label.npy").shape == (1, 8)
    assert json.loads((tmp_path / "sample_ids.json").read_text(encoding="utf-8")) == ["task_1_scene_1_robot"]

    rerun = run_semantic_annotation_pipeline(config)
    assert rerun["count"] == 1
