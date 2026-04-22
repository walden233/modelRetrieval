import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.batch import build_semantic_batch_requests, ingest_semantic_batch_results


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_build_semantic_batch_requests_creates_jsonl_shards(tmp_path: Path):
    video_path = tmp_path / "cam_0_robot.mp4"
    paired_path = tmp_path / "cam_0_human.mp4"
    video_path.write_bytes(b"")
    paired_path.write_bytes(b"")

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
                "video_path": str(video_path),
                "paired_video_path": str(paired_path),
                "cam_id": "cam_0",
                "joint_prompt_version": "joint_prompt_with_taxonomy_v1",
            }
        ],
    )

    provider_path = tmp_path / "provider.json"
    _write_json(
        provider_path,
        {
            "provider_name": "openai_compatible",
            "base_url": "https://example.com/v4",
            "model_name": "glm-4.6v",
            "thinking_type": "disabled",
        },
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_json(
        taxonomy_path,
        {
            "capability_tags": {"allowed_tags": ["transport"], "tag_aliases": {}},
            "task_complexity_options": ["高", "中", "低", "unknown"],
            "environment_tags": {"allowed_tags": ["无障碍物", "unknown"], "tag_aliases": {}},
            "scene_category_options": ["工业", "unknown"],
        },
    )
    joint_prompt_path = tmp_path / "joint_prompt.json"
    _write_json(
        joint_prompt_path,
        {
            "version": "joint_prompt_with_taxonomy_v1",
            "system_prompt": "Return JSON only.",
            "user_template": "Label sample {sample_id} from {video_role_phrase}.",
            "output_schema": {
                "task_description": "string",
                "capability_tags": ["string"],
                "task_complexity": "string",
                "environment_tags": ["string"],
                "scene_category": "string",
            },
        },
    )

    config = {
        "dataset_type": "rh20t",
        "dataset_variant": "test",
        "output_prefix": str(tmp_path / "semantic"),
        "execution_mode": "batch",
        "prompt_mode": "single_stage",
        "manifest_path": str(manifest_path),
        "provider_config_path": str(provider_path),
        "taxonomy_config_path": str(taxonomy_path),
        "joint_prompt_config_path": str(joint_prompt_path),
        "raw_response_path": str(tmp_path / "raw.jsonl"),
        "normalized_annotation_path": str(tmp_path / "annotations.jsonl"),
        "errors_path": str(tmp_path / "errors.jsonl"),
        "frame_count": 0,
        "batch_endpoint": "/v4/chat/completions",
        "batch_max_requests_per_file": 50000,
        "batch_max_file_bytes": 100000000,
    }

    summary = build_semantic_batch_requests(config)
    assert summary["request_count"] == 1
    request_manifest = json.loads((tmp_path / "semantic" / "rh20t" / "test" / "batch" / "request_files.json").read_text(encoding="utf-8"))
    assert len(request_manifest["files"]) == 1
    request_file_path = Path(request_manifest["files"][0]["request_file_path"])
    request_lines = [json.loads(line) for line in request_file_path.read_text(encoding="utf-8").splitlines()]
    assert request_lines[0]["custom_id"] == "task_1_scene_1_robot"
    assert request_lines[0]["url"] == "/v4/chat/completions"
    assert request_lines[0]["body"]["model"] == "glm-4.6v"


def test_ingest_semantic_batch_results_updates_semantic_outputs(tmp_path: Path):
    video_path = tmp_path / "cam_0_robot.mp4"
    paired_path = tmp_path / "cam_0_human.mp4"
    video_path.write_bytes(b"")
    paired_path.write_bytes(b"")

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
                "video_path": str(video_path),
                "paired_video_path": str(paired_path),
                "cam_id": "cam_0",
                "joint_prompt_version": "joint_prompt_with_taxonomy_v1",
                "taxonomy_version": "taxonomy_v1",
            }
        ],
    )
    provider_path = tmp_path / "provider.json"
    _write_json(
        provider_path,
        {
            "provider_name": "openai_compatible",
            "base_url": "https://example.com/v4",
            "model_name": "glm-4.6v",
            "thinking_type": "disabled",
        },
    )
    embedder_path = tmp_path / "embedder.json"
    _write_json(embedder_path, {"provider_name": "hash", "dimension": 8})
    taxonomy_path = tmp_path / "taxonomy.json"
    _write_json(
        taxonomy_path,
        {
            "capability_tags": {"allowed_tags": ["transport"], "tag_aliases": {}},
            "task_complexity_options": ["高", "中", "低", "unknown"],
            "environment_tags": {"allowed_tags": ["无障碍物", "unknown"], "tag_aliases": {}},
            "scene_category_options": ["工业", "unknown"],
        },
    )
    output_file_path = tmp_path / "batch_output.jsonl"
    _write_jsonl(
        output_file_path,
        [
            {
                "custom_id": "task_1_scene_1_robot",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "task_description": "the robot transports an object to a target location",
                                            "capability_tags": ["transport"],
                                            "task_complexity": "低",
                                            "environment_tags": ["无障碍物"],
                                            "scene_category": "工业",
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            }
                        ]
                    },
                },
            }
        ],
    )
    jobs_path = tmp_path / "jobs.json"
    _write_json(
        jobs_path,
        {
            "jobs": [
                {
                    "batch_id": "batch_1",
                    "status": "completed",
                    "output_file_path": str(output_file_path),
                }
            ]
        },
    )

    config = {
        "dataset_type": "rh20t",
        "dataset_variant": "test",
        "output_prefix": str(tmp_path / "semantic"),
        "execution_mode": "batch",
        "prompt_mode": "single_stage",
        "manifest_path": str(manifest_path),
        "provider_config_path": str(provider_path),
        "embedder_config_path": str(embedder_path),
        "taxonomy_config_path": str(taxonomy_path),
        "raw_response_path": str(tmp_path / "raw.jsonl"),
        "normalized_annotation_path": str(tmp_path / "annotations.jsonl"),
        "sample_ids_path": str(tmp_path / "sample_ids.json"),
        "text_embedding_path": str(tmp_path / "text.npy"),
        "label_embedding_path": str(tmp_path / "label.npy"),
        "errors_path": str(tmp_path / "errors.jsonl"),
        "feature_store_path": str(tmp_path / "feature_store.json"),
        "cache_path": str(tmp_path / "cache.json"),
        "batch_jobs_path": str(jobs_path),
    }

    summary = ingest_semantic_batch_results(config)
    assert summary["ingested_count"] == 1
    annotations = [json.loads(line) for line in (tmp_path / "annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert annotations[0]["task_description"] == "the robot transports an object to a target location"
    assert annotations[0]["capability_tags"] == ["transport"]
    assert annotations[0]["metadata"]["prompt_mode"] == "single_stage"
    raw_records = [json.loads(line) for line in (tmp_path / "raw.jsonl").read_text(encoding="utf-8").splitlines()]
    assert raw_records[0]["request_type"] == "joint"
    cache = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    assert len(cache) == 1
    assert np.load(tmp_path / "text.npy").shape == (1, 8)
    assert np.load(tmp_path / "label.npy").shape == (1, 8)
