import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.pipeline import run_semantic_annotation_pipeline, slice_manifest_records
from bise.modalities.semantic.schemas import SemanticManifestRecord


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
        {
            "capability_tags": {"allowed_tags": ["transport"], "tag_aliases": {"move": "transport"}},
            "task_complexity_options": ["高", "中", "低", "unknown"],
            "environment_tags": {"allowed_tags": ["无障碍物", "动态环境", "unknown"], "tag_aliases": {}},
            "scene_category_options": ["工业", "家庭", "医疗", "室外", "unknown"],
        },
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
            "output_schema": {
                "capability_tags": ["string"],
                "task_complexity": "string",
                "environment_tags": ["string"],
                "scene_category": "string"
            },
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


def test_semantic_pipeline_supports_single_stage_prompt_mode(tmp_path: Path):
    robot_video = tmp_path / "cam_0_robot.mp4"
    human_video = tmp_path / "cam_0_human.mp4"
    robot_video.write_bytes(b"")
    human_video.write_bytes(b"")

    manifest_path = tmp_path / "manifest_joint.jsonl"
    _write_jsonl(
        manifest_path,
        [
            {
                "sample_id": "task_1_scene_1_human",
                "pair_id": "task_1_scene_1",
                "task_id": "task_1",
                "scene_id": "scene_1",
                "dataset_name": "RH20T",
                "video_role": "human",
                "video_path": str(human_video),
                "paired_video_path": str(robot_video),
                "cam_id": "cam_0",
                "prompt_mode": "single_stage",
                "joint_prompt_version": "joint_prompt_with_taxonomy_v1",
            }
        ],
    )

    provider_path = tmp_path / "provider_joint.json"
    _write_json(provider_path, {"provider_name": "stub"})
    embedder_path = tmp_path / "embedder_joint.json"
    _write_json(embedder_path, {"provider_name": "hash", "dimension": 8})
    taxonomy_path = tmp_path / "taxonomy_joint.json"
    _write_json(
        taxonomy_path,
        {
            "capability_tags": {"allowed_tags": ["transport"], "tag_aliases": {}},
            "task_complexity_options": ["高", "中", "低", "unknown"],
            "environment_tags": {"allowed_tags": ["无障碍物", "unknown"], "tag_aliases": {}},
            "scene_category_options": ["工业", "unknown"],
        },
    )
    joint_prompt_path = tmp_path / "prompt_joint.json"
    _write_json(
        joint_prompt_path,
        {
            "version": "joint_prompt_with_taxonomy_v1",
            "system_prompt": "Return JSON",
            "user_template": "Summarize and label sample {sample_id} from {video_role_phrase}.",
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
        "output_prefix": str(tmp_path / "semantic_joint"),
        "manifest_path": str(manifest_path),
        "raw_response_path": str(tmp_path / "raw_joint.jsonl"),
        "normalized_annotation_path": str(tmp_path / "annotations_joint.jsonl"),
        "sample_ids_path": str(tmp_path / "sample_ids_joint.json"),
        "text_embedding_path": str(tmp_path / "text_joint.npy"),
        "label_embedding_path": str(tmp_path / "label_joint.npy"),
        "errors_path": str(tmp_path / "errors_joint.jsonl"),
        "feature_store_path": str(tmp_path / "feature_store_joint.json"),
        "cache_path": str(tmp_path / "cache_joint.json"),
        "provider_config_path": str(provider_path),
        "embedder_config_path": str(embedder_path),
        "taxonomy_config_path": str(taxonomy_path),
        "joint_prompt_config_path": str(joint_prompt_path),
        "prompt_mode": "single_stage",
        "frame_count": 0,
        "skip_completed": True,
    }

    summary = run_semantic_annotation_pipeline(config)
    assert summary["count"] == 1
    raw_records = [json.loads(line) for line in (tmp_path / "raw_joint.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(raw_records) == 1
    assert raw_records[0]["request_type"] == "joint"
    annotations = [json.loads(line) for line in (tmp_path / "annotations_joint.jsonl").read_text(encoding="utf-8").splitlines()]
    assert annotations[0]["metadata"]["prompt_mode"] == "single_stage"
    assert annotations[0]["metadata"]["joint_prompt_version"] == "joint_prompt_with_taxonomy_v1"


def test_slice_manifest_records_honors_start_end_indices():
    records = [
        SemanticManifestRecord(
            sample_id=f"sample_{index}",
            pair_id=f"pair_{index}",
            task_id="task_1",
            scene_id=f"scene_{index}",
            dataset_name="RH20T",
            video_role="robot",
            video_path=f"/tmp/{index}.mp4",
        )
        for index in range(5)
    ]
    sliced = slice_manifest_records(records, start_index=1, end_index=3)
    assert [record.sample_id for record in sliced] == ["sample_1", "sample_2"]


def test_semantic_pipeline_rebuilds_embedding_outputs_from_all_annotations(tmp_path: Path):
    robot_video_1 = tmp_path / "cam_0_robot_1.mp4"
    human_video_1 = tmp_path / "cam_0_human_1.mp4"
    robot_video_2 = tmp_path / "cam_0_robot_2.mp4"
    human_video_2 = tmp_path / "cam_0_human_2.mp4"
    for path in [robot_video_1, human_video_1, robot_video_2, human_video_2]:
        path.write_bytes(b"")

    manifest_path = tmp_path / "manifest_incremental.jsonl"
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
                "video_path": str(robot_video_1),
                "paired_video_path": str(human_video_1),
                "cam_id": "cam_0",
            },
            {
                "sample_id": "task_1_scene_2_robot",
                "pair_id": "task_1_scene_2",
                "task_id": "task_1",
                "scene_id": "scene_2",
                "dataset_name": "RH20T",
                "video_role": "robot",
                "video_path": str(robot_video_2),
                "paired_video_path": str(human_video_2),
                "cam_id": "cam_0",
            },
        ],
    )

    provider_path = tmp_path / "provider_incremental.json"
    _write_json(provider_path, {"provider_name": "stub"})
    embedder_path = tmp_path / "embedder_incremental.json"
    _write_json(embedder_path, {"provider_name": "hash", "dimension": 8})
    taxonomy_path = tmp_path / "taxonomy_incremental.json"
    _write_json(
        taxonomy_path,
        {
            "capability_tags": {"allowed_tags": ["transport"], "tag_aliases": {}},
            "task_complexity_options": ["高", "中", "低", "unknown"],
            "environment_tags": {"allowed_tags": ["无障碍物", "unknown"], "tag_aliases": {}},
            "scene_category_options": ["工业", "unknown"],
        },
    )
    desc_prompt_path = tmp_path / "prompt_description_incremental.json"
    _write_json(
        desc_prompt_path,
        {
            "version": "description_prompt_v1",
            "system_prompt": "Return JSON",
            "user_template": "Describe sample {sample_id}",
            "output_schema": {"task_description": "string"},
        },
    )
    label_prompt_path = tmp_path / "prompt_label_incremental.json"
    _write_json(
        label_prompt_path,
        {
            "version": "label_prompt_v1",
            "system_prompt": "Return JSON",
            "user_template": "Label sample {sample_id}",
            "output_schema": {
                "capability_tags": ["string"],
                "task_complexity": "string",
                "environment_tags": ["string"],
                "scene_category": "string",
            },
        },
    )

    base_config = {
        "dataset_type": "rh20t",
        "dataset_variant": "test",
        "output_prefix": str(tmp_path / "semantic_incremental"),
        "manifest_path": str(manifest_path),
        "raw_response_path": str(tmp_path / "raw_incremental.jsonl"),
        "normalized_annotation_path": str(tmp_path / "annotations_incremental.jsonl"),
        "sample_ids_path": str(tmp_path / "sample_ids_incremental.json"),
        "text_embedding_path": str(tmp_path / "text_incremental.npy"),
        "label_embedding_path": str(tmp_path / "label_incremental.npy"),
        "errors_path": str(tmp_path / "errors_incremental.jsonl"),
        "feature_store_path": str(tmp_path / "feature_store_incremental.json"),
        "cache_path": str(tmp_path / "cache_incremental.json"),
        "provider_config_path": str(provider_path),
        "embedder_config_path": str(embedder_path),
        "taxonomy_config_path": str(taxonomy_path),
        "description_prompt_config_path": str(desc_prompt_path),
        "label_prompt_config_path": str(label_prompt_path),
        "frame_count": 0,
        "skip_completed": True,
    }

    run_semantic_annotation_pipeline({**base_config, "manifest_start_index": 0, "manifest_end_index": 1})
    run_semantic_annotation_pipeline({**base_config, "manifest_start_index": 1, "manifest_end_index": 2})

    assert np.load(tmp_path / "text_incremental.npy").shape == (2, 8)
    assert np.load(tmp_path / "label_incremental.npy").shape == (2, 8)
    assert json.loads((tmp_path / "sample_ids_incremental.json").read_text(encoding="utf-8")) == [
        "task_1_scene_1_robot",
        "task_1_scene_2_robot",
    ]
