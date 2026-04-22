from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from bise.common.paths import ensure_directory, resolve_path
from bise.common.schemas import EmbeddingSample
from bise.modalities.semantic.cache import JsonCache
from bise.modalities.semantic.embedder import build_text_embedder
from bise.modalities.semantic.evaluator import load_jsonl, save_jsonl
from bise.modalities.semantic.normalizer import (
    build_label_canonical_text,
    normalize_capability_tags,
    normalize_environment_tags,
    normalize_scene_category,
    normalize_task_complexity,
    validate_annotation,
)
from bise.modalities.semantic.paths import materialize_pipeline_paths
from bise.modalities.semantic.parser import parse_description_response, parse_joint_response, parse_label_response
from bise.modalities.semantic.prompts import build_description_prompt, build_joint_prompt, build_label_prompt
from bise.modalities.semantic.sampler import extract_camera_id
from bise.modalities.semantic.schemas import SemanticAnnotation, SemanticManifestRecord
from bise.modalities.semantic.vlm_client import VLMResponse, build_vlm_client
from bise.retrieval.extractor import build_embedding_sample
from bise.retrieval.feature_store import FeatureStore


def load_manifest(path: str | Path) -> List[SemanticManifestRecord]:
    return [SemanticManifestRecord.from_dict(record) for record in load_jsonl(path)]


def save_manifest(path: str | Path, records: Iterable[SemanticManifestRecord]) -> None:
    save_jsonl(path, [record.to_dict() for record in records])


def slice_manifest_records(
    records: List[SemanticManifestRecord],
    start_index: int | None = 0,
    end_index: int | None = None,
) -> List[SemanticManifestRecord]:
    start = 0 if start_index is None else int(start_index)
    end = None if end_index is None or end_index==-1 else int(end_index)
    if start < 0:
        raise ValueError("manifest_start_index must be >= 0.")
    if end is not None and end < start:
        raise ValueError("manifest_end_index must be >= manifest_start_index.")
    return records[start:end]


def run_semantic_annotation_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    config = materialize_pipeline_paths(config)
    execution_mode = _normalize_execution_mode(config.get("execution_mode", "sync"))
    if execution_mode != "sync":
        raise ValueError("run_semantic_annotation_pipeline only supports execution_mode='sync'. Use batch tools for batch mode.")
    manifest_path = resolve_path(config["manifest_path"])
    raw_response_path = resolve_path(config["raw_response_path"])
    normalized_annotation_path = resolve_path(config["normalized_annotation_path"])
    sample_ids_path = resolve_path(config["sample_ids_path"])
    text_embedding_path = resolve_path(config["text_embedding_path"])
    label_embedding_path = resolve_path(config["label_embedding_path"])
    errors_path = resolve_path(config.get("errors_path", "artifacts/semantic/errors/failed_samples.jsonl"))
    feature_store_path = resolve_path(config.get("feature_store_path", "artifacts/semantic/feature_store/semantic_features.json"))

    ensure_directory(raw_response_path.parent)
    ensure_directory(normalized_annotation_path.parent)
    ensure_directory(sample_ids_path.parent)
    ensure_directory(errors_path.parent)
    ensure_directory(feature_store_path.parent)

    manifest_records = load_manifest(manifest_path)
    selected_manifest_records = slice_manifest_records(
        manifest_records,
        start_index=config.get("manifest_start_index", 0),
        end_index=config.get("manifest_end_index"),
    )
    taxonomy = _load_json(config["taxonomy_config_path"])
    prompt_mode = _normalize_prompt_mode(config.get("prompt_mode", "two_stage"))
    description_prompt_config_path = config.get("description_prompt_config_path", "")
    label_prompt_config_path = config.get("label_prompt_config_path", "")
    joint_prompt_config_path = config.get("joint_prompt_config_path", "")
    provider_config = _load_json(config["provider_config_path"])
    embedder_config = _load_json(config["embedder_config_path"])

    vlm_client = build_vlm_client(provider_config)
    embedder = build_text_embedder(embedder_config)
    cache = JsonCache(str(resolve_path(config["cache_path"])))

    raw_records, existing_annotations, failed_records = load_semantic_state(
        raw_response_path=raw_response_path,
        normalized_annotation_path=normalized_annotation_path,
        errors_path=errors_path,
    )
    for record in selected_manifest_records:
        if config.get("skip_completed", True) and record.status == "completed" and record.sample_id in existing_annotations:
            continue
        else:
            try:
                annotation, raw_entries = _process_manifest_record(
                    record=record,
                    cache=cache,
                    taxonomy=taxonomy,
                    prompt_mode=prompt_mode,
                    description_prompt_config_path=description_prompt_config_path,
                    label_prompt_config_path=label_prompt_config_path,
                    joint_prompt_config_path=joint_prompt_config_path,
                    vlm_client=vlm_client,
                    embedder=embedder,
                    provider_model_name=provider_config.get("model_name", ""),
                    frame_count=int(config.get("frame_count", 0)),
                )
                existing_annotations[record.sample_id] = annotation
                raw_records = merge_raw_records(raw_records, raw_entries)
                record.status = "completed"
            except Exception as exc:  # noqa: BLE001
                record.status = "failed"
                failed_records = merge_failed_records(
                    failed_records,
                    [{"sample_id": record.sample_id, "error": str(exc)}],
                )
                continue

    annotations = _sort_annotations(existing_annotations.values())
    persist_semantic_outputs(
        manifest_path=manifest_path,
        manifest_records=manifest_records,
        raw_response_path=raw_response_path,
        raw_records=raw_records,
        normalized_annotation_path=normalized_annotation_path,
        annotations=annotations,
        errors_path=errors_path,
        failed_records=failed_records,
        sample_ids_path=sample_ids_path,
        text_embedding_path=text_embedding_path,
        label_embedding_path=label_embedding_path,
        feature_store_path=feature_store_path,
    )
    return {
        "count": len(annotations),
        "selected_count": len(selected_manifest_records),
        "failed": len([record for record in selected_manifest_records if record.status == "failed"]),
        "feature_store_path": str(feature_store_path),
    }


def _process_manifest_record(
    record: SemanticManifestRecord,
    cache: JsonCache,
    taxonomy: Dict[str, Any],
    prompt_mode: str,
    description_prompt_config_path: str,
    label_prompt_config_path: str,
    joint_prompt_config_path: str,
    vlm_client: Any,
    embedder: Any,
    provider_model_name: str,
    frame_count: int = 0,
) -> Tuple[SemanticAnnotation, List[Dict[str, Any]]]:
    frames = _load_frames(record, frame_count)
    raw_entries: List[Dict[str, Any]]
    description_prompt_version = ""
    label_prompt_version = ""
    joint_prompt_version = ""

    if prompt_mode == "single_stage":
        if not joint_prompt_config_path:
            raise ValueError("joint_prompt_config_path is required when prompt_mode=single_stage.")
        joint_prompt = build_joint_prompt(record, joint_prompt_config_path, taxonomy=taxonomy)
        joint_response, joint_cache_key = _invoke_cached_vlm(
            cache=cache,
            client=vlm_client,
            request_type="joint",
            sample_id=record.sample_id,
            video_path=record.video_path,
            paired_video_path=record.paired_video_path,
            video_role=record.video_role,
            cam_id=record.cam_id,
            model_name=provider_model_name,
            prompt_version=joint_prompt["version"],
            taxonomy_version=record.taxonomy_version,
            frames=frames,
            prompt=joint_prompt,
        )
        parsed_semantics = parse_joint_response(joint_response)
        task_description = parsed_semantics.task_description
        parsed_labels = parsed_semantics
        joint_prompt_version = joint_prompt["version"]
        raw_entries = [
            {
                "sample_id": record.sample_id,
                "request_type": "joint",
                "cache_key": joint_cache_key,
                "content": joint_response.content,
                "metadata": joint_response.metadata,
            }
        ]
    else:
        if not description_prompt_config_path:
            raise ValueError("description_prompt_config_path is required when prompt_mode=two_stage.")
        if not label_prompt_config_path:
            raise ValueError("label_prompt_config_path is required when prompt_mode=two_stage.")
        description_prompt = build_description_prompt(record, description_prompt_config_path)
        label_prompt = build_label_prompt(record, label_prompt_config_path, taxonomy=taxonomy)

        description_response, description_cache_key = _invoke_cached_vlm(
            cache=cache,
            client=vlm_client,
            request_type="description",
            sample_id=record.sample_id,
            video_path=record.video_path,
            paired_video_path=record.paired_video_path,
            video_role=record.video_role,
            cam_id=record.cam_id,
            model_name=provider_model_name,
            prompt_version=description_prompt["version"],
            taxonomy_version=record.taxonomy_version,
            frames=frames,
            prompt=description_prompt,
        )
        task_description = parse_description_response(description_response)

        label_response, label_cache_key = _invoke_cached_vlm(
            cache=cache,
            client=vlm_client,
            request_type="labels",
            sample_id=record.sample_id,
            video_path=record.video_path,
            paired_video_path=record.paired_video_path,
            video_role=record.video_role,
            cam_id=record.cam_id,
            model_name=provider_model_name,
            prompt_version=label_prompt["version"],
            taxonomy_version=record.taxonomy_version,
            frames=frames,
            prompt=label_prompt,
        )
        parsed_labels = parse_label_response(label_response)
        description_prompt_version = description_prompt["version"]
        label_prompt_version = label_prompt["version"]
        raw_entries = [
            {
                "sample_id": record.sample_id,
                "request_type": "description",
                "cache_key": description_cache_key,
                "content": description_response.content,
                "metadata": description_response.metadata,
            },
            {
                "sample_id": record.sample_id,
                "request_type": "labels",
                "cache_key": label_cache_key,
                "content": label_response.content,
                "metadata": label_response.metadata,
            },
        ]

    normalized_tags = normalize_capability_tags(parsed_labels.capability_tags, taxonomy)
    task_complexity = normalize_task_complexity(parsed_labels.task_complexity, taxonomy)
    environment_tags = normalize_environment_tags(parsed_labels.environment_tags, taxonomy)
    scene_category = normalize_scene_category(parsed_labels.scene_category, taxonomy)
    annotation = build_semantic_annotation(
        record=record,
        prompt_mode=prompt_mode,
        task_description=task_description,
        capability_tags=normalized_tags,
        task_complexity=task_complexity,
        environment_tags=environment_tags,
        scene_category=scene_category,
        embedder=embedder,
        description_prompt_version=description_prompt_version,
        label_prompt_version=label_prompt_version,
        joint_prompt_version=joint_prompt_version,
        cache_keys=[entry["cache_key"] for entry in raw_entries],
    )
    return annotation, raw_entries


def _invoke_cached_vlm(
    cache: JsonCache,
    client: Any,
    request_type: str,
    sample_id: str,
    video_path: str,
    paired_video_path: str,
    video_role: str,
    cam_id: str,
    model_name: str,
    prompt_version: str,
    taxonomy_version: str,
    frames: List[Any],
    prompt: Dict[str, str],
):
    cache_key = JsonCache.build_cache_key(
        sample_id,
        video_path,
        paired_video_path,
        video_role,
        cam_id,
        model_name,
        prompt_version,
        taxonomy_version,
        request_type,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return VLMResponse(content=cached["content"], metadata=cached.get("metadata", {})), cache_key
    payload = {
        "video_path": video_path,
        "paired_video_path": paired_video_path,
        "video_role": video_role,
        "frames": frames,
        "prompt": prompt["user_prompt"],
        "system_prompt": prompt["system_prompt"],
        "model": model_name,
    }
    if request_type == "description":
        response = client.annotate_description(payload)
    elif request_type == "labels":
        response = client.annotate_labels(payload)
    elif request_type == "joint":
        response = client.annotate_semantics(payload)
    else:
        raise ValueError(f"Unsupported semantic request_type: {request_type}")
    cache.set(cache_key, {"content": response.content, "metadata": response.metadata})
    return response, cache_key


def _load_frames(record: SemanticManifestRecord, frame_count: int) -> List[Any]:
    if frame_count <= 0:
        return []
    from bise.data.rh20t.scanner import sample_video_frames

    return sample_video_frames(record.video_path, num_frames=frame_count) or []


def _build_feature_store_samples(annotations: Iterable[SemanticAnnotation]) -> List[EmbeddingSample]:
    samples: List[EmbeddingSample] = []
    for annotation in annotations:
        samples.append(
            build_embedding_sample(
                sample_id=annotation.sample_id,
                task_id=annotation.task_id,
                scene_id=annotation.scene_id,
                embeddings={
                    "text_embedding": annotation.metadata["text_embedding"],
                    "label_embedding": annotation.metadata["label_embedding"],
                },
                metadata={
                    "semantic": {
                        "pair_id": annotation.pair_id,
                        "video_role": annotation.video_role,
                        "video_path": annotation.video_path,
                        "paired_video_path": annotation.paired_video_path,
                        "cam_id": annotation.cam_id,
                        "task_description": annotation.task_description,
                        "capability_tags": annotation.capability_tags,
                        "task_complexity": annotation.task_complexity,
                        "environment_tags": annotation.environment_tags,
                        "scene_category": annotation.scene_category,
                    }
                },
            )
        )
    return samples


def build_semantic_annotation(
    record: SemanticManifestRecord,
    prompt_mode: str,
    task_description: str,
    capability_tags: List[str],
    task_complexity: str,
    environment_tags: List[str],
    scene_category: str,
    embedder: Any,
    description_prompt_version: str = "",
    label_prompt_version: str = "",
    joint_prompt_version: str = "",
    cache_keys: List[str] | None = None,
) -> SemanticAnnotation:
    validate_annotation(task_description, capability_tags, task_complexity, environment_tags, scene_category)
    label_canonical_text = build_label_canonical_text(
        capability_tags,
        task_complexity,
        environment_tags,
        scene_category,
    )
    text_embedding = embedder.encode_texts([task_description])[0].astype(np.float32).tolist()
    label_embedding = embedder.encode_texts([label_canonical_text])[0].astype(np.float32).tolist()
    return SemanticAnnotation(
        sample_id=record.sample_id,
        pair_id=record.pair_id,
        task_id=record.task_id,
        scene_id=record.scene_id,
        dataset_name=record.dataset_name,
        video_role=record.video_role,
        video_path=record.video_path,
        paired_video_path=record.paired_video_path,
        cam_id=record.cam_id or extract_camera_id(
            record.video_path,
            "_robot.mp4" if record.video_role == "robot" else "_human.mp4",
        ),
        task_description=task_description,
        capability_tags=capability_tags,
        task_complexity=task_complexity,
        environment_tags=environment_tags,
        scene_category=scene_category,
        label_canonical_text=label_canonical_text,
        metadata={
            "prompt_mode": prompt_mode,
            "description_prompt_version": description_prompt_version,
            "label_prompt_version": label_prompt_version,
            "joint_prompt_version": joint_prompt_version,
            "taxonomy_version": record.taxonomy_version,
            "video_role": record.video_role,
            "text_embedding": text_embedding,
            "label_embedding": label_embedding,
            "cache_keys": list(cache_keys or []),
        },
    )


def load_semantic_state(
    raw_response_path: str | Path,
    normalized_annotation_path: str | Path,
    errors_path: str | Path,
) -> tuple[List[Dict[str, Any]], Dict[str, SemanticAnnotation], List[Dict[str, Any]]]:
    raw_records = load_jsonl(raw_response_path)
    annotations = [SemanticAnnotation.from_dict(record) for record in load_jsonl(normalized_annotation_path)]
    failed_records = load_jsonl(errors_path)
    return raw_records, {annotation.sample_id: annotation for annotation in annotations}, failed_records


def persist_semantic_outputs(
    manifest_path: str | Path,
    manifest_records: Iterable[SemanticManifestRecord],
    raw_response_path: str | Path,
    raw_records: List[Dict[str, Any]],
    normalized_annotation_path: str | Path,
    annotations: List[SemanticAnnotation],
    errors_path: str | Path,
    failed_records: List[Dict[str, Any]],
    sample_ids_path: str | Path,
    text_embedding_path: str | Path,
    label_embedding_path: str | Path,
    feature_store_path: str | Path,
) -> None:
    ensure_directory(Path(raw_response_path).parent)
    ensure_directory(Path(normalized_annotation_path).parent)
    ensure_directory(Path(errors_path).parent)
    ensure_directory(Path(sample_ids_path).parent)
    ensure_directory(Path(feature_store_path).parent)
    save_manifest(manifest_path, manifest_records)
    save_jsonl(raw_response_path, raw_records)
    save_jsonl(normalized_annotation_path, [annotation.to_dict() for annotation in annotations])
    save_jsonl(errors_path, failed_records)

    sample_ids = [annotation.sample_id for annotation in annotations]
    text_vectors = [annotation.metadata["text_embedding"] for annotation in annotations]
    label_vectors = [annotation.metadata["label_embedding"] for annotation in annotations]
    np.save(text_embedding_path, np.asarray(text_vectors, dtype=np.float32))
    np.save(label_embedding_path, np.asarray(label_vectors, dtype=np.float32))
    Path(sample_ids_path).write_text(json.dumps(sample_ids, indent=2, ensure_ascii=False), encoding="utf-8")

    feature_store = FeatureStore(str(feature_store_path))
    feature_store.save(_build_feature_store_samples(annotations))


def merge_raw_records(
    existing_records: List[Dict[str, Any]],
    new_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for record in existing_records + new_records:
        key = (
            str(record.get("sample_id", "")),
            str(record.get("request_type", "")),
            str(record.get("cache_key", "")),
        )
        merged[key] = record
    return list(merged.values())


def merge_failed_records(
    existing_records: List[Dict[str, Any]],
    new_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record in existing_records + new_records:
        key = (
            str(record.get("sample_id", "")),
            str(record.get("error", "")),
        )
        merged[key] = record
    return list(merged.values())


def _sort_annotations(annotations: Iterable[SemanticAnnotation]) -> List[SemanticAnnotation]:
    return sorted(
        annotations,
        key=lambda item: (
            item.task_id,
            item.scene_id,
            item.video_role,
            item.sample_id,
        ),
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    candidate = resolve_path(path)
    return json.loads(candidate.read_text(encoding="utf-8"))


def _normalize_prompt_mode(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"two_stage", "single_stage"}:
        raise ValueError("prompt_mode must be 'two_stage' or 'single_stage'.")
    return normalized


def _normalize_execution_mode(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized not in {"sync", "batch"}:
        raise ValueError("execution_mode must be 'sync' or 'batch'.")
    return normalized
