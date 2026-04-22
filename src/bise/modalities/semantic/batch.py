from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from bise.common.paths import ensure_directory, resolve_path
from bise.modalities.semantic.batch_client import build_batch_client
from bise.modalities.semantic.cache import JsonCache
from bise.modalities.semantic.embedder import build_text_embedder
from bise.modalities.semantic.evaluator import load_jsonl, save_jsonl
from bise.modalities.semantic.normalizer import (
    normalize_capability_tags,
    normalize_environment_tags,
    normalize_scene_category,
    normalize_task_complexity,
)
from bise.modalities.semantic.parser import parse_joint_response
from bise.modalities.semantic.paths import materialize_pipeline_paths
from bise.modalities.semantic.pipeline import (
    build_semantic_annotation,
    load_manifest,
    load_semantic_state,
    merge_failed_records,
    merge_raw_records,
    persist_semantic_outputs,
    save_manifest,
    slice_manifest_records,
)
from bise.modalities.semantic.prompts import build_joint_prompt
from bise.modalities.semantic.vlm_client import VLMResponse, build_openai_chat_completion_body


def build_semantic_batch_requests(config: Dict[str, Any]) -> Dict[str, Any]:
    config = materialize_pipeline_paths(config)
    _validate_batch_mode(config)
    prompt_mode = str(config.get("prompt_mode", "single_stage")).strip().lower()
    if prompt_mode != "single_stage":
        raise ValueError("Batch request generation currently supports prompt_mode='single_stage' only.")

    manifest_records = load_manifest(resolve_path(config["manifest_path"]))
    selected_records = slice_manifest_records(
        manifest_records,
        start_index=config.get("manifest_start_index", 0),
        end_index=config.get("manifest_end_index"),
    )
    provider_config = _load_json(config["provider_config_path"])
    taxonomy = _load_json(config["taxonomy_config_path"])
    raw_records, existing_annotations, _ = load_semantic_state(
        raw_response_path=resolve_path(config["raw_response_path"]),
        normalized_annotation_path=resolve_path(config["normalized_annotation_path"]),
        errors_path=resolve_path(config["errors_path"]),
    )
    existing_raw_keys = {
        (str(record.get("sample_id", "")), str(record.get("request_type", "")))
        for record in raw_records
    }
    request_lines: List[Dict[str, Any]] = []
    skipped_count = 0
    for record in selected_records:
        if config.get("skip_completed", True) and record.status == "completed" and record.sample_id in existing_annotations:
            skipped_count += 1
            continue
        if (record.sample_id, "joint") in existing_raw_keys:
            skipped_count += 1
            continue
        frames = _load_frames(record, int(config.get("frame_count", 0)))
        joint_prompt = build_joint_prompt(record, config["joint_prompt_config_path"], taxonomy=taxonomy)
        payload = {
            "frames": frames,
            "prompt": joint_prompt["user_prompt"],
            "system_prompt": joint_prompt["system_prompt"],
            "model": provider_config.get("model_name", ""),
            "thinking_type": provider_config.get("thinking_type", "enabled"),
            "max_tokens": config.get("batch_max_tokens"),
            "temperature": config.get("batch_temperature", 0.0),
        }
        body = build_openai_chat_completion_body(
            payload,
            default_model_name=str(provider_config.get("model_name", "")),
            default_thinking_type=str(provider_config.get("thinking_type", "enabled")),
        )
        request_lines.append(
            {
                "custom_id": record.sample_id,
                "method": "POST",
                "url": str(config.get("batch_endpoint", "/v4/chat/completions")),
                "body": body,
            }
        )

    requests_dir = ensure_directory(config["batch_requests_dir"])
    manifest_path = resolve_path(config["batch_request_manifest_path"])
    shards = _write_batch_request_shards(
        request_lines=request_lines,
        output_dir=requests_dir,
        max_requests=int(config.get("batch_max_requests_per_file", 50000)),
        max_bytes=int(config.get("batch_max_file_bytes", 100 * 1024 * 1024)),
    )
    manifest_payload = {
        "prompt_mode": prompt_mode,
        "request_count": len(request_lines),
        "skipped_count": skipped_count,
        "files": shards,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "request_count": len(request_lines),
        "skipped_count": skipped_count,
        "file_count": len(shards),
        "request_manifest_path": str(manifest_path),
    }


def submit_semantic_batch_jobs(config: Dict[str, Any]) -> Dict[str, Any]:
    config = materialize_pipeline_paths(config)
    _validate_batch_mode(config)
    provider_config = _load_json(config["provider_config_path"])
    client = build_batch_client(provider_config)
    request_manifest = _load_batch_request_manifest(config["batch_request_manifest_path"])
    jobs_path = resolve_path(config["batch_jobs_path"])
    existing_jobs = _load_json_if_exists(jobs_path).get("jobs", [])
    jobs_by_request_file = {
        (str(job.get("request_file_path", "")), str(job.get("request_file_sha256", ""))): job
        for job in existing_jobs
    }
    submitted_count = 0

    for request_file in request_manifest.get("files", []):
        request_file_path = str(request_file["request_file_path"])
        request_file_sha256 = str(request_file.get("request_file_sha256", ""))
        existing_job = jobs_by_request_file.get((request_file_path, request_file_sha256))
        if existing_job is not None and existing_job.get("batch_id"):
            continue
        uploaded = client.upload_batch_file(request_file_path, purpose="batch")
        batch_job = client.create_batch(
            input_file_id=uploaded.id,
            endpoint=str(config.get("batch_endpoint", "/v4/chat/completions")),
            auto_delete_input_file=bool(config.get("batch_auto_delete_input_file", True)),
            metadata={
                "description": str(config.get("batch_description", "semantic annotation batch")),
                "project": str(config.get("batch_project", "semantic_annotation")),
                "request_file": Path(request_file_path).name,
            },
        )
        jobs_by_request_file[(request_file_path, request_file_sha256)] = {
            "request_file_path": request_file_path,
            "request_file_sha256": request_file_sha256,
            "request_count": int(request_file.get("request_count", 0)),
            "batch_id": batch_job.id,
            "input_file_id": uploaded.id,
            "status": str(batch_job.payload.get("status", "")),
            "output_file_id": batch_job.payload.get("output_file_id"),
            "error_file_id": batch_job.payload.get("error_file_id"),
        }
        submitted_count += 1

    jobs_payload = {"jobs": list(jobs_by_request_file.values())}
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(json.dumps(jobs_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "submitted_count": submitted_count,
        "jobs_path": str(jobs_path),
        "total_jobs": len(jobs_payload["jobs"]),
    }


def sync_semantic_batch_jobs(config: Dict[str, Any], download_completed: bool = True) -> Dict[str, Any]:
    config = materialize_pipeline_paths(config)
    _validate_batch_mode(config)
    provider_config = _load_json(config["provider_config_path"])
    client = build_batch_client(provider_config)
    jobs_path = resolve_path(config["batch_jobs_path"])
    jobs_payload = _load_json_if_exists(jobs_path)
    jobs = jobs_payload.get("jobs", [])
    download_dir = ensure_directory(config["batch_download_dir"])
    updated = 0
    completed = 0

    for job in jobs:
        batch_id = str(job.get("batch_id", ""))
        if not batch_id:
            continue
        status_payload = client.retrieve_batch(batch_id).payload
        job["status"] = str(status_payload.get("status", ""))
        job["output_file_id"] = status_payload.get("output_file_id")
        job["error_file_id"] = status_payload.get("error_file_id")
        updated += 1
        if job["status"] == "completed":
            completed += 1
            if download_completed:
                if job.get("output_file_id"):
                    output_path = download_dir / f"{batch_id}_output.jsonl"
                    output_path.write_bytes(client.download_file(str(job["output_file_id"])))
                    job["output_file_path"] = str(output_path)
                if job.get("error_file_id"):
                    error_path = download_dir / f"{batch_id}_error.jsonl"
                    error_path.write_bytes(client.download_file(str(job["error_file_id"])))
                    job["error_file_path"] = str(error_path)

    jobs_path.write_text(json.dumps({"jobs": jobs}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "updated_jobs": updated,
        "completed_jobs": completed,
        "jobs_path": str(jobs_path),
    }


def ingest_semantic_batch_results(config: Dict[str, Any]) -> Dict[str, Any]:
    config = materialize_pipeline_paths(config)
    _validate_batch_mode(config)
    prompt_mode = str(config.get("prompt_mode", "single_stage")).strip().lower()
    if prompt_mode != "single_stage":
        raise ValueError("Batch result ingestion currently supports prompt_mode='single_stage' only.")

    manifest_path = resolve_path(config["manifest_path"])
    raw_response_path = resolve_path(config["raw_response_path"])
    normalized_annotation_path = resolve_path(config["normalized_annotation_path"])
    errors_path = resolve_path(config["errors_path"])
    sample_ids_path = resolve_path(config["sample_ids_path"])
    text_embedding_path = resolve_path(config["text_embedding_path"])
    label_embedding_path = resolve_path(config["label_embedding_path"])
    feature_store_path = resolve_path(config["feature_store_path"])
    jobs_payload = _load_json_if_exists(resolve_path(config["batch_jobs_path"]))
    manifest_records = load_manifest(manifest_path)
    manifest_map = {record.sample_id: record for record in manifest_records}
    taxonomy = _load_json(config["taxonomy_config_path"])
    provider_config = _load_json(config["provider_config_path"])
    embedder = build_text_embedder(_load_json(config["embedder_config_path"]))
    cache = JsonCache(str(resolve_path(config["cache_path"])))
    raw_records, existing_annotations, failed_records = load_semantic_state(
        raw_response_path=raw_response_path,
        normalized_annotation_path=normalized_annotation_path,
        errors_path=errors_path,
    )

    ingested_count = 0
    for job in jobs_payload.get("jobs", []):
        if str(job.get("status", "")) != "completed":
            continue
        output_file_path = str(job.get("output_file_path", ""))
        if output_file_path:
            batch_raw_entries, batch_annotations, batch_failed_records = _ingest_output_file(
                output_file_path=output_file_path,
                manifest_map=manifest_map,
                taxonomy=taxonomy,
                provider_model_name=str(provider_config.get("model_name", "")),
                embedder=embedder,
                cache=cache,
            )
            raw_records = merge_raw_records(raw_records, batch_raw_entries)
            existing_annotations.update(batch_annotations)
            failed_records = merge_failed_records(failed_records, batch_failed_records)
            ingested_count += len(batch_annotations)
        error_file_path = str(job.get("error_file_path", ""))
        if error_file_path:
            failed_records = merge_failed_records(
                failed_records,
                _load_batch_error_file(error_file_path),
            )

    annotations = sorted(existing_annotations.values(), key=lambda item: (item.task_id, item.scene_id, item.video_role, item.sample_id))
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
        "ingested_count": ingested_count,
        "annotation_count": len(annotations),
        "failed_count": len(failed_records),
    }


def _ingest_output_file(
    output_file_path: str | Path,
    manifest_map: Dict[str, Any],
    taxonomy: Dict[str, Any],
    provider_model_name: str,
    embedder: Any,
    cache: JsonCache,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    raw_entries: List[Dict[str, Any]] = []
    annotations: Dict[str, Any] = {}
    failed_records: List[Dict[str, Any]] = []
    for row in load_jsonl(output_file_path):
        sample_id = str(row.get("custom_id", "")).strip()
        record = manifest_map.get(sample_id)
        if record is None:
            failed_records.append({"sample_id": sample_id, "error": "Missing manifest record for batch result."})
            continue
        status_code = int(row.get("response", {}).get("status_code", 0) or 0)
        if status_code != 200:
            record.status = "failed"
            failed_records.append({"sample_id": sample_id, "error": f"Batch response status_code={status_code}"})
            continue
        try:
            content = _extract_batch_result_content(row)
            parsed = parse_joint_response(VLMResponse(content=content, metadata={"source": "batch"}))
            normalized_tags = normalize_capability_tags(parsed.capability_tags, taxonomy)
            task_complexity = normalize_task_complexity(parsed.task_complexity, taxonomy)
            environment_tags = normalize_environment_tags(parsed.environment_tags, taxonomy)
            scene_category = normalize_scene_category(parsed.scene_category, taxonomy)
            cache_key = JsonCache.build_cache_key(
                record.sample_id,
                record.video_path,
                record.paired_video_path,
                record.video_role,
                record.cam_id,
                provider_model_name,
                record.joint_prompt_version,
                record.taxonomy_version,
                "joint",
            )
            cache.set(cache_key, {"content": content, "metadata": row.get("response", {})})
            raw_entries.append(
                {
                    "sample_id": record.sample_id,
                    "request_type": "joint",
                    "cache_key": cache_key,
                    "content": content,
                    "metadata": row.get("response", {}),
                }
            )
            annotations[record.sample_id] = build_semantic_annotation(
                record=record,
                prompt_mode="single_stage",
                task_description=parsed.task_description,
                capability_tags=normalized_tags,
                task_complexity=task_complexity,
                environment_tags=environment_tags,
                scene_category=scene_category,
                embedder=embedder,
                joint_prompt_version=record.joint_prompt_version,
                cache_keys=[cache_key],
            )
            record.status = "completed"
        except Exception as exc:  # noqa: BLE001
            record.status = "failed"
            failed_records.append({"sample_id": sample_id, "error": str(exc)})
    return raw_entries, annotations, failed_records


def _load_batch_error_file(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in load_jsonl(path):
        sample_id = str(row.get("custom_id", "")).strip()
        message = json.dumps(row, ensure_ascii=False)
        records.append({"sample_id": sample_id, "error": message})
    return records


def _extract_batch_result_content(row: Dict[str, Any]) -> str:
    try:
        return str(row["response"]["body"]["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected batch result payload: {row}") from exc


def _write_batch_request_shards(
    request_lines: Iterable[Dict[str, Any]],
    output_dir: Path,
    max_requests: int,
    max_bytes: int,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    current_sample_ids: List[str] = []
    current_bytes = 0
    shard_index = 0

    def flush():
        nonlocal current_lines, current_sample_ids, current_bytes, shard_index
        if not current_lines:
            return
        shard_index += 1
        path = output_dir / f"semantic_batch_requests_{shard_index:04d}.jsonl"
        content = "".join(current_lines)
        path.write_text(content, encoding="utf-8")
        shards.append(
            {
                "request_file_path": str(path),
                "request_file_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "request_count": len(current_lines),
                "byte_count": current_bytes,
                "sample_ids": list(current_sample_ids),
            }
        )
        current_lines = []
        current_sample_ids = []
        current_bytes = 0

    for request_line in request_lines:
        line = json.dumps(request_line, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if current_lines and (len(current_lines) >= max_requests or current_bytes + line_bytes > max_bytes):
            flush()
        if line_bytes > max_bytes:
            raise ValueError(f"Single batch request exceeds max file size: {request_line['custom_id']}")
        current_lines.append(line)
        current_sample_ids.append(str(request_line["custom_id"]))
        current_bytes += line_bytes

    flush()
    return shards


def _validate_batch_mode(config: Dict[str, Any]) -> None:
    execution_mode = str(config.get("execution_mode", "batch")).strip().lower()
    if execution_mode != "batch":
        raise ValueError("Batch tools require execution_mode='batch'.")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _load_json_if_exists(path: str | Path) -> Dict[str, Any]:
    candidate = resolve_path(path)
    if not candidate.exists():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def _load_batch_request_manifest(path: str | Path) -> Dict[str, Any]:
    candidate = resolve_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Batch request manifest not found: {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def _load_frames(record, frame_count: int):
    if frame_count <= 0:
        return []
    from bise.data.rh20t.scanner import sample_video_frames

    return sample_video_frames(record.video_path, num_frames=frame_count) or []
