from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from bise.common.paths import resolve_path


def semantic_output_root(config: Dict[str, Any]) -> Path:
    output_prefix = str(config.get("output_prefix", "artifacts/semantic")).strip()
    dataset_type = str(config.get("dataset_type", "")).strip().lower()
    dataset_variant = str(config.get("dataset_variant", "")).strip().lower()
    if not dataset_type:
        raise ValueError("dataset_type must be configured.")
    if not dataset_variant:
        raise ValueError("dataset_variant must be configured.")
    return resolve_path(Path(output_prefix) / dataset_type / dataset_variant)


def materialize_pipeline_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    root = semantic_output_root(config)
    merged = dict(config)
    merged["output_root"] = str(root)
    merged.setdefault("manifest_path", str(root / "manifests" / "semantic_manifest_v1.jsonl"))
    merged.setdefault("raw_response_path", str(root / "annotations" / "raw_responses.jsonl"))
    merged.setdefault("normalized_annotation_path", str(root / "annotations" / "normalized_annotations.jsonl"))
    merged.setdefault("sample_ids_path", str(root / "embeddings" / "sample_ids_v1.json"))
    merged.setdefault("text_embedding_path", str(root / "embeddings" / "text_embedding_v1.npy"))
    merged.setdefault("label_embedding_path", str(root / "embeddings" / "label_embedding_v1.npy"))
    merged.setdefault("errors_path", str(root / "errors" / "failed_samples.jsonl"))
    merged.setdefault("feature_store_path", str(root / "feature_store" / "semantic_features.json"))
    merged.setdefault("cache_path", str(root / "cache" / "semantic_cache.json"))
    return merged
