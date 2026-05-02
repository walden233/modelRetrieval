import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from bise.modalities.semantic.evaluator import (
    evaluate_annotation_retrieval,
    extract_annotation_embeddings,
    extract_positive_keys,
    load_semantic_annotations,
    split_annotations_by_role,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate semantic retrieval from normalized semantic annotations.")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to normalized semantic annotations JSONL. If --gallery-annotations is omitted, this file is split by role.",
    )
    parser.add_argument(
        "--gallery-annotations",
        help="Optional gallery annotations JSONL path. If omitted, gallery comes from --annotations after role split.",
    )
    parser.add_argument(
        "--query-role",
        default="human",
        help="Query video_role when using a shared annotations file. Default: human",
    )
    parser.add_argument(
        "--gallery-role",
        default="robot",
        help="Gallery video_role when using a shared annotations file. Default: robot",
    )
    parser.add_argument(
        "--positive-key",
        default="pair_id",
        choices=["pair_id", "task_id", "scene_id", "sample_id"],
        help="Field used to define positives. Default: pair_id",
    )
    parser.add_argument("--exclude-self", action="store_true", help="Exclude identical sample_id matches.")
    parser.add_argument("--output-dir", help="Optional directory for metrics, cases, matrices, and metadata.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval cases to export.")
    return parser.parse_args()


def main():
    args = parse_args()
    query_annotations = load_semantic_annotations(args.annotations)
    if args.gallery_annotations:
        gallery_annotations = load_semantic_annotations(args.gallery_annotations)
        query_annotations = _filter_annotations_by_role(query_annotations, args.query_role)
        gallery_annotations = _filter_annotations_by_role(gallery_annotations, args.gallery_role)
    else:
        query_annotations, gallery_annotations = split_annotations_by_role(
            query_annotations,
            query_role=args.query_role,
            gallery_role=args.gallery_role,
        )

    mode_specs = {
        "text_only": "text",
        "label_only": "label",
        "text_plus_label": "combined",
    }
    h2r_name = f"{args.query_role.strip().lower()}_to_{args.gallery_role.strip().lower()}"
    r2h_name = f"{args.gallery_role.strip().lower()}_to_{args.query_role.strip().lower()}"
    metrics, matrices, embeddings = _evaluate_all_modes_and_levels(
        query_annotations=query_annotations,
        gallery_annotations=gallery_annotations,
        h2r_name=h2r_name,
        r2h_name=r2h_name,
        mode_specs=mode_specs,
        exclude_self=args.exclude_self,
    )
    selected_level = _level_name(args.positive_key)

    payload = {
        "query_role": args.query_role,
        "gallery_role": args.gallery_role,
        "positive_key": args.positive_key,
        "query_count": len(query_annotations),
        "gallery_count": len(gallery_annotations),
        "metrics": metrics,
    }
    for mode_name in mode_specs:
        payload[mode_name] = metrics[h2r_name][selected_level][mode_name]
    if args.output_dir:
        _save_outputs(
            output_dir=Path(args.output_dir),
            payload=payload,
            query_annotations=query_annotations,
            gallery_annotations=gallery_annotations,
            matrices=matrices,
            embeddings=embeddings,
            mode_specs=mode_specs,
            positive_key=args.positive_key,
            top_k=args.top_k,
            h2r_name=h2r_name,
            r2h_name=r2h_name,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _filter_annotations_by_role(annotations, role: str):
    normalized_role = str(role).strip().lower()
    filtered = [annotation for annotation in annotations if annotation.video_role.strip().lower() == normalized_role]
    if not filtered:
        raise ValueError(f"No annotations found for video_role={role!r}.")
    return filtered


def _evaluate_all_modes_and_levels(
    query_annotations,
    gallery_annotations,
    h2r_name: str,
    r2h_name: str,
    mode_specs: dict[str, str],
    exclude_self: bool,
):
    levels = ("pair_id", "task_id", "scene_id", "sample_id")
    metrics = {h2r_name: {}, r2h_name: {}}
    matrices = {}
    embeddings = {}
    for mode_name, mode in mode_specs.items():
        query_embeddings, _, query_sample_ids = extract_annotation_embeddings(query_annotations, mode)
        gallery_embeddings, _, gallery_sample_ids = extract_annotation_embeddings(gallery_annotations, mode)
        embeddings[f"{h2r_name}_{mode_name}_query"] = query_embeddings
        embeddings[f"{h2r_name}_{mode_name}_gallery"] = gallery_embeddings
        matrices[f"{h2r_name}_{mode_name}"] = query_embeddings @ gallery_embeddings.T
        matrices[f"{r2h_name}_{mode_name}"] = gallery_embeddings @ query_embeddings.T
        for positive_key in levels:
            level_name = _level_name(positive_key)
            metrics[h2r_name].setdefault(level_name, {})[mode_name] = evaluate_annotation_retrieval(
                query_annotations=query_annotations,
                gallery_annotations=gallery_annotations,
                mode=mode,
                positive_key=positive_key,
                exclude_self=exclude_self,
            )
            metrics[r2h_name].setdefault(level_name, {})[mode_name] = evaluate_annotation_retrieval(
                query_annotations=gallery_annotations,
                gallery_annotations=query_annotations,
                mode=mode,
                positive_key=positive_key,
                exclude_self=exclude_self,
            )
    return metrics, matrices, embeddings


def _level_name(positive_key: str) -> str:
    return {
        "pair_id": "pair",
        "task_id": "task",
        "scene_id": "scene",
        "sample_id": "sample",
    }[positive_key]


def _save_outputs(
    output_dir: Path,
    payload: dict,
    query_annotations,
    gallery_annotations,
    matrices: dict[str, np.ndarray],
    embeddings: dict[str, np.ndarray],
    mode_specs: dict[str, str],
    positive_key: str,
    top_k: int,
    h2r_name: str,
    r2h_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(payload["metrics"], indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "query_role": payload["query_role"],
                "gallery_role": payload["gallery_role"],
                "positive_key": positive_key,
                "query": [_annotation_metadata(annotation) for annotation in query_annotations],
                "gallery": [_annotation_metadata(annotation) for annotation in gallery_annotations],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(output_dir / "similarity_matrices.npz", **matrices)
    np.savez_compressed(output_dir / "semantic_embeddings.npz", **embeddings)
    cases = {
        h2r_name: {
            mode_name: _build_cases(
                matrices[f"{h2r_name}_{mode_name}"],
                query_annotations,
                gallery_annotations,
                positive_key=positive_key,
                top_k=top_k,
            )
            for mode_name in mode_specs
        },
        r2h_name: {
            mode_name: _build_cases(
                matrices[f"{r2h_name}_{mode_name}"],
                gallery_annotations,
                query_annotations,
                positive_key=positive_key,
                top_k=top_k,
            )
            for mode_name in mode_specs
        },
    }
    (output_dir / "cases.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _build_cases(similarity_matrix, query_annotations, gallery_annotations, positive_key: str, top_k: int):
    cases = []
    query_labels = extract_positive_keys(query_annotations, positive_key)
    gallery_labels = extract_positive_keys(gallery_annotations, positive_key)
    for query_index, query_annotation in enumerate(query_annotations):
        ranked_indices = np.argsort(-similarity_matrix[query_index])[:top_k]
        query_label = query_labels[query_index]
        cases.append(
            {
                "query_index": query_index,
                "query_sample_id": query_annotation.sample_id,
                "query_label": query_label,
                "query_task_id": query_annotation.task_id,
                "query_scene_id": query_annotation.scene_id,
                "query_video_role": query_annotation.video_role,
                "query_video_path": query_annotation.video_path,
                "retrieved": [
                    {
                        "index": int(candidate_index),
                        "sample_id": gallery_annotations[int(candidate_index)].sample_id,
                        "label": gallery_labels[int(candidate_index)],
                        "task_id": gallery_annotations[int(candidate_index)].task_id,
                        "scene_id": gallery_annotations[int(candidate_index)].scene_id,
                        "video_role": gallery_annotations[int(candidate_index)].video_role,
                        "video_path": gallery_annotations[int(candidate_index)].video_path,
                        "score": float(similarity_matrix[query_index, int(candidate_index)]),
                        "is_positive": bool(gallery_labels[int(candidate_index)] == query_label),
                    }
                    for candidate_index in ranked_indices
                ],
            }
        )
    return cases


def _annotation_metadata(annotation) -> dict:
    return {
        "sample_id": annotation.sample_id,
        "pair_id": annotation.pair_id,
        "task_id": annotation.task_id,
        "scene_id": annotation.scene_id,
        "dataset_name": annotation.dataset_name,
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


if __name__ == "__main__":
    main()
