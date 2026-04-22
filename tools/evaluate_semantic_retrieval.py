import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from bise.modalities.semantic.evaluator import (
    evaluate_annotation_retrieval,
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

    payload = {
        "query_role": args.query_role,
        "gallery_role": args.gallery_role,
        "positive_key": args.positive_key,
        "query_count": len(query_annotations),
        "gallery_count": len(gallery_annotations),
        "text_only": evaluate_annotation_retrieval(
            query_annotations=query_annotations,
            gallery_annotations=gallery_annotations,
            mode="text",
            positive_key=args.positive_key,
            exclude_self=args.exclude_self,
        ),
        "label_only": evaluate_annotation_retrieval(
            query_annotations=query_annotations,
            gallery_annotations=gallery_annotations,
            mode="label",
            positive_key=args.positive_key,
            exclude_self=args.exclude_self,
        ),
        "text_plus_label": evaluate_annotation_retrieval(
            query_annotations=query_annotations,
            gallery_annotations=gallery_annotations,
            mode="combined",
            positive_key=args.positive_key,
            exclude_self=args.exclude_self,
        ),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _filter_annotations_by_role(annotations, role: str):
    normalized_role = str(role).strip().lower()
    filtered = [annotation for annotation in annotations if annotation.video_role.strip().lower() == normalized_role]
    if not filtered:
        raise ValueError(f"No annotations found for video_role={role!r}.")
    return filtered


if __name__ == "__main__":
    main()
