import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.modalities.semantic.evaluator import evaluate_label_predictions, load_jsonl, summarize_description_reviews
from bise.modalities.semantic.schemas import SemanticAnnotation


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate semantic labels and description reviews.")
    parser.add_argument("--annotations", required=True, help="Path to normalized semantic annotations JSONL.")
    parser.add_argument("--gold-labels", required=True, help="Path to gold label JSONL.")
    parser.add_argument("--description-reviews", help="Path to description review JSONL.")
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = [SemanticAnnotation.from_dict(record) for record in load_jsonl(args.annotations)]
    gold_labels = load_jsonl(args.gold_labels)
    payload = {"labels": evaluate_label_predictions(annotations, gold_labels)}
    if args.description_reviews:
        payload["descriptions"] = summarize_description_reviews(load_jsonl(args.description_reviews))
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
