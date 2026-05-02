import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.retrieval import FaissIndex, FeatureStore


def parse_args():
    parser = argparse.ArgumentParser(description="Build a FAISS index from a feature store JSON file.")
    parser.add_argument("--features", required=True, help="Path to feature_store JSON.")
    parser.add_argument("--output", required=True, help="Output FAISS index path.")
    parser.add_argument("--field", default="text_embedding", help="Embedding field to index.")
    return parser.parse_args()


def main():
    args = parse_args()
    store = FeatureStore(args.features)
    payload = store.load()
    embeddings = [record[args.field] for record in payload if record.get(args.field)]
    if not embeddings:
        raise ValueError(f"No embeddings found for field: {args.field}")
    index = FaissIndex(len(embeddings[0]))
    index.build(embeddings)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    index.save(args.output)
    print(json.dumps({"output": args.output, "count": len(embeddings), "dimension": len(embeddings[0])}, indent=2))


if __name__ == "__main__":
    main()
