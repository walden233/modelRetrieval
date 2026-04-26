import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.modalities.semantic.batch import ingest_semantic_batch_results
from bise.modalities.semantic.paths import materialize_pipeline_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Ingest downloaded semantic batch results into semantic artifacts.")
    parser.add_argument("--config", required=True, help="Path to semantic pipeline JSON config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = materialize_pipeline_paths(load_json_config(args.config))
    summary = ingest_semantic_batch_results(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
