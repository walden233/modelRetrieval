import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.modalities.semantic.batch import submit_semantic_batch_jobs
from bise.modalities.semantic.paths import materialize_pipeline_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Submit semantic batch jobs.")
    parser.add_argument("--config", required=True, help="Path to semantic pipeline JSON config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = materialize_pipeline_paths(load_json_config(args.config))
    summary = submit_semantic_batch_jobs(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
