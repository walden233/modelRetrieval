import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.modalities.semantic.batch import sync_semantic_batch_jobs
from bise.modalities.semantic.paths import materialize_pipeline_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Poll semantic batch jobs and download completed results.")
    parser.add_argument("--config", required=True, help="Path to semantic pipeline JSON config.")
    parser.add_argument("--no-download", action="store_true", help="Do not download completed output/error files.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = materialize_pipeline_paths(load_json_config(args.config))
    summary = sync_semantic_batch_jobs(config, download_completed=not args.no_download)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
