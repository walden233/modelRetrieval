import argparse
import json

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config, merge_overrides
from bise.modalities.semantic.paths import materialize_pipeline_paths
from bise.modalities.semantic.pipeline import run_semantic_annotation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run the semantic annotation pipeline.")
    parser.add_argument("--config", required=True, help="Path to semantic pipeline JSON config.")
    parser.add_argument("--manifest", help="Override manifest path.")
    parser.add_argument("--start-index", type=int, help="Override manifest_start_index.")
    parser.add_argument("--end-index", type=int, help="Override manifest_end_index.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = materialize_pipeline_paths(load_json_config(args.config))
    runtime_config = merge_overrides(
        config,
        {
            "manifest_path": args.manifest,
            "manifest_start_index": args.start_index,
            "manifest_end_index": args.end_index,
        },
    )
    summary = run_semantic_annotation_pipeline(runtime_config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
