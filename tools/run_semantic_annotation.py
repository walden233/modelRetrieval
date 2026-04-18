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
    return parser.parse_args()


def main():
    args = parse_args()
    config = materialize_pipeline_paths(load_json_config(args.config))
    runtime_config = merge_overrides(config, {"manifest_path": args.manifest})
    summary = run_semantic_annotation_pipeline(runtime_config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
