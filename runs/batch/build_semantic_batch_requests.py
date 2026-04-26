import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config, merge_overrides
from bise.modalities.semantic.batch import build_semantic_batch_requests
from bise.modalities.semantic.paths import materialize_pipeline_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Build batch request files for semantic annotation.")
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
    summary = build_semantic_batch_requests(runtime_config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
