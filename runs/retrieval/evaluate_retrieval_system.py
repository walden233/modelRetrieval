import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.retrieval.system import evaluate_retrieval_system, load_retrieval_library
from bise.retrieval.system.io import save_json, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the unified retrieval system.")
    parser.add_argument("--library-dir", required=True, help="Retrieval library directory.")
    parser.add_argument("--config", required=True, help="Retrieval system config JSON.")
    parser.add_argument("--level", default="scene", choices=["scene", "task", "mixed"])
    parser.add_argument("--require-modalities", default="", help="Comma-separated required query modalities.")
    parser.add_argument("--enabled-modalities", default="", help="Comma-separated enabled modalities override.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    library = load_retrieval_library(args.library_dir)
    config = load_json_config(args.config)
    if args.enabled_modalities:
        # enabled_modalities 用于系统消融，例如只启用 video 或 video+semantic。
        config["modalities"] = _split_csv(args.enabled_modalities)
    require_modalities = _split_csv(args.require_modalities)
    # 评估时从 human eval query 发起检索，gallery 始终是 robot，符合真实系统方向。
    result = evaluate_retrieval_system(
        library=library,
        config=config,
        level=args.level,
        require_modalities=require_modalities,
        top_k=args.top_k,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", result["metrics"])
    save_json(output_dir / "cases.json", result["cases"])
    # per_query_results.jsonl 便于后续人工筛选失败案例或做定性分析。
    save_jsonl(output_dir / "per_query_results.jsonl", result["cases"])
    save_json(
        output_dir / "run_info.json",
        {
            "library_dir": str(args.library_dir),
            "config": config,
            "level": args.level,
            "require_modalities": require_modalities,
            "query_count": result["query_count"],
            "gallery_count": result["gallery_count"],
        },
    )
    np.save(output_dir / "fused_similarity_matrix.npy", result["similarity_matrix"])
    for modality, matrix in result["modality_matrices"].items():
        np.save(output_dir / f"{modality}_similarity_matrix.npy", matrix)
    print(json.dumps({"metrics": result["metrics"], "query_count": result["query_count"], "gallery_count": result["gallery_count"]}, indent=2, ensure_ascii=False))


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
