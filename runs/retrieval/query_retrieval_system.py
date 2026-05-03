import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.modalities.video.batch import extract_pixel_values
from bise.modalities.video.factory import build_video_model
from bise.modalities.video.frame_sampling import sample_video_frames
from bise.retrieval.system import RetrievalQuery, load_retrieval_library, retrieve_top_k
from bise.retrieval.system.io import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Query the unified retrieval system.")
    parser.add_argument("--library-dir", required=True, help="Retrieval library directory.")
    parser.add_argument("--config", required=True, help="Retrieval system config JSON.")
    parser.add_argument("--query-id", help="Use a stored human eval query from the library.")
    parser.add_argument("--video-path", help="Raw human video path to encode.")
    parser.add_argument("--video-config", help="Video model config for raw video query.")
    parser.add_argument("--video-checkpoint", help="Video checkpoint for raw video query.")
    parser.add_argument("--video-feature", help="Path to JSON/NPY video embedding.")
    parser.add_argument("--trajectory-feature", help="Path to JSON/NPY trajectory embedding.")
    parser.add_argument("--semantic-feature", help="Path to JSON semantic embedding payload.")
    parser.add_argument("--top-k", type=int, help="Override top-k.")
    parser.add_argument("--output", help="Optional output JSON path.")
    return parser.parse_args()


def main():
    args = parse_args()
    library = load_retrieval_library(args.library_dir)
    config = load_json_config(args.config)
    top_k = args.top_k or int(config.get("top_k", 10))
    if args.query_id:
        # 调试/离线评估场景：直接复用检索库里保存的 human query 特征。
        query = _query_from_library(library, args.query_id)
    else:
        # 真实查询场景：外部输入可以只提供一个或多个已编码特征，也可以提供 raw video 现场编码。
        query = RetrievalQuery(
            query_id="adhoc_query",
            video_embedding=_load_embedding(args.video_feature) if args.video_feature else _encode_video_query(args),
            trajectory_embedding=_load_embedding(args.trajectory_feature) if args.trajectory_feature else None,
            **_load_semantic_query(args.semantic_feature),
        )
    results = [result.to_dict() for result in retrieve_top_k(library, query, config=config, top_k=top_k)]
    payload = {"query_id": query.query_id, "results": results}
    if args.output:
        save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def _query_from_library(library, query_id: str) -> RetrievalQuery:
    # 把 query_human_eval.jsonl 中的 feature_ids 展开为 RetrievalQuery，路径与在线 query 保持一致。
    for item in library.query_items:
        if item.query_id == query_id:
            features = library.item_feature_map(item)
            return RetrievalQuery(
                query_id=query_id,
                video_embedding=_tolist(features.get("video")),
                trajectory_embedding=_tolist(features.get("trajectory")),
                semantic_text_embedding=_tolist(features.get("semantic_text")),
                semantic_label_embedding=_tolist(features.get("semantic_label")),
                semantic_combined_embedding=_tolist(features.get("semantic_combined")),
                metadata=item.metadata,
            )
    raise ValueError(f"query_id not found in library eval manifest: {query_id}")


def _load_embedding(path: str | None):
    if not path:
        return None
    candidate = Path(path)
    if candidate.suffix.lower() == ".npy":
        return np.load(candidate).astype(np.float32).reshape(-1).tolist()
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("embedding", "vector", "video_embedding", "trajectory_embedding"):
            if key in payload:
                return payload[key]
    return payload


def _load_semantic_query(path: str | None) -> dict:
    # 语义 query 第一版只消费 embedding，不在检索阶段调用 VLM，避免 API 波动影响实验复现。
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "semantic_text_embedding": payload.get("semantic_text_embedding") or payload.get("text_embedding"),
        "semantic_label_embedding": payload.get("semantic_label_embedding") or payload.get("label_embedding"),
        "semantic_combined_embedding": payload.get("semantic_combined_embedding") or payload.get("combined_embedding"),
    }


def _encode_video_query(args):
    # raw video 查询只走 human 分支编码；robot 分支只用于离线 gallery 建库。
    if not args.video_path:
        return None
    if not args.video_config or not args.video_checkpoint:
        raise ValueError("--video-config and --video-checkpoint are required with --video-path.")
    config = load_json_config(args.video_config)
    processor, model = build_video_model(config["model"])
    frames = sample_video_frames(
        args.video_path,
        num_frames=config.get("dataset", {}).get("num_frames", 16),
        strategy=config.get("dataset", {}).get("sampling_strategy", "uniform"),
        seed=config.get("dataset", {}).get("seed", 42),
        stride=config.get("dataset", {}).get("sampling_stride"),
    )
    if frames is None:
        raise ValueError(f"Failed to read video: {args.video_path}")
    inputs = processor([frames], return_tensors="pt")
    pixel_values = extract_pixel_values(inputs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.video_checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        return model.encode_human(pixel_values.to(device)).cpu().numpy()[0].tolist()


def _tolist(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).tolist()


if __name__ == "__main__":
    main()
