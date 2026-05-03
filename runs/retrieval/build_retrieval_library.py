import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config
from bise.data import RH20TTrajectoryDataset, collate_trajectories
from bise.modalities.trajectory import CrossModalTrajectoryModel
from bise.modalities.video import collate_video_pairs
from bise.modalities.video.factory import build_video_dataset, build_video_model
from bise.retrieval.system.io import load_json, save_json, save_jsonl
from bise.retrieval.system.keys import make_entity_key, normalize_camera_id, normalize_scene_id, scene_name_from_scene_id


def parse_args():
    parser = argparse.ArgumentParser(description="Build a unified robot-gallery retrieval library.")
    parser.add_argument("--config", help="Optional retrieval library JSON config.")
    parser.add_argument("--cfg", help="RH20T cfg name or number, for example cfg2 or 2.")
    parser.add_argument("--data-root", help="RH20T data root.")
    parser.add_argument("--output-dir", help="Output retrieval library directory.")
    parser.add_argument("--scenes-per-task", type=int, help="Scenes per task. <=0 means all.")
    parser.add_argument("--cameras-per-scene", type=int, help="Cameras per scene. <=0 means all.")
    parser.add_argument("--video-config", help="Video encoder config.")
    parser.add_argument("--video-checkpoint", help="Video encoder checkpoint.")
    parser.add_argument("--trajectory-config", help="Trajectory encoder config.")
    parser.add_argument("--trajectory-checkpoint", help="Trajectory encoder checkpoint.")
    parser.add_argument("--semantic-root", help="Semantic root, for example artifacts/semantic/rh20t.")
    parser.add_argument("--semantic-cfg", help="Semantic cfg directory, for example cfg2.")
    parser.add_argument("--include-human", action="store_true", help="Export human query features for evaluation.")
    parser.add_argument("--include-robot", action="store_true", help="Export robot gallery features.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing library metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _merge_config(args)
    output_dir = Path(config["output_dir"])
    # 检索库目录固定分层：features 存向量，manifests 存索引记录，indices 预留给 FAISS。
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "features").mkdir(exist_ok=True)
    (output_dir / "manifests").mkdir(exist_ok=True)
    (output_dir / "indices").mkdir(exist_ok=True)
    (output_dir / "eval").mkdir(exist_ok=True)

    builder = LibraryBuilder(config, output_dir)
    if config.get("video", {}).get("enabled", False):
        builder.add_video_features()
    if config.get("trajectory", {}).get("enabled", False):
        builder.add_trajectory_features()
    if config.get("semantic", {}).get("enabled", False):
        builder.add_semantic_features()
    builder.save()
    print(json.dumps({"output_dir": str(output_dir), "coverage": builder.coverage()}, indent=2, ensure_ascii=False))


class LibraryBuilder:
    # LibraryBuilder 只负责“离线建库”：把视频/轨迹模型输出和已有语义特征统一写成检索库格式。
    def __init__(self, config: dict[str, Any], output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = str(config.get("dataset_name", "rh20t")).lower()
        self.cfg = _normalize_cfg(config["cfg"])
        self.include_human = bool(config.get("domains", {}).get("include_human", True))
        self.include_robot = bool(config.get("domains", {}).get("include_robot", True))
        self.feature_records: list[dict[str, Any]] = []
        self.scenes: dict[str, dict[str, Any]] = {}
        self.arrays: dict[str, list[np.ndarray]] = defaultdict(list)

    def add_video_features(self) -> None:
        # 视频特征直接从模型 checkpoint 重新抽取，不依赖任何 <best_run>/final_test 评估产物。
        video_config_path = self.config["video"]["config"]
        checkpoint = self.config["video"]["checkpoint"]
        video_config = load_json_config(video_config_path)
        video_config.setdefault("dataset", {})["root_dir"] = self.config["data_root"]
        video_config["dataset"]["max_pairs_per_scene"] = _positive_or_none(_sampling(self.config).get("cameras_per_scene"))
        processor, model = build_video_model(video_config["model"])
        dataset = build_video_dataset(video_config["dataset"], processor=processor, is_train=False)
        dataset.samples = _filter_video_samples(dataset.samples, _sampling(self.config))
        dataloader = DataLoader(
            dataset,
            batch_size=int(self.config["video"].get("batch_size", video_config.get("training", {}).get("eval_batch_size", 4))),
            shuffle=False,
            num_workers=int(video_config.get("training", {}).get("num_workers", 0)),
            collate_fn=collate_video_pairs,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                human_values = batch["human_pixel_values"].to(device)
                robot_values = batch["robot_pixel_values"].to(device)
                if self.include_human:
                    # human 特征只用于离线评估 query；生产检索库可以通过 include_human=false 跳过。
                    human_embeddings = model.encode_human(human_values).cpu().numpy()
                    for index, embedding in enumerate(human_embeddings):
                        self._add_feature(
                            modality="video",
                            domain="human",
                            embedding=embedding,
                            task_id=batch["task_ids"][index],
                            scene_id=batch["scene_ids"][index],
                            camera_id=batch["camera_ids"][index],
                            source_path=batch["human_video_paths"][index],
                        )
                if self.include_robot:
                    # robot 特征是检索系统真正的 gallery。
                    robot_embeddings = model.encode_robot(robot_values).cpu().numpy()
                    for index, embedding in enumerate(robot_embeddings):
                        self._add_feature(
                            modality="video",
                            domain="robot",
                            embedding=embedding,
                            task_id=batch["task_ids"][index],
                            scene_id=batch["scene_ids"][index],
                            camera_id=batch["camera_ids"][index],
                            source_path=batch["robot_video_paths"][index],
                        )

    def add_trajectory_features(self) -> None:
        # 轨迹特征同样从 checkpoint 重新抽取，human pose 对应 query，robot tcp 对应 gallery。
        trajectory_config = load_json_config(self.config["trajectory"]["config"])
        trajectory_config["data_root"] = self.config["data_root"]
        dataset = RH20TTrajectoryDataset(
            root_dir=trajectory_config["data_root"],
            use_6_keypoints=trajectory_config.get("use_6_keypoints", False),
        )
        dataset.scenes = _filter_trajectory_scenes(dataset.scenes, _sampling(self.config))
        dataloader = DataLoader(
            dataset,
            batch_size=int(self.config["trajectory"].get("batch_size", trajectory_config.get("batch_size", 16))),
            shuffle=False,
            collate_fn=collate_trajectories,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CrossModalTrajectoryModel(**trajectory_config["model_params"]).to(device)
        model.load_state_dict(torch.load(self.config["trajectory"]["checkpoint"], map_location=device))
        model.eval()
        camera_limit = _positive_or_none(_sampling(self.config).get("cameras_per_scene"))
        camera_counts: dict[tuple[str, str], int] = defaultdict(int)
        with torch.no_grad():
            for batch in dataloader:
                if self.include_human:
                    human_embeddings = model.forward_human(batch["human_poses"].to(device), batch["human_mask"].to(device)).cpu().numpy()
                    for index, embedding in enumerate(human_embeddings):
                        if _skip_camera(camera_counts, "human", batch["human_scene_ids"][index], camera_limit):
                            continue
                        self._add_feature(
                            modality="trajectory",
                            domain="human",
                            embedding=embedding,
                            task_id=batch["human_task_ids"][index],
                            scene_id=batch["human_scene_ids"][index],
                            camera_id=batch["human_camera_ids"][index],
                            source_path=batch["human_scene_paths"][index],
                        )
                if self.include_robot:
                    robot_embeddings = model.forward_robot(batch["tcp_bases"].to(device), batch["tcp_mask"].to(device)).cpu().numpy()
                    for index, embedding in enumerate(robot_embeddings):
                        if _skip_camera(camera_counts, "robot", batch["robot_scene_ids"][index], camera_limit):
                            continue
                        self._add_feature(
                            modality="trajectory",
                            domain="robot",
                            embedding=embedding,
                            task_id=batch["robot_task_ids"][index],
                            scene_id=batch["robot_scene_ids"][index],
                            camera_id=batch["robot_camera_ids"][index],
                            source_path=batch["robot_scene_paths"][index],
                        )

    def add_semantic_features(self) -> None:
        # 语义特征不重新调用 VLM，直接导入 artifacts/semantic/rh20t/<cfg> 中已有的 embedding。
        semantic_config = self.config.get("semantic", {})
        cfg = semantic_config.get("cfg") or self.cfg
        feature_store_path = Path(semantic_config["root"]) / str(cfg) / "feature_store" / "semantic_features.json"
        payload = load_json(feature_store_path, default=[]) or []
        modes = set(semantic_config.get("modes") or ["text", "label", "combined"])
        for record in payload:
            metadata = dict(record.get("metadata", {}).get("semantic", {}))
            domain = str(metadata.get("video_role") or _domain_from_sample_id(record.get("sample_id", ""))).lower()
            if domain == "human" and not self.include_human:
                continue
            if domain == "robot" and not self.include_robot:
                continue
            task_id = str(record["task_id"])
            scene_id = normalize_scene_id(task_id, str(record["scene_id"]))
            camera_id = normalize_camera_id(metadata.get("cam_id"))
            source_path = str(metadata.get("video_path", ""))
            text_embedding = np.asarray(record.get("text_embedding") or [], dtype=np.float32)
            label_embedding = np.asarray(record.get("label_embedding") or [], dtype=np.float32)
            if "text" in modes and text_embedding.size:
                self._add_feature("semantic_text", domain, text_embedding, task_id, scene_id, camera_id, source_path)
            if "label" in modes and label_embedding.size:
                self._add_feature("semantic_label", domain, label_embedding, task_id, scene_id, camera_id, source_path)
            if "combined" in modes and text_embedding.size and label_embedding.size:
                # combined 与语义单模态评估保持一致：text 和 label 各自归一后相加再归一。
                combined = _normalize_vector(_normalize_vector(text_embedding) + _normalize_vector(label_embedding))
                self._add_feature("semantic_combined", domain, combined, task_id, scene_id, camera_id, source_path)

    def save(self) -> None:
        # 所有向量按 modality_domain 分组写 npy，manifest 只保存 row_index，避免 JSON 体积过大。
        array_paths = {}
        for array_name, vectors in self.arrays.items():
            path = Path("features") / f"{array_name}.npy"
            np.save(self.output_dir / path, np.asarray(vectors, dtype=np.float32))
            array_paths[array_name] = str(path)
        for record in self.feature_records:
            record["array_path"] = array_paths[_array_name(record["modality"], record["domain"])]
        manifest_dir = self.output_dir / "manifests"
        save_jsonl(manifest_dir / "feature_records.jsonl", self.feature_records)
        save_jsonl(manifest_dir / "scenes.jsonl", self.scenes.values())
        gallery_items, query_items = self._build_items()
        save_jsonl(manifest_dir / "gallery_robot.jsonl", gallery_items)
        save_jsonl(manifest_dir / "query_human_eval.jsonl", query_items)
        save_json(self.output_dir / "library_config.json", self.config)
        save_json(self.output_dir / "build_info.json", {"feature_arrays": array_paths})
        save_json(self.output_dir / "coverage.json", self.coverage(gallery_items, query_items))

    def coverage(self, gallery_items: list[dict] | None = None, query_items: list[dict] | None = None) -> dict[str, Any]:
        by_modality = defaultdict(lambda: {"human": 0, "robot": 0})
        for record in self.feature_records:
            by_modality[record["modality"]][record["domain"]] += 1
        return {
            "scene_count": len(self.scenes),
            "feature_count": len(self.feature_records),
            "gallery_count": len(gallery_items or []),
            "query_count": len(query_items or []),
            "by_modality": dict(by_modality),
        }

    def _add_feature(
        self,
        modality: str,
        domain: str,
        embedding: np.ndarray,
        task_id: str,
        scene_id: str,
        camera_id: str | None,
        source_path: str,
    ) -> None:
        # 所有模态最终都落到同一套 FeatureRecord 格式，后续查询/评估不再关心特征来源。
        scene_id = normalize_scene_id(task_id, scene_id)
        camera_id = normalize_camera_id(camera_id)
        entity_key = make_entity_key(self.dataset_name, self.cfg, task_id, scene_id)
        self.scenes.setdefault(
            entity_key,
            {
                "entity_key": entity_key,
                "dataset_name": self.dataset_name,
                "cfg": self.cfg,
                "task_id": task_id,
                "scene_id": scene_id,
                "scene_name": scene_name_from_scene_id(scene_id),
                "scene_path": str(Path(source_path).parent) if source_path else "",
                "camera_ids": [],
                "available_modalities": {},
            },
        )
        scene = self.scenes[entity_key]
        if camera_id and camera_id not in scene["camera_ids"]:
            scene["camera_ids"].append(camera_id)
        scene["available_modalities"][modality] = True
        array_name = _array_name(modality, domain)
        row_index = len(self.arrays[array_name])
        self.arrays[array_name].append(_normalize_vector(np.asarray(embedding, dtype=np.float32)))
        feature_id = "::".join([entity_key, camera_id or "scene", domain, modality])
        self.feature_records.append(
            {
                "feature_id": feature_id,
                "entity_key": entity_key,
                "domain": domain,
                "modality": modality,
                "array_path": "",
                "row_index": row_index,
                "task_id": task_id,
                "scene_id": scene_id,
                "camera_id": camera_id,
                "source_path": source_path,
                "metadata": {},
            }
        )

    def _build_items(self):
        # gallery/query item 是面向检索的候选单位；一个 item 汇总同一 domain/entity/camera 的多模态 feature_id。
        grouped: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        semantic_by_entity_domain: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for record in self.feature_records:
            if record["modality"].startswith("semantic_"):
                semantic_by_entity_domain[(record["entity_key"], record["domain"])][record["modality"]] = record["feature_id"]

        for record in self.feature_records:
            if record["modality"].startswith("semantic_"):
                continue
            key = (record["domain"], record["entity_key"], record["camera_id"])
            item = grouped.setdefault(key, _base_item(record))
            item["feature_ids"][record["modality"]] = record["feature_id"]
        for record in self.feature_records:
            if not record["modality"].startswith("semantic_"):
                continue
            key = (record["domain"], record["entity_key"], record["camera_id"])
            item = grouped.setdefault(key, _base_item(record))
            item["feature_ids"][record["modality"]] = record["feature_id"]
        for (domain, entity_key, _), item in grouped.items():
            # 语义通常是 scene 级或 selected-camera，这里按 entity+domain 共享给同 scene 的 camera 候选。
            item["feature_ids"].update(semantic_by_entity_domain.get((entity_key, domain), {}))

        gallery_items = []
        query_items = []
        for item in grouped.values():
            if item["domain"] == "robot":
                gallery = dict(item)
                gallery["gallery_id"] = gallery.pop("item_id")
                gallery_items.append(gallery)
            elif item["domain"] == "human":
                query = dict(item)
                query["query_id"] = query.pop("item_id")
                query_items.append(query)
        gallery_items.sort(key=lambda item: item["gallery_id"])
        query_items.sort(key=lambda item: item["query_id"])
        return gallery_items, query_items


def _merge_config(args) -> dict[str, Any]:
    # 配置文件是主入口，命令行只做覆盖，便于脚本化跑 cfg2/cfg3 或不同采样设置。
    config = load_json_config(args.config) if args.config else {}
    config = dict(config)
    overrides = {
        "cfg": args.cfg,
        "data_root": args.data_root,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    sampling = dict(config.get("sampling") or {})
    if args.scenes_per_task is not None:
        sampling["scenes_per_task"] = args.scenes_per_task
    if args.cameras_per_scene is not None:
        sampling["cameras_per_scene"] = args.cameras_per_scene
    config["sampling"] = sampling
    if args.video_config or args.video_checkpoint:
        video = dict(config.get("video") or {})
        video["enabled"] = True
        if args.video_config:
            video["config"] = args.video_config
        if args.video_checkpoint:
            video["checkpoint"] = args.video_checkpoint
        config["video"] = video
    if args.trajectory_config or args.trajectory_checkpoint:
        trajectory = dict(config.get("trajectory") or {})
        trajectory["enabled"] = True
        if args.trajectory_config:
            trajectory["config"] = args.trajectory_config
        if args.trajectory_checkpoint:
            trajectory["checkpoint"] = args.trajectory_checkpoint
        config["trajectory"] = trajectory
    if args.semantic_root or args.semantic_cfg:
        semantic = dict(config.get("semantic") or {})
        semantic["enabled"] = True
        if args.semantic_root:
            semantic["root"] = args.semantic_root
        if args.semantic_cfg:
            semantic["cfg"] = args.semantic_cfg
        config["semantic"] = semantic
    domains = dict(config.get("domains") or {})
    if args.include_human:
        domains["include_human"] = True
    if args.include_robot:
        domains["include_robot"] = True
    domains.setdefault("include_human", True)
    domains.setdefault("include_robot", True)
    config["domains"] = domains
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    for key in ("cfg", "data_root", "output_dir"):
        if not config.get(key):
            raise ValueError(f"Missing required config field: {key}")


def _sampling(config: dict[str, Any]) -> dict[str, int]:
    return dict(config.get("sampling") or {})


def _normalize_cfg(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text.startswith("cfg") else f"cfg{text}"


def _positive_or_none(value: Any) -> int | None:
    if value is None:
        return None
    count = int(value)
    return count if count > 0 else None


def _filter_video_samples(samples, sampling: dict[str, Any]):
    scenes_per_task = _positive_or_none(sampling.get("scenes_per_task"))
    if scenes_per_task is None:
        return list(samples)
    seen_scenes: dict[str, list[str]] = defaultdict(list)
    filtered = []
    for sample in samples:
        if sample.scene_id not in seen_scenes[sample.task_id]:
            if len(seen_scenes[sample.task_id]) >= scenes_per_task:
                continue
            seen_scenes[sample.task_id].append(sample.scene_id)
        filtered.append(sample)
    return filtered


def _filter_trajectory_scenes(scenes, sampling: dict[str, Any]):
    scenes_per_task = _positive_or_none(sampling.get("scenes_per_task"))
    if scenes_per_task is None:
        return list(scenes)
    counts: dict[str, int] = defaultdict(int)
    filtered = []
    for scene in scenes:
        task_id = Path(scene.scene_path).parent.name
        if counts[task_id] >= scenes_per_task:
            continue
        counts[task_id] += 1
        filtered.append(scene)
    return filtered


def _skip_camera(counts: dict[tuple[str, str], int], domain: str, scene_id: str, limit: int | None) -> bool:
    if limit is None:
        return False
    key = (domain, scene_id)
    counts[key] += 1
    return counts[key] > limit


def _array_name(modality: str, domain: str) -> str:
    return f"{modality}_{domain}"


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _domain_from_sample_id(sample_id: str) -> str:
    text = str(sample_id).lower()
    if text.endswith("_human") or "::human" in text:
        return "human"
    if text.endswith("_robot") or "::robot" in text:
        return "robot"
    return ""


def _base_item(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "item_id": "::".join([record["entity_key"], record.get("camera_id") or "scene", record["domain"]]),
        "entity_key": record["entity_key"],
        "domain": record["domain"],
        "task_id": record["task_id"],
        "scene_id": record["scene_id"],
        "camera_id": record.get("camera_id"),
        "feature_ids": {},
        "metadata": {"source_path": record.get("source_path", "")},
    }


if __name__ == "__main__":
    main()
