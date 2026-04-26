import argparse
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _bootstrap import bootstrap

bootstrap()

from bise.common import load_json_config, resolve_path
from bise.data.rh20t.scanner import scan_task_scenes
from bise.modalities.semantic.evaluator import save_jsonl
from bise.modalities.semantic.paths import materialize_pipeline_paths
from bise.modalities.semantic.sampler import extract_camera_id, select_scene_camera_pair
from bise.modalities.semantic.schemas import SemanticManifestRecord


def parse_args():
    parser = argparse.ArgumentParser(description="Build a scene-level semantic manifest.")
    parser.add_argument("--config", required=True, help="Path to semantic pipeline JSON config.")
    parser.add_argument("--dataset-type", choices=["rh20t", "whirl"], help="Override dataset type from config.")
    parser.add_argument("--data-root", help="RH20T root directory or WHIRL dataset root.")
    parser.add_argument("--csv", help="WHIRL CSV manifest path.")
    parser.add_argument("--output", help="Optional manifest output override.")
    parser.add_argument("--scenes-per-task", type=int, help="Maximum number of scenes to keep per task. <=0 means all.")
    return parser.parse_args()


def build_rh20t_manifest(config, data_root: str) -> List[SemanticManifestRecord]:
    task_scenes = scan_task_scenes(data_root)
    records: List[SemanticManifestRecord] = []
    scenes_per_task = _normalize_scenes_per_task(config.get("scenes_per_task", 2))
    strategy = {
        "preferred_robot_cam_id": config.get("preferred_robot_cam_id", ""),
        "preferred_human_cam_id": config.get("preferred_human_cam_id", ""),
    }
    for task in task_scenes:
        for scene in _limit_task_scenes(task, scenes_per_task):
            human_video_path, robot_video_path = select_scene_camera_pair(scene.video_pairs, strategy)
            scene_path = Path(scene.scene_path)
            task_id = scene_path.parent.name
            scene_id = scene_path.name
            pair_id = f"{task_id}_{scene_id}"
            robot_cam_id = extract_camera_id(robot_video_path, "_robot.mp4")
            human_cam_id = extract_camera_id(human_video_path, "_human.mp4")
            common_kwargs = {
                "pair_id": pair_id,
                "task_id": task_id,
                "scene_id": scene_id,
                "description_prompt_version": str(config.get("description_prompt_version", "description_prompt_v1")),
                "label_prompt_version": str(config.get("label_prompt_version", "label_prompt_with_taxonomy_v1")),
                "joint_prompt_version": str(config.get("joint_prompt_version", "joint_prompt_with_taxonomy_v1")),
                "taxonomy_version": str(config.get("taxonomy_version", "taxonomy_v1")),
            }
            records.append(
                SemanticManifestRecord(
                    sample_id=f"{pair_id}_robot",
                    dataset_name="RH20T",
                    video_role="robot",
                    video_path=str(Path(robot_video_path).resolve()),
                    paired_video_path=str(Path(human_video_path).resolve()),
                    cam_id=robot_cam_id,
                    **common_kwargs,
                )
            )
            records.append(
                SemanticManifestRecord(
                    sample_id=f"{pair_id}_human",
                    dataset_name="RH20T",
                    video_role="human",
                    video_path=str(Path(human_video_path).resolve()),
                    paired_video_path=str(Path(robot_video_path).resolve()),
                    cam_id=human_cam_id,
                    **common_kwargs,
                )
            )
    return records


def build_whirl_manifest(config, csv_path: str) -> List[SemanticManifestRecord]:
    dataframe = pd.read_csv(csv_path)
    records: List[SemanticManifestRecord] = []
    scenes_per_task = _normalize_scenes_per_task(config.get("scenes_per_task", 2))
    scene_counts: dict[str, int] = {}
    for index, row in dataframe.iterrows():
        task_id = str(row.get("task_id", f"task_{index:04d}"))
        if scenes_per_task is not None:
            current_count = scene_counts.get(task_id, 0)
            if current_count >= scenes_per_task:
                continue
            scene_counts[task_id] = current_count + 1
        scene_id = str(row.get("scene_id", task_id))
        pair_id = f"{task_id}_{scene_id}_{index}"
        common_kwargs = {
            "pair_id": pair_id,
            "task_id": task_id,
            "scene_id": scene_id,
            "dataset_name": "WHIRL",
            "description_prompt_version": str(config.get("description_prompt_version", "description_prompt_v1")),
            "label_prompt_version": str(config.get("label_prompt_version", "label_prompt_with_taxonomy_v1")),
            "joint_prompt_version": str(config.get("joint_prompt_version", "joint_prompt_with_taxonomy_v1")),
            "taxonomy_version": str(config.get("taxonomy_version", "taxonomy_v1")),
        }
        robot_video_path = str(resolve_path(row["robot_video_path"]))
        human_video_path = str(resolve_path(row["human_video_path"]))
        records.append(
            SemanticManifestRecord(
                sample_id=f"{pair_id}_robot",
                video_role="robot",
                video_path=robot_video_path,
                paired_video_path=human_video_path,
                cam_id=str(row.get("robot_cam_id", "robot_cam")),
                **common_kwargs,
            )
        )
        records.append(
            SemanticManifestRecord(
                sample_id=f"{pair_id}_human",
                video_role="human",
                video_path=human_video_path,
                paired_video_path=robot_video_path,
                cam_id=str(row.get("human_cam_id", "human_cam")),
                **common_kwargs,
            )
        )
    return records


def _limit_task_scenes(task_scenes, scenes_per_task: int | None):
    if scenes_per_task is None:
        return task_scenes
    return list(task_scenes[:scenes_per_task])


def _normalize_scenes_per_task(value) -> int | None:
    if value is None:
        return None
    count = int(value)
    if count <= 0:
        return None
    return count


def main():
    args = parse_args()
    config = materialize_pipeline_paths(load_json_config(args.config))
    if args.scenes_per_task is not None:
        config["scenes_per_task"] = args.scenes_per_task
    dataset_type = args.dataset_type or str(config.get("dataset_type", "")).strip().lower()
    if not dataset_type:
        raise ValueError("dataset_type must be provided via --dataset-type or config.")
    data_root = args.data_root or config.get("data_root")
    csv_path = args.csv or config.get("csv_path") or data_root
    output_path = resolve_path(args.output or config["manifest_path"])
    if dataset_type == "rh20t":
        if not data_root:
            raise ValueError("data_root is required for rh20t manifests.")
        records = build_rh20t_manifest(config, str(data_root))
    else:
        if not csv_path:
            raise ValueError("csv_path is required for whirl manifests.")
        records = build_whirl_manifest(config, str(csv_path))
    save_jsonl(output_path, [record.to_dict() for record in records])
    print(
        json.dumps(
            {
                "dataset_type": dataset_type,
                "data_root": str(data_root) if data_root else "",
                "output": str(output_path),
                "count": len(records),
                "scenes_per_task": _normalize_scenes_per_task(config.get("scenes_per_task", 2)),
                "manifest_start_index": config.get("manifest_start_index", 0),
                "manifest_end_index": config.get("manifest_end_index"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
