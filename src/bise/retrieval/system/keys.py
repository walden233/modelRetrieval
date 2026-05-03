from __future__ import annotations


def normalize_task_id(task_id: str) -> str:
    return str(task_id).strip()


def normalize_scene_id(task_id: str, scene_id: str) -> str:
    # RH20T 的 scene_1 会跨 task 重复，统一转成 task_0001/scene_1 防止正样本碰撞。
    task_id = normalize_task_id(task_id)
    scene_id = str(scene_id).strip()
    if not scene_id:
        return task_id
    if scene_id.startswith(f"{task_id}/"):
        return scene_id
    return f"{task_id}/{scene_id}"


def normalize_camera_id(camera_id: str | None) -> str | None:
    # 不同模块有的带 cam_ 前缀、有的不带；检索库内部统一不带前缀。
    if camera_id is None:
        return None
    text = str(camera_id).strip()
    if not text:
        return None
    return text.removeprefix("cam_")


def make_entity_key(dataset_name: str, cfg: str, task_id: str, scene_id: str) -> str:
    # entity_key 是 scene 级主键，带 cfg 是为了避免 cfg2/cfg3 同名 task/scene 混淆。
    task = normalize_task_id(task_id)
    scene = normalize_scene_id(task, scene_id)
    return f"{str(dataset_name).strip().lower()}::{str(cfg).strip()}::{task}::{scene}"


def scene_name_from_scene_id(scene_id: str) -> str:
    return str(scene_id).rstrip("/").split("/")[-1]
