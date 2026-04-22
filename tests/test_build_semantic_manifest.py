import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _load_build_manifest_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    module_path = Path(__file__).resolve().parents[1] / "tools" / "build_semantic_manifest.py"
    spec = importlib.util.spec_from_file_location("build_semantic_manifest_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_whirl_manifest_limits_scenes_per_task(tmp_path: Path):
    module = _load_build_manifest_module()
    csv_path = tmp_path / "whirl.csv"
    robot_videos = []
    human_videos = []
    for index in range(4):
        robot_path = tmp_path / f"robot_{index}.mp4"
        human_path = tmp_path / f"human_{index}.mp4"
        robot_path.write_bytes(b"")
        human_path.write_bytes(b"")
        robot_videos.append(robot_path)
        human_videos.append(human_path)

    csv_path.write_text(
        "\n".join(
            [
                "task_id,scene_id,robot_video_path,human_video_path,robot_cam_id,human_cam_id",
                f"task_1,scene_1,{robot_videos[0]},{human_videos[0]},cam_0,cam_0",
                f"task_1,scene_2,{robot_videos[1]},{human_videos[1]},cam_0,cam_0",
                f"task_1,scene_3,{robot_videos[2]},{human_videos[2]},cam_0,cam_0",
                f"task_2,scene_1,{robot_videos[3]},{human_videos[3]},cam_0,cam_0",
            ]
        ),
        encoding="utf-8",
    )

    records = module.build_whirl_manifest({"scenes_per_task": 2}, str(csv_path))
    sample_ids = [record.sample_id for record in records]
    assert sample_ids == [
        "task_1_scene_1_0_robot",
        "task_1_scene_1_0_human",
        "task_1_scene_2_1_robot",
        "task_1_scene_2_1_human",
        "task_2_scene_1_3_robot",
        "task_2_scene_1_3_human",
    ]
    assert all("prompt_mode" not in record.to_dict() for record in records)


def test_normalize_scenes_per_task_treats_non_positive_as_all():
    module = _load_build_manifest_module()
    assert module._normalize_scenes_per_task(2) == 2
    assert module._normalize_scenes_per_task(0) is None
    assert module._normalize_scenes_per_task(-1) is None
