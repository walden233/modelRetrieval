import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.sampler import select_scene_camera_pair


def test_select_scene_camera_pair_prefers_configured_camera():
    pairs = [
        ("/tmp/cam_1_human.mp4", "/tmp/cam_1_robot.mp4"),
        ("/tmp/cam_0_human.mp4", "/tmp/cam_0_robot.mp4"),
    ]
    human_path, robot_path = select_scene_camera_pair(
        pairs,
        {"preferred_human_cam_id": "cam_0", "preferred_robot_cam_id": "cam_0"},
    )
    assert human_path.endswith("cam_0_human.mp4")
    assert robot_path.endswith("cam_0_robot.mp4")


def test_select_scene_camera_pair_falls_back_to_sorted_first():
    pairs = [
        ("/tmp/cam_2_human.mp4", "/tmp/cam_2_robot.mp4"),
        ("/tmp/cam_1_human.mp4", "/tmp/cam_1_robot.mp4"),
    ]
    human_path, robot_path = select_scene_camera_pair(pairs, {})
    assert human_path.endswith("cam_1_human.mp4")
    assert robot_path.endswith("cam_1_robot.mp4")
