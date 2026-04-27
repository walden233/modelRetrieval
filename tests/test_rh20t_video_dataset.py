import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.data.rh20t import video_dataset


class _Processor:
    pass


def test_rh20t_scene_ids_include_task_to_avoid_cross_task_collisions(monkeypatch):
    scenes = [
        [
            SimpleNamespace(
                scene_path="/dataset/task_0001/scene_1",
                video_pairs=[
                    (
                        "/dataset/task_0001/scene_1/cam_0_human.mp4",
                        "/dataset/task_0001/scene_1/cam_0_robot.mp4",
                    )
                ],
            )
        ],
        [
            SimpleNamespace(
                scene_path="/dataset/task_0002/scene_1",
                video_pairs=[
                    (
                        "/dataset/task_0002/scene_1/cam_0_human.mp4",
                        "/dataset/task_0002/scene_1/cam_0_robot.mp4",
                    )
                ],
            )
        ],
    ]
    monkeypatch.setattr(video_dataset, "scan_task_scenes", lambda root_dir: scenes)

    dataset = video_dataset.RH20TVideoDataset("/dataset", processor=_Processor())

    assert [sample.scene_id for sample in dataset.samples] == ["task_0001/scene_1", "task_0002/scene_1"]
    assert len(dataset.scene_to_index) == 2
