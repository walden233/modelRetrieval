import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.common.schemas import VideoPairSample
from bise.modalities.video.factory import build_split_manifest, split_video_dataset


class _Dataset:
    def __init__(self):
        self.samples = [
            VideoPairSample(
                sample_id="s1",
                pair_id="p1",
                dataset_name="toy",
                task_id="task_1",
                scene_id="scene_a",
                camera_id="cam_0",
                human_video_path="/tmp/h1.mp4",
                robot_video_path="/tmp/r1.mp4",
            ),
            VideoPairSample(
                sample_id="s2",
                pair_id="p2",
                dataset_name="toy",
                task_id="task_1",
                scene_id="scene_b",
                camera_id="cam_1",
                human_video_path="/tmp/h2.mp4",
                robot_video_path="/tmp/r2.mp4",
            ),
            VideoPairSample(
                sample_id="s3",
                pair_id="p3",
                dataset_name="toy",
                task_id="task_2",
                scene_id="scene_c",
                camera_id="cam_2",
                human_video_path="/tmp/h3.mp4",
                robot_video_path="/tmp/r3.mp4",
            ),
        ]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def test_split_video_dataset_by_scene_returns_subsets():
    splits = split_video_dataset(
        _Dataset(),
        {
            "unit": "scene",
            "seed": 7,
            "ratios": {"train": 0.34, "val": 0.33, "test": 0.33},
        },
    )
    assert set(splits.keys()) == {"train", "val", "test"}
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 3


def test_split_manifest_round_trip_preserves_sample_order(tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(
        '{"train": ["s3", "s1"], "val": ["s2"], "test": []}',
        encoding="utf-8",
    )

    splits = split_video_dataset(_Dataset(), {"manifest_path": str(manifest_path)})

    assert [splits["train"].dataset.samples[index].sample_id for index in splits["train"].indices] == ["s3", "s1"]
    assert build_split_manifest(splits) == {"train": ["s3", "s1"], "val": ["s2"], "test": []}
