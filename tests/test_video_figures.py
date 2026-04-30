from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.video.figures import keep_first_camera_per_scene


def test_keep_first_camera_per_scene_filters_second_camera():
    matrix = np.arange(16).reshape(4, 4)
    metadata = {
        "sample_ids": ["s1_cam_a", "s1_cam_b", "s2_cam_a", "s2_cam_b"],
        "task_ids": ["task_1", "task_1", "task_1", "task_1"],
        "scene_ids": ["scene_1", "scene_1", "scene_2", "scene_2"],
        "camera_ids": ["cam_a", "cam_b", "cam_a", "cam_b"],
    }

    filtered_matrix, filtered_metadata = keep_first_camera_per_scene(matrix, metadata)

    assert filtered_matrix.tolist() == [[0, 2], [8, 10]]
    assert filtered_metadata["sample_ids"] == ["s1_cam_a", "s2_cam_a"]
    assert filtered_metadata["camera_ids"] == ["cam_a", "cam_a"]
