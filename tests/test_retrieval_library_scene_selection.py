import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[1] / "runs" / "retrieval" / "build_retrieval_library.py"
    spec = importlib.util.spec_from_file_location("build_retrieval_library_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_selection_filters_video_trajectory_and_semantic_features(tmp_path):
    module = _load_module()
    manifest = tmp_path / "fixed_test.json"
    manifest.write_text(json.dumps({"test": ["task_0001/scene_2"]}))
    config = {
        "cfg": "cfg2",
        "data_root": str(tmp_path),
        "output_dir": str(tmp_path / "library"),
        "scene_selection": {"manifest_path": str(manifest), "split": "test"},
        "sampling": {"scenes_per_task": 0},
    }
    allowed, audit = module._load_scene_selection(config)
    assert allowed == {"task_0001/scene_2"}
    assert audit["split"] == "test"

    video = [
        SimpleNamespace(task_id="task_0001", scene_id="task_0001/scene_1"),
        SimpleNamespace(task_id="task_0001", scene_id="task_0001/scene_2"),
    ]
    assert module._filter_video_samples(video, config["sampling"], allowed) == [video[1]]

    trajectory = [
        SimpleNamespace(scene_path=str(tmp_path / "task_0001" / "scene_1")),
        SimpleNamespace(scene_path=str(tmp_path / "task_0001" / "scene_2")),
    ]
    assert module._filter_trajectory_scenes(trajectory, config["sampling"], allowed) == [trajectory[1]]

    builder = module.LibraryBuilder(config, tmp_path / "library")
    builder._add_feature("semantic_text", "human", [1.0, 0.0],
                         "task_0001", "scene_1", "cam_1", "")
    builder._add_feature("semantic_text", "human", [1.0, 0.0],
                         "task_0001", "scene_2", "cam_1", "")
    assert len(builder.feature_records) == 1
    assert builder.feature_records[0]["scene_id"] == "task_0001/scene_2"


def test_cfg2_video_intra_library_uses_shared_fixed_test():
    config = json.loads(
        (Path(__file__).resolve().parents[1]
         / "configs/retrieval/library_rh20t_cfg2_video_intra.json").read_text()
    )
    assert config["scene_selection"] == {
        "manifest_path": "artifacts/manifests/rh20t_shared_v1/cfg2_fixed_test_scenes.json",
        "split": "test",
    }
    assert config["sampling"]["scenes_per_task"] == 0
    assert "cfg2_test" in config["output_dir"]
