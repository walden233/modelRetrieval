import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bise.modalities.semantic.prompts import build_joint_prompt, build_label_prompt
from bise.modalities.semantic.schemas import SemanticManifestRecord


def test_build_label_prompt_renders_multidimensional_taxonomy(tmp_path: Path):
    prompt_path = tmp_path / "prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "version": "label_prompt_v1",
                "system_prompt": "Return JSON only.",
                "user_template": "Label {sample_id} from {video_role_phrase}.",
                "output_schema": {"capability_tags": ["string"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = SemanticManifestRecord(
        sample_id="sample_1",
        pair_id="pair_1",
        task_id="task_1",
        scene_id="scene_1",
        dataset_name="RH20T",
        video_role="robot",
        video_path="/tmp/robot.mp4",
    )
    taxonomy = {
        "capability_tags": {"allowed_tags": ["grasp", "place"], "tag_aliases": {"pick": "grasp"}},
        "task_complexity_options": ["高", "中", "低", "unknown"],
        "environment_tags": {"allowed_tags": ["有障碍物", "无障碍物"], "tag_aliases": {}},
        "scene_category_options": ["工业", "家庭", "unknown"],
    }
    payload = build_label_prompt(manifest, prompt_path, taxonomy=taxonomy)
    user_prompt = payload["user_prompt"]
    assert "Allowed capability tags: grasp, place" in user_prompt
    assert "Allowed task complexity values: 高, 中, 低, unknown" in user_prompt
    assert "Allowed environment tags: 有障碍物, 无障碍物" in user_prompt
    assert "Allowed scene categories: 工业, 家庭, unknown" in user_prompt


def test_build_joint_prompt_renders_taxonomy_and_single_video_wording(tmp_path: Path):
    prompt_path = tmp_path / "joint_prompt.json"
    prompt_path.write_text(
        json.dumps(
            {
                "version": "joint_prompt_v1",
                "system_prompt": "Return JSON only.",
                "user_template": (
                    "Given {video_role_phrase} for sample {sample_id}, "
                    "produce task_description and semantic labels from this single video only. "
                    "Do not mention color information."
                ),
                "output_schema": {
                    "task_description": "string",
                    "capability_tags": ["string"],
                    "task_complexity": "string",
                    "environment_tags": ["string"],
                    "scene_category": "string",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = SemanticManifestRecord(
        sample_id="sample_2",
        pair_id="pair_2",
        task_id="task_2",
        scene_id="scene_2",
        dataset_name="RH20T",
        video_role="human",
        video_path="/tmp/human.mp4",
    )
    taxonomy = {
        "capability_tags": {"allowed_tags": ["transport"], "tag_aliases": {}},
        "task_complexity_options": ["高", "中", "低", "unknown"],
        "environment_tags": {"allowed_tags": ["动态环境", "unknown"], "tag_aliases": {}},
        "scene_category_options": ["家庭", "unknown"],
    }
    payload = build_joint_prompt(manifest, prompt_path, taxonomy=taxonomy)
    user_prompt = payload["user_prompt"]
    assert "a human demonstration video" in user_prompt
    assert "Allowed capability tags: transport" in user_prompt
    assert "Allowed task complexity values: 高, 中, 低, unknown" in user_prompt
    assert "Allowed environment tags: 动态环境, unknown" in user_prompt
    assert "Allowed scene categories: 家庭, unknown" in user_prompt
