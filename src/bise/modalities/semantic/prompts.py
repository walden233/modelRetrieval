from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from bise.common.config import load_json_config
from bise.common.paths import resolve_path
from bise.modalities.semantic.schemas import PromptTemplate, SemanticManifestRecord


def _load_prompt_template(config_path: str | Path) -> PromptTemplate:
    payload = load_json_config(resolve_path(config_path))
    return PromptTemplate(
        version=str(payload["version"]),
        system_prompt=str(payload.get("system_prompt", "")).strip(),
        user_template=str(payload.get("user_template", "")).strip(),
        examples=list(payload.get("examples", [])),
        output_schema=dict(payload.get("output_schema", {})),
    )


def _render_examples(examples: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for example in examples:
        lines.append(f"Example input: {example.get('input', '')}")
        lines.append(f"Example output: {json.dumps(example.get('output', {}), ensure_ascii=False)}")
    return "\n".join(lines).strip()


def _render_taxonomy(taxonomy: Dict[str, Any] | None) -> str:
    if not taxonomy:
        return ""
    lines: List[str] = []

    capability_config = taxonomy.get("capability_tags", {})
    capability_tags = capability_config.get("allowed_tags", [])
    capability_aliases = capability_config.get("tag_aliases", {})
    if capability_tags:
        lines.append(f"Allowed capability tags: {', '.join(capability_tags)}")
    if capability_aliases:
        lines.append(f"Capability tag aliases: {json.dumps(capability_aliases, ensure_ascii=False)}")

    task_complexity_options = taxonomy.get("task_complexity_options", [])
    if task_complexity_options:
        lines.append(f"Allowed task complexity values: {', '.join(task_complexity_options)}")

    environment_config = taxonomy.get("environment_tags", {})
    environment_tags = environment_config.get("allowed_tags", [])
    environment_aliases = environment_config.get("tag_aliases", {})
    if environment_tags:
        lines.append(f"Allowed environment tags: {', '.join(environment_tags)}")
    if environment_aliases:
        lines.append(f"Environment tag aliases: {json.dumps(environment_aliases, ensure_ascii=False)}")

    scene_category_options = taxonomy.get("scene_category_options", [])
    if scene_category_options:
        lines.append(f"Allowed scene categories: {', '.join(scene_category_options)}")

    return "\n".join(lines).strip()


def build_description_prompt(
    manifest_record: SemanticManifestRecord,
    prompt_config_path: str | Path,
) -> Dict[str, str]:
    template = _load_prompt_template(prompt_config_path)
    video_role_phrase = _video_role_phrase(manifest_record.video_role)
    prompt_body = template.user_template.format(
        sample_id=manifest_record.sample_id,
        pair_id=manifest_record.pair_id,
        task_id=manifest_record.task_id,
        scene_id=manifest_record.scene_id,
        dataset_name=manifest_record.dataset_name,
        video_role=manifest_record.video_role,
        video_role_phrase=video_role_phrase,
    )
    examples = _render_examples(template.examples)
    if examples:
        prompt_body = f"{prompt_body}\n\n{examples}"
    if template.output_schema:
        prompt_body = f"{prompt_body}\n\nOutput schema:\n{json.dumps(template.output_schema, indent=2, ensure_ascii=False)}"
    return {
        "version": template.version,
        "system_prompt": template.system_prompt,
        "user_prompt": prompt_body.strip(),
    }


def build_label_prompt(
    manifest_record: SemanticManifestRecord,
    prompt_config_path: str | Path,
    taxonomy: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    template = _load_prompt_template(prompt_config_path)
    video_role_phrase = _video_role_phrase(manifest_record.video_role)
    prompt_body = template.user_template.format(
        sample_id=manifest_record.sample_id,
        pair_id=manifest_record.pair_id,
        task_id=manifest_record.task_id,
        scene_id=manifest_record.scene_id,
        dataset_name=manifest_record.dataset_name,
        video_role=manifest_record.video_role,
        video_role_phrase=video_role_phrase,
    )
    taxonomy_text = _render_taxonomy(taxonomy)
    if taxonomy_text:
        prompt_body = f"{prompt_body}\n\n{taxonomy_text}"
    examples = _render_examples(template.examples)
    if examples:
        prompt_body = f"{prompt_body}\n\n{examples}"
    if template.output_schema:
        prompt_body = f"{prompt_body}\n\nOutput schema:\n{json.dumps(template.output_schema, indent=2, ensure_ascii=False)}"
    return {
        "version": template.version,
        "system_prompt": template.system_prompt,
        "user_prompt": prompt_body.strip(),
    }


LABEL_PROMPT = "Use build_label_prompt() with config-backed templates."
TASK_DESCRIPTION_PROMPT = "Use build_description_prompt() with config-backed templates."


def _video_role_phrase(video_role: str) -> str:
    normalized = str(video_role).strip().lower()
    if normalized == "robot":
        return "a robot execution video"
    if normalized == "human":
        return "a human demonstration video"
    return f"a {normalized} video"
