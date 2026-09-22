"""Run configured evaluation steps and record one auditable pipeline manifest.

This runner deliberately delegates model work to the existing ``runs/`` entry
points.  A JSON file therefore describes *which* checkpoints, splits and
outputs belong to an experiment without duplicating training/evaluation code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Pipeline JSON configuration.")
    parser.add_argument("--only", help="Comma-separated step names to run.")
    parser.add_argument("--group", help="Comma-separated step groups to run.")
    parser.add_argument("--skip", help="Comma-separated step names to skip.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps whose declared outputs exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failures and continue with later steps.")
    parser.add_argument("--no-bundle", action="store_true", help="Do not export the configured paper bundle.")
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def split_names(value: str | None) -> set[str]:
    return {item.strip() for item in (value or "").split(",") if item.strip()}


def output_paths(step: dict[str, Any]) -> list[Path]:
    values = step.get("outputs", [])
    if isinstance(values, str):
        values = [values]
    return [resolve(value) for value in values]


def command_for(step: dict[str, Any], python_bin: str) -> list[str]:
    runner = str(step.get("runner", "python")).lower()
    script = step.get("script")
    if not script:
        raise ValueError(f"Step {step.get('name', '<unnamed>')} has no script.")
    if runner == "bash":
        command = ["bash", str(resolve(script))]
    elif runner == "python":
        command = [python_bin, str(resolve(script))]
    else:
        raise ValueError(f"Unsupported runner={runner!r} in step {step.get('name')!r}.")
    args = step.get("args", [])
    if not isinstance(args, list):
        raise ValueError(f"Step args must be a list: {step.get('name')!r}")
    return command + [str(item) for item in args]


def output_state(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    experiment_id = str(config.get("experiment_id") or config_path.stem)
    python_bin = str(config.get("python_bin") or sys.executable)
    steps = config.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError("config.steps must be a list")

    only = split_names(args.only)
    groups = split_names(args.group)
    skipped = split_names(args.skip)
    log_dir = resolve(config.get("log_dir", f"artifacts/experiments/{experiment_id}/logs"))
    manifest_path = resolve(config.get("manifest", f"artifacts/experiments/{experiment_id}/pipeline_manifest.json"))
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "project_root": str(ROOT),
        "started_at": now(),
        "steps": [],
    }
    write_json(manifest_path, manifest)

    for raw_step in steps:
        if not isinstance(raw_step, dict):
            raise ValueError("Every config.steps item must be an object")
        step = dict(raw_step)
        name = str(step.get("name") or step.get("script") or f"step_{len(manifest['steps']) + 1}")
        if only and name not in only:
            continue
        step_groups = step.get("groups") or step.get("group") or []
        if isinstance(step_groups, str):
            step_groups = [step_groups]
        if groups and not groups.intersection(str(item) for item in step_groups):
            continue
        if name in skipped:
            continue
        outputs = output_paths(step)
        record: dict[str, Any] = {
            "name": name,
            "started_at": now(),
            "outputs": [str(path) for path in outputs],
        }
        try:
            command = command_for(step, python_bin)
            record["command"] = command
            record["display_command"] = shlex.join(command)
            if args.skip_existing and output_state(outputs):
                record["status"] = "skipped_existing"
                print(f"[{name}] skip existing outputs", flush=True)
            else:
                env = os.environ.copy()
                for key, value in (step.get("env") or {}).items():
                    env[str(key)] = str(value)
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{name}.log"
                record["log"] = str(log_path)
                print(f"[{name}] {shlex.join(command)}", flush=True)
                if args.dry_run:
                    record["status"] = "dry_run"
                else:
                    start = time.monotonic()
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    record["returncode"] = completed.returncode
                    record["duration_seconds"] = round(time.monotonic() - start, 3)
                    if completed.returncode:
                        raise subprocess.CalledProcessError(completed.returncode, command)
                    record["status"] = "completed"
            record["finished_at"] = now()
            manifest["steps"].append(record)
            write_json(manifest_path, manifest)
        except Exception as exc:  # Keep the manifest useful even after a failed step.
            record["status"] = "failed"
            record["error"] = repr(exc)
            record["finished_at"] = now()
            manifest["steps"].append(record)
            write_json(manifest_path, manifest)
            print(f"[{name}] failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                manifest["finished_at"] = now()
                write_json(manifest_path, manifest)
                return 1

    manifest["finished_at"] = now()
    write_json(manifest_path, manifest)
    bundle = config.get("bundle") or {}
    if bundle.get("enabled", False) and not args.no_bundle and not args.dry_run:
        from export_paper_bundle import export_bundle

        export_bundle(manifest_path, bundle)
    print(f"Pipeline manifest: {manifest_path}")
    return 0 if all(item.get("status") != "failed" for item in manifest["steps"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
