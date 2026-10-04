#!/usr/bin/env python3
"""Replace non-video files in cfg3 scenes with matching files from cfg2."""

import argparse
import shutil
from pathlib import Path


def scene_directories(task_directory: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in task_directory.iterdir()
        if path.is_dir() and path.name.startswith("scene_")
    }


def non_mp4_files(scene_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in scene_directory.iterdir()
        if path.is_file()
        and not path.name.lower().endswith(".mp4")
    )


def copyable_files(scene_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in non_mp4_files(scene_directory)
        if not path.name.endswith("Zone.Identifier")
    )


def camera_id(path: Path) -> str:
    if not path.name.startswith("cam_"):
        raise ValueError(f"Expected a cam_ file: {path}")
    camera_id, separator, _ = path.name[4:].partition("_")
    if not separator or not camera_id:
        raise ValueError(f"Cannot parse camera ID from: {path}")
    return camera_id


def camera_npy_files(scene_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in scene_directory.iterdir()
        if path.is_file()
        and path.name.startswith("cam_")
        and path.name.lower().endswith(".npy")
    )


def camera_mp4_files(scene_directory: Path) -> list[Path]:
    return sorted(
        path
        for path in scene_directory.iterdir()
        if path.is_file()
        and path.name.startswith("cam_")
        and path.name.lower().endswith(".mp4")
    )


def build_mp4_rename_pairs(source_scene: Path, target_scene: Path) -> list[tuple[Path, Path]]:
    source_camera_ids = sorted({camera_id(path) for path in camera_npy_files(source_scene)})
    target_camera_groups: dict[str, list[Path]] = {}
    for path in camera_mp4_files(target_scene):
        target_camera_groups.setdefault(camera_id(path), []).append(path)
    target_camera_ids = sorted(target_camera_groups)

    if len(source_camera_ids) != 3:
        raise ValueError(
            f"Expected 3 camera .npy files in {source_scene}, "
            f"found {source_camera_ids}"
        )
    if len(target_camera_ids) != 3:
        raise ValueError(
            f"Expected 3 camera MP4 groups in {target_scene}, "
            f"found {target_camera_ids}"
        )

    rename_pairs = []
    for old_camera_id, new_camera_id in zip(target_camera_ids, source_camera_ids):
        old_prefix = f"cam_{old_camera_id}"
        for source_file in target_camera_groups[old_camera_id]:
            target_file = source_file.with_name(
                f"cam_{new_camera_id}{source_file.name[len(old_prefix):]}"
            )
            if source_file != target_file:
                rename_pairs.append((source_file, target_file))
    return rename_pairs


def rename_mp4_files(source_scene: Path, target_scene: Path, dry_run: bool) -> int:
    rename_pairs = build_mp4_rename_pairs(source_scene, target_scene)
    if dry_run:
        for source_file, target_file in rename_pairs:
            print(f"RENAME {source_file} -> {target_file}")
        return len(rename_pairs)

    temporary_pairs = []
    for index, (source_file, target_file) in enumerate(rename_pairs):
        temporary_file = source_file.with_name(
            f".__rh20t_camera_rename_{index}_{source_file.name}"
        )
        if temporary_file.exists():
            raise FileExistsError(f"Temporary rename path already exists: {temporary_file}")
        source_file.rename(temporary_file)
        temporary_pairs.append((temporary_file, target_file))

    for temporary_file, target_file in temporary_pairs:
        temporary_file.rename(target_file)
    return len(rename_pairs)


def build_plan(
    dataset_root: Path, start_task: int, end_task: int
) -> list[tuple[Path, Path]]:
    """Validate the whole range and return matching cfg2/cfg3 scene directories."""
    plan = []
    for task_number in range(start_task, end_task + 1):
        task_name = f"task_{task_number:04d}"
        cfg2_task = dataset_root / "RH20T_cfg2" / task_name
        cfg3_task = dataset_root / "RH20T_cfg3" / task_name

        if not cfg2_task.is_dir():
            raise FileNotFoundError(f"cfg2 task directory does not exist: {cfg2_task}")
        if not cfg3_task.is_dir():
            raise FileNotFoundError(f"cfg3 task directory does not exist: {cfg3_task}")

        cfg2_scenes = scene_directories(cfg2_task)
        cfg3_scenes = scene_directories(cfg3_task)
        if set(cfg2_scenes) != set(cfg3_scenes):
            raise ValueError(
                f"Scene directories do not match for {task_name}: "
                f"cfg2={sorted(cfg2_scenes)}, cfg3={sorted(cfg3_scenes)}"
            )

        for scene_name in sorted(cfg3_scenes):
            build_mp4_rename_pairs(cfg2_scenes[scene_name], cfg3_scenes[scene_name])
            plan.append((cfg2_scenes[scene_name], cfg3_scenes[scene_name]))

    return plan


def sync_scenes(plan: list[tuple[Path, Path]], dry_run: bool) -> tuple[int, int, int]:
    deleted_count = 0
    copied_count = 0
    renamed_count = 0

    for source_scene, target_scene in plan:
        for target_file in non_mp4_files(target_scene):
            if dry_run:
                print(f"DELETE {target_file}")
            else:
                target_file.unlink()
            deleted_count += 1

        for source_file in copyable_files(source_scene):
            target_file = target_scene / source_file.name
            if dry_run:
                print(f"COPY   {source_file} -> {target_file}")
            else:
                shutil.copy2(source_file, target_file)
            copied_count += 1

        renamed_count += rename_mp4_files(source_scene, target_scene, dry_run)

    return deleted_count, copied_count, renamed_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_task", type=int, help="First task number, inclusive")
    parser.add_argument("end_task", type=int, help="Last task number, inclusive")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset/RH20T_subset"),
        help="RH20T subset root (default: dataset/RH20T_subset)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned operations without deleting or copying files",
    )
    args = parser.parse_args()

    if args.start_task < 0 or args.end_task < args.start_task:
        parser.error("Require 0 <= start_task <= end_task")

    dataset_root = args.dataset_root.expanduser().resolve()
    try:
        plan = build_plan(dataset_root, args.start_task, args.end_task)
        deleted_count, copied_count, renamed_count = sync_scenes(plan, args.dry_run)
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    action = "Would process" if args.dry_run else "Processed"
    print(
        f"{action} {args.end_task - args.start_task + 1} task(s), "
        f"{len(plan)} scene(s): {deleted_count} file(s) deleted, "
        f"{copied_count} file(s) copied, {renamed_count} MP4 file(s) renamed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
