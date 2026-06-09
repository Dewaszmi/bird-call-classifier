#!/usr/bin/env python3

from __future__ import annotations

import re
import shutil
from pathlib import Path

# Characters rejected by Kaggle datasets and problematic on multiple platforms.
FORBIDDEN_FILENAME_CHARS = "<>:\"/\\|?*[]'`,,&"


def sanitize_filename(name: str) -> str:
    path = Path(name)
    stem = path.stem
    suffix = path.suffix

    for char in FORBIDDEN_FILENAME_CHARS:
        stem = stem.replace(char, "_")

    stem = re.sub(r"_+", "_", stem).strip("._ ")
    if not stem:
        stem = "file"

    return f"{stem}{suffix}"


def unique_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_tree_with_safe_names(source: Path, destination: Path) -> int:
    if destination.exists():
        shutil.rmtree(destination)

    copied = 0
    for src_path in sorted(source.rglob("*")):
        if not src_path.is_file():
            continue

        rel_path = src_path.relative_to(source)
        safe_parts = [sanitize_filename(part) for part in rel_path.parts]
        dest_path = unique_path(destination / Path(*safe_parts[:-1]), safe_parts[-1])
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)
        copied += 1

    return copied


def rename_tree_with_safe_names(
    root: Path, *, dry_run: bool = False
) -> list[tuple[Path, Path]]:
    planned: list[tuple[Path, Path]] = []
    reserved = {path.resolve() for path in root.rglob("*") if path.is_file()}

    for src_path in sorted(
        root.rglob("*"), key=lambda path: len(path.parts), reverse=True
    ):
        if not src_path.is_file():
            continue

        safe_name = sanitize_filename(src_path.name)
        if safe_name == src_path.name:
            continue

        dest_path = unique_path(src_path.parent, safe_name)
        if dest_path.resolve() == src_path.resolve():
            continue

        planned.append((src_path, dest_path))
        reserved.discard(src_path.resolve())
        reserved.add(dest_path.resolve())

    if dry_run:
        return planned

    for src_path, dest_path in planned:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.rename(dest_path)

    return planned
