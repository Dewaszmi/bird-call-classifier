#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from safe_filenames import rename_tree_with_safe_names

DEFAULT_TARGETS = {
    "data": ROOT / "data",
    "processed": ROOT / "processed_data",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename local dataset files to remove characters forbidden by Kaggle."
    )
    parser.add_argument(
        "--target",
        choices=sorted(DEFAULT_TARGETS),
        default="data",
        help="Which dataset folder to sanitize (default: data)",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Custom folder to sanitize instead of the default target path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned renames without changing files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = args.path or DEFAULT_TARGETS[args.target]

    if not target_dir.exists():
        print(f"Error: directory not found: {target_dir}", file=sys.stderr)
        return 1

    renames = rename_tree_with_safe_names(target_dir, dry_run=args.dry_run)
    if not renames:
        print(f"No filenames to change under {target_dir}")
        return 0

    action = "Would rename" if args.dry_run else "Renamed"
    print(f"{action} {len(renames)} file(s) under {target_dir}")
    for src_path, dest_path in renames[:20]:
        print(f"  {src_path.name} -> {dest_path.name}")
    if len(renames) > 20:
        print(f"  ... and {len(renames) - 20} more")

    if args.dry_run:
        print("\nRe-run without --dry-run to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
