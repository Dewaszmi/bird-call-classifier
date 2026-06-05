#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = {
    "processed": ROOT / "processed_data",
    "raw": ROOT / "data",
}
DEFAULT_OUTPUT = ROOT / "kaggle_upload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package bird-call data for upload as a Kaggle dataset."
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your Kaggle username (used in dataset id: username/slug)",
    )
    parser.add_argument(
        "--dataset-slug",
        default="bird-call-processed",
        help="Dataset slug (default: bird-call-processed)",
    )
    parser.add_argument(
        "--title",
        default="Bird Call Processed Spectrograms",
        help="Human-readable dataset title",
    )
    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_DIRS),
        default="processed",
        help=(
            "Which local folder to package: "
            "'processed' = mel spectrograms (.npy, recommended); "
            "'raw' = downloaded audio files"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output folder for Kaggle upload (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = SOURCE_DIRS[args.source]

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        if args.source == "processed":
            print("Run: python scripts/preprocess_data.py", file=sys.stderr)
        else:
            print("Run: python scripts/download_data.py", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_name = "processed_data" if args.source == "processed" else "data"
    dest_dir = output_dir / folder_name
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)

    metadata = {
        "title": args.title,
        "id": f"{args.username}/{args.dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    file_count = sum(1 for path in dest_dir.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in dest_dir.rglob("*") if path.is_file())

    print(f"Packaged {file_count} files ({total_bytes / 1e6:.1f} MB)")
    print(f"Output: {output_dir}")
    print(f"Dataset id: {metadata['id']}")
    print()
    print("Upload to Kaggle:")
    print(f"  kaggle datasets create -p {output_dir}")
    print()
    print("Or upload manually at https://www.kaggle.com/datasets/new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
