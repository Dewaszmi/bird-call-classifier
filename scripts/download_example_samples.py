#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_data import (
    Species,
    XenoCantoClient,
    build_query,
    load_species_list,
    parse_recording_length_seconds,
    recording_length_in_range,
    request_with_retries,
    species_folder_name,
    DOWNLOAD_TIMEOUT,
)
from safe_filenames import sanitize_filename

DEFAULT_SPECIES_FILE = Path(__file__).resolve().parent.parent / "bird_species.tsv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "example_samples"
DEFAULT_TRAINING_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_FILE = "manifest.json"
DEFAULT_MAX_RECORDING_LENGTH_SEC = 30
FALLBACK_MAX_LENGTHS_SEC = (60, 120, 300)
SEARCH_PAGE_SIZE = 500
MAX_SEARCH_PAGES = 20


def load_excluded_recording_ids(training_dir: Path) -> set[str]:
    """Return xeno-canto recording IDs present in the training dataset."""
    excluded: set[str] = set()

    ids_path = training_dir / "downloaded_ids.json"
    if ids_path.exists():
        with ids_path.open(encoding="utf-8") as f:
            excluded.update(str(rec_id) for rec_id in json.load(f))

    for path in training_dir.rglob("*"):
        if not path.is_file():
            continue
        stem = path.stem
        if stem.upper().startswith("XC"):
            numeric = stem[2:].split("-", 1)[0].split("_", 1)[0]
            if numeric.isdigit():
                excluded.add(numeric)

    return excluded


def download_recording_to_path(
    session: requests.Session,
    recording: dict,
    dest_path: Path,
    *,
    verbose: bool = False,
) -> Path:
    file_url = recording.get("file", "")
    if not file_url:
        raise ValueError(f"No file URL for recording {recording.get('id')}")
    if not file_url.startswith("http"):
        file_url = "https:" + file_url

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = request_with_retries(
        session,
        "GET",
        file_url,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
        verbose=verbose,
    )

    with dest_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    if verbose:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"    Saved {dest_path.name} ({size_mb:.2f} MB)")

    return dest_path


def example_filename(species: Species, recording: dict) -> str:
    rec_id = recording.get("id", "unknown")
    ext = Path(recording.get("file-name") or f"XC{rec_id}.mp3").suffix or ".mp3"
    base = f"{species.genus}_{species.epithet}_XC{rec_id}{ext}"
    return sanitize_filename(base)


def find_unseen_recording(
    client: XenoCantoClient,
    species: Species,
    excluded_ids: set[str],
    *,
    min_length_sec: int,
    max_length_sec: int,
    verbose: bool = False,
) -> dict | None:
    max_lengths = [max_length_sec, *FALLBACK_MAX_LENGTHS_SEC]
    seen_lengths: set[int] = set()

    for current_max in max_lengths:
        if current_max in seen_lengths or current_max < min_length_sec:
            continue
        seen_lengths.add(current_max)

        query = build_query(
            species.genus,
            species.epithet,
            min_length_sec=min_length_sec,
            max_length_sec=current_max,
        )

        for page in range(1, MAX_SEARCH_PAGES + 1):
            data = client._fetch_page(query, page, SEARCH_PAGE_SIZE, verbose=verbose)
            recordings = data.get("recordings", [])
            if not recordings:
                break

            for recording in recordings:
                rec_id = str(recording.get("id", ""))
                if not rec_id or rec_id in excluded_ids:
                    continue

                length_sec = parse_recording_length_seconds(recording)
                if not recording_length_in_range(
                    length_sec,
                    min_length_sec=min_length_sec,
                    max_length_sec=current_max,
                ):
                    continue

                if verbose and current_max != max_length_sec:
                    print(
                        f"    Using fallback max length {current_max}s for "
                        f"{species.scientific_name}"
                    )
                return recording

            current_page = int(data.get("page", page))
            total_pages = int(data.get("numPages", 1))
            if current_page >= total_pages:
                break

            time.sleep(0.2)

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download one xeno-canto evaluation recording per species, "
            "excluding IDs already used for training."
        ),
    )
    parser.add_argument(
        "--species-file",
        type=Path,
        default=DEFAULT_SPECIES_FILE,
        help=f"Path to species list (default: {DEFAULT_SPECIES_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for evaluation samples (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help=f"Training data directory (default: {DEFAULT_TRAINING_DIR})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("XENO_CANTO_API_KEY"),
        help="xeno-canto API key (or set XENO_CANTO_API_KEY)",
    )
    parser.add_argument(
        "--max-recording-length",
        type=int,
        default=DEFAULT_MAX_RECORDING_LENGTH_SEC,
        metavar="SEC",
        help=(
            "Maximum recording length in seconds "
            f"(default: {DEFAULT_MAX_RECORDING_LENGTH_SEC})"
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress details",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: xeno-canto API key required.\n"
            "Register at https://xeno-canto.org/account and set XENO_CANTO_API_KEY "
            "or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    if not args.species_file.exists():
        print(f"Error: species file not found: {args.species_file}", file=sys.stderr)
        return 1

    if not args.training_dir.exists():
        print(
            f"Error: training directory not found: {args.training_dir}",
            file=sys.stderr,
        )
        return 1

    species_list = load_species_list(args.species_file)
    if not species_list:
        print("Error: no species to download.", file=sys.stderr)
        return 1

    excluded_ids = load_excluded_recording_ids(args.training_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = XenoCantoClient(args.api_key)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "bird-call-classifier/1.0 (example sample download script)",
        }
    )

    manifest: dict[str, dict] = {}
    manifest_path = args.output_dir / MANIFEST_FILE
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            manifest = json.load(f)

    failed: list[str] = []
    print(
        f"Fetching one unseen recording per species into {args.output_dir} "
        f"(excluding {len(excluded_ids)} training IDs)"
    )

    for species in tqdm(species_list, desc="Species", unit="species"):
        folder_key = species_folder_name(species.genus, species.epithet)
        existing = manifest.get(folder_key, {})
        existing_path = args.output_dir / existing.get("filename", "")
        existing_id = str(existing.get("id", ""))
        if existing_id and existing_id not in excluded_ids and existing_path.exists():
            if args.verbose:
                print(f"  {species.scientific_name}: keep {existing_path.name}")
            continue

        recording = find_unseen_recording(
            client,
            species,
            excluded_ids,
            min_length_sec=0,
            max_length_sec=args.max_recording_length,
            verbose=args.verbose,
        )
        if recording is None:
            failed.append(species.scientific_name)
            continue

        rec_id = str(recording.get("id", ""))
        filename = example_filename(species, recording)
        dest_path = args.output_dir / filename

        if args.verbose:
            print(
                f"  {species.scientific_name}: downloading XC{rec_id} -> {filename}"
            )

        try:
            download_recording_to_path(
                session,
                recording,
                dest_path,
                verbose=args.verbose,
            )
        except Exception as exc:
            failed.append(f"{species.scientific_name} ({exc})")
            continue

        manifest[folder_key] = {
            "scientific_name": species.scientific_name,
            "id": rec_id,
            "filename": filename,
            "en": recording.get("en", ""),
            "length": recording.get("length", ""),
            "url": f"https://xeno-canto.org/{rec_id}",
        }
        excluded_ids.add(rec_id)
        time.sleep(0.3)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    if failed:
        print("\nFailed species:", file=sys.stderr)
        for item in failed:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"\nDone. Saved {len(manifest)} evaluation sample(s) and {MANIFEST_FILE}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
