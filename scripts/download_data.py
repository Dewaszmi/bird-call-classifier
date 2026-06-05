#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from safe_filenames import sanitize_filename

API_ENDPOINT = "https://xeno-canto.org/api/3/recordings"
DEFAULT_SPECIES_FILE = Path(__file__).resolve().parent.parent / "bird_species.tsv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MAX_PER_SPECIES = 200
DEFAULT_MIN_RECORDING_LENGTH_SEC = 0
DEFAULT_MAX_RECORDING_LENGTH_SEC = 30
DOWNLOADED_IDS_FILE = "downloaded_ids.json"
API_TIMEOUT = (15, 120)  # (connect, read) seconds
DOWNLOAD_TIMEOUT = (15, 180)
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 3


@dataclass(frozen=True)
class Species:
    genus: str
    epithet: str

    @property
    def scientific_name(self) -> str:
        return f"{self.genus} {self.epithet}"


def parse_scientific_name(scientific_name: str) -> tuple[str, str] | None:
    """Parse 'Genus species' into (genus, epithet)."""
    name = scientific_name.strip()
    if not name:
        return None
    parts = name.split()
    if len(parts) < 2:
        print(f"Warning: skipping invalid scientific name: {name!r}", file=sys.stderr)
        return None
    return parts[0], parts[1]


def load_species_list(path: Path) -> list[Species]:
    species: list[Species] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "Scientific name" not in reader.fieldnames:
            raise ValueError("Species file must contain a 'Scientific name' column")

        for row in reader:
            parsed = parse_scientific_name(row["Scientific name"])
            if not parsed:
                continue
            genus, epithet = parsed
            species.append(Species(genus=genus, epithet=epithet))

    species.sort(key=lambda item: item.scientific_name.lower())
    return species


def species_folder_name(genus: str, epithet: str) -> str:
    return f"{genus}_{epithet}"


def parse_recording_length_seconds(recording: dict) -> float | None:
    """Parse xeno-canto 'length' field (e.g. '0:23', '1:05') into seconds."""
    raw = recording.get("length")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    raw = str(raw).strip()
    if not raw:
        return None
    parts = raw.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None


def build_query(
    genus: str,
    epithet: str,
    *,
    min_length_sec: int,
    max_length_sec: int,
) -> str:
    """Build an xeno-canto API v3 query for one species."""
    return (
        f'grp:birds gen:{genus} sp:"{epithet}" '
        f"len:{min_length_sec}-{max_length_sec}"
    )


def recording_length_in_range(
    length_sec: float | None,
    *,
    min_length_sec: int,
    max_length_sec: int,
) -> bool:
    if length_sec is None:
        return True
    return min_length_sec <= length_sec <= max_length_sec


def request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: tuple[int, int],
    verbose: bool = False,
    **kwargs,
) -> requests.Response:
    """Retry on timeouts and transient connection errors."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code in {429, 502, 503, 504}:
                response.close()
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}", response=response
                )
            response.raise_for_status()
            return response
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                break
            wait = RETRY_BACKOFF_SEC * attempt
            if verbose:
                print(
                    f"    Request failed ({exc}), retrying in {wait}s...",
                    file=sys.stderr,
                )
            time.sleep(wait)
    raise last_error  # type: ignore[misc]


class XenoCantoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "bird-call-classifier/1.0 (data download script)",
            }
        )

    def search(
        self,
        query: str,
        *,
        per_page: int | None = None,
        max_results: int | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        if per_page is None:
            per_page = min(max_results or DEFAULT_MAX_PER_SPECIES, 500)
        per_page = max(50, per_page)

        all_recordings: list[dict] = []
        page = 1

        while True:
            if verbose:
                print(f"  Fetching API page {page}...")

            data = self._fetch_page(query, page, per_page, verbose=verbose)
            recordings = data.get("recordings", [])
            all_recordings.extend(recordings)

            if max_results and len(all_recordings) >= max_results:
                return all_recordings[:max_results]

            current_page = int(data.get("page", page))
            total_pages = int(data.get("numPages", 1))
            if current_page >= total_pages:
                break

            page += 1
            time.sleep(0.2)

        return all_recordings

    def _fetch_page(
        self, query: str, page: int, per_page: int, *, verbose: bool = False
    ) -> dict:
        params = {
            "query": query,
            "key": self.api_key,
            "page": page,
            "per_page": per_page,
        }
        response = request_with_retries(
            self.session,
            "GET",
            API_ENDPOINT,
            params=params,
            timeout=API_TIMEOUT,
            verbose=verbose,
        )
        data = response.json()

        if "error" in data:
            message = data.get("message", data.get("error", "Unknown API error"))
            raise RuntimeError(f"xeno-canto API error: {message}")

        return data


def load_downloaded_ids(output_dir: Path) -> set[str]:
    path = output_dir / DOWNLOADED_IDS_FILE
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {str(i) for i in data}


def save_downloaded_ids(output_dir: Path, ids: set[str]) -> None:
    path = output_dir / DOWNLOADED_IDS_FILE
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted(ids, key=int), f, indent=2)


def download_recording(
    session: requests.Session,
    recording: dict,
    dest_dir: Path,
    *,
    verbose: bool = False,
) -> Path:
    file_url = recording.get("file", "")
    if not file_url:
        raise ValueError(f"No file URL for recording {recording.get('id')}")
    if not file_url.startswith("http"):
        file_url = "https:" + file_url

    file_name = recording.get("file-name") or f"XC{recording.get('id', 'unknown')}.mp3"
    file_name = sanitize_filename(file_name)
    dest_path = dest_dir / file_name

    if dest_path.exists():
        if verbose:
            print(f"    Already on disk: {file_name}")
        return dest_path

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
        print(f"    Saved {file_name} ({size_mb:.2f} MB)")

    return dest_path


def log_message(message: str, *, verbose: bool, progress: tqdm | None) -> None:
    if not verbose:
        return
    if progress is not None:
        progress.write(message)
    else:
        print(message)


def download_species(
    client: XenoCantoClient,
    session: requests.Session,
    species: Species,
    output_dir: Path,
    *,
    max_recordings: int,
    min_length_sec: int,
    max_length_sec: int,
    skip_existing: bool,
    verbose: bool,
    progress: tqdm | None = None,
) -> dict[str, int]:
    query = build_query(
        species.genus,
        species.epithet,
        min_length_sec=min_length_sec,
        max_length_sec=max_length_sec,
    )
    folder = output_dir / species_folder_name(species.genus, species.epithet)
    folder.mkdir(parents=True, exist_ok=True)

    log_message(
        f"\n{species.scientific_name}\n  Query: {query}\n  Output: {folder}",
        verbose=verbose,
        progress=progress,
    )

    recordings = client.search(query, max_results=max_recordings, verbose=verbose)
    log_message(
        f"  Found {len(recordings)} recording(s)",
        verbose=verbose,
        progress=progress,
    )

    if progress is not None:
        progress.total = (progress.total or 0) + len(recordings)
        progress.refresh()

    downloaded_ids = load_downloaded_ids(output_dir) if skip_existing else set()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for i, recording in enumerate(recordings, 1):
        rec_id = str(recording.get("id", ""))
        if progress is not None:
            progress.set_postfix_str(species.scientific_name, refresh=False)

        if skip_existing and rec_id in downloaded_ids:
            stats["skipped"] += 1
            log_message(
                f"  [{i}/{len(recordings)}] Skip ID {rec_id} (already downloaded)",
                verbose=verbose,
                progress=progress,
            )
            if progress is not None:
                progress.update(1)
            continue

        length_sec = parse_recording_length_seconds(recording)
        if not recording_length_in_range(
            length_sec,
            min_length_sec=min_length_sec,
            max_length_sec=max_length_sec,
        ):
            stats["skipped"] += 1
            log_message(
                f"  [{i}/{len(recordings)}] Skip ID {rec_id} "
                f"(length {length_sec:.1f}s outside "
                f"{min_length_sec}-{max_length_sec}s)",
                verbose=verbose,
                progress=progress,
            )
            if progress is not None:
                progress.update(1)
            continue

        if verbose:
            en = recording.get("en", "")
            log_message(
                f"  [{i}/{len(recordings)}] Downloading ID {rec_id} ({en})",
                verbose=verbose,
                progress=progress,
            )

        try:
            download_recording(session, recording, folder, verbose=verbose)
            downloaded_ids.add(rec_id)
            stats["downloaded"] += 1
            time.sleep(0.3)
        except Exception as exc:
            stats["failed"] += 1
            message = f"  Failed ID {rec_id}: {exc}"
            if progress is not None:
                progress.write(message)
            else:
                print(message, file=sys.stderr)

        if progress is not None:
            progress.update(1)

    if skip_existing:
        save_downloaded_ids(output_dir, downloaded_ids)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download bird call recordings from xeno-canto.org.",
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
        help=f"Base directory for downloads (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("XENO_CANTO_API_KEY"),
        help="xeno-canto API key (or set XENO_CANTO_API_KEY)",
    )
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=DEFAULT_MAX_PER_SPECIES,
        metavar="N",
        help=f"Maximum recordings per species (default: {DEFAULT_MAX_PER_SPECIES})",
    )
    parser.add_argument(
        "--min-recording-length",
        type=int,
        default=DEFAULT_MIN_RECORDING_LENGTH_SEC,
        metavar="SEC",
        help=(
            "Minimum recording length in seconds "
            f"(default: {DEFAULT_MIN_RECORDING_LENGTH_SEC}, no lower limit)"
        ),
    )
    parser.add_argument(
        "--max-recording-length",
        type=int,
        default=DEFAULT_MAX_RECORDING_LENGTH_SEC,
        metavar="SEC",
        help=f"Maximum recording length in seconds (default: {DEFAULT_MAX_RECORDING_LENGTH_SEC})",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        metavar="NAME",
        help='Download only these species, e.g. "Grus grus" "Parus major"',
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download even if recording ID was downloaded before",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress details",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the global progress bar",
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

    if args.min_recording_length < 0:
        print("Error: --min-recording-length must be >= 0", file=sys.stderr)
        return 1

    if args.max_recording_length < args.min_recording_length:
        print(
            "Error: --max-recording-length must be >= --min-recording-length",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.species:
        species_list = []
        for name in args.species:
            parsed = parse_scientific_name(name)
            if parsed:
                genus, epithet = parsed
                species_list.append(Species(genus=genus, epithet=epithet))
    else:
        species_list = load_species_list(args.species_file)

    if not species_list:
        print("Error: no species to download.", file=sys.stderr)
        return 1

    client = XenoCantoClient(args.api_key)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "bird-call-classifier/1.0 (data download script)",
        }
    )

    totals = {"downloaded": 0, "skipped": 0, "failed": 0}
    print(f"Downloading {len(species_list)} species to {args.output_dir}")

    progress: tqdm | None = None
    if not args.no_progress:
        progress = tqdm(
            total=0,
            unit="recording",
            dynamic_ncols=True,
            desc="Downloading",
        )

    try:
        for species in species_list:
            try:
                stats = download_species(
                    client,
                    session,
                    species,
                    args.output_dir,
                    max_recordings=args.max_per_species,
                    min_length_sec=args.min_recording_length,
                    max_length_sec=args.max_recording_length,
                    skip_existing=not args.no_skip,
                    verbose=args.verbose,
                    progress=progress,
                )
            except requests.exceptions.RequestException as exc:
                message = f"Failed {species.scientific_name}: {exc}"
                if progress is not None:
                    progress.write(message)
                else:
                    print(message, file=sys.stderr)
                totals["failed"] += 1
                continue
            for key in totals:
                totals[key] += stats[key]
    finally:
        if progress is not None:
            progress.close()

    print(
        f"\nDone. Downloaded: {totals['downloaded']}, "
        f"skipped: {totals['skipped']}, failed: {totals['failed']}"
    )
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
