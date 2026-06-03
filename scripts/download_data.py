#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_ENDPOINT = "https://xeno-canto.org/api/3/recordings"
DEFAULT_SPECIES_FILE = Path(__file__).resolve().parent.parent / "bird_species.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MAX_PER_SPECIES = 50
DOWNLOADED_IDS_FILE = "downloaded_ids.json"
API_TIMEOUT = (15, 120)  # (connect, read) seconds
DOWNLOAD_TIMEOUT = (15, 180)
MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 3


def parse_species_line(line: str) -> tuple[str, str] | None:
    """Parse 'Genus species' into (genus, epithet)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 2:
        print(f"Warning: skipping invalid species line: {line!r}", file=sys.stderr)
        return None
    genus, epithet = parts[0], parts[1]
    return genus, epithet


def load_species_list(path: Path) -> list[tuple[str, str]]:
    species = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parsed = parse_species_line(line)
            if parsed:
                species.append(parsed)
    return species


def species_folder_name(genus: str, epithet: str) -> str:
    return f"{genus}_{epithet}"


def build_query(genus: str, epithet: str) -> str:
    """Build an xeno-canto API v3 query for one species."""
    return f'grp:birds gen:{genus} sp:"{epithet}"'


def sanitize_filename(name: str) -> str:
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip(". ")


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
                print(f"    Request failed ({exc}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise last_error  # type: ignore[misc]


class XenoCantoClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "bird-call-classifier/1.0 (data download script)",
        })

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


def download_species(
    client: XenoCantoClient,
    session: requests.Session,
    genus: str,
    epithet: str,
    output_dir: Path,
    *,
    max_recordings: int,
    skip_existing: bool,
    verbose: bool,
) -> dict[str, int]:
    query = build_query(genus, epithet)
    folder = output_dir / species_folder_name(genus, epithet)
    folder.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{genus} {epithet}")
        print(f"  Query: {query}")
        print(f"  Output: {folder}")

    recordings = client.search(query, max_results=max_recordings, verbose=verbose)
    if verbose:
        print(f"  Found {len(recordings)} recording(s)")

    downloaded_ids = load_downloaded_ids(output_dir) if skip_existing else set()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for i, recording in enumerate(recordings, 1):
        rec_id = str(recording.get("id", ""))
        if skip_existing and rec_id in downloaded_ids:
            stats["skipped"] += 1
            if verbose:
                print(f"  [{i}/{len(recordings)}] Skip ID {rec_id} (already downloaded)")
            continue

        if verbose:
            en = recording.get("en", "")
            print(f"  [{i}/{len(recordings)}] Downloading ID {rec_id} ({en})")

        try:
            download_recording(session, recording, folder, verbose=verbose)
            downloaded_ids.add(rec_id)
            stats["downloaded"] += 1
            time.sleep(0.3)
        except Exception as exc:
            stats["failed"] += 1
            print(f"  Failed ID {rec_id}: {exc}", file=sys.stderr)

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
        "-v", "--verbose",
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.species:
        species_list = []
        for name in args.species:
            parsed = parse_species_line(name)
            if parsed:
                species_list.append(parsed)
    else:
        species_list = load_species_list(args.species_file)

    if not species_list:
        print("Error: no species to download.", file=sys.stderr)
        return 1

    client = XenoCantoClient(args.api_key)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "bird-call-classifier/1.0 (data download script)",
    })

    totals = {"downloaded": 0, "skipped": 0, "failed": 0}
    print(f"Downloading {len(species_list)} species to {args.output_dir}")

    for genus, epithet in species_list:
        try:
            stats = download_species(
                client,
                session,
                genus,
                epithet,
                args.output_dir,
                max_recordings=args.max_per_species,
                skip_existing=not args.no_skip,
                verbose=args.verbose,
            )
        except requests.exceptions.RequestException as exc:
            print(f"Failed {genus} {epithet}: {exc}", file=sys.stderr)
            totals["failed"] += 1
            continue
        for key in totals:
            totals[key] += stats[key]

    print(
        f"\nDone. Downloaded: {totals['downloaded']}, "
        f"skipped: {totals['skipped']}, failed: {totals['failed']}"
    )
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
