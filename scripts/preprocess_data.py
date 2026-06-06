#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.spectrogram import (
    DEFAULT_MAX_DURATION_SEC,
    HOP_LENGTH,
    N_MELS,
    SAMPLE_RATE,
)


def truncate_audio(y: np.ndarray, max_length: int) -> np.ndarray:
    if len(y) > max_length:
        return y[:max_length]
    return y


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
    n_fft: int = 2048,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert audio to variable-length log-mel spectrograms.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION_SEC,
        help=(
            f"Truncate clips longer than this many seconds (default: {DEFAULT_MAX_DURATION_SEC}). "
            "Shorter clips are kept at their natural length."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if the output .npy already exists",
    )
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "data")
    output_dir = os.path.join(ROOT, "processed_data")

    os.makedirs(output_dir, exist_ok=True)

    target_sr = SAMPLE_RATE
    max_samples = int(target_sr * args.max_duration)

    species_dirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]

    for species in species_dirs:
        species_dir = os.path.join(data_dir, species)
        output_species_dir = os.path.join(output_dir, species)
        os.makedirs(output_species_dir, exist_ok=True)

        audio_files = []
        for ext in ("*.mp3", "*.wav"):
            audio_files.extend(glob.glob(os.path.join(species_dir, ext)))

        for file_path in audio_files:
            file_name = os.path.basename(file_path)
            output_file_name = os.path.splitext(file_name)[0] + ".npy"
            output_file_path = os.path.join(output_species_dir, output_file_name)

            if os.path.exists(output_file_path) and not args.force:
                continue

            try:
                y, sr = librosa.load(file_path, sr=target_sr)
                y = truncate_audio(y, max_samples)

                mel_spec = compute_mel_spectrogram(y, sr)
                np.save(output_file_path, mel_spec)
                print(f"Processed: {species} / {file_name} -> shape {mel_spec.shape}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
