#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.spectrogram import (
    DEFAULT_FIXED_DURATION_SEC,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DURATION_SEC,
    duration_to_frames,
    preprocess_spectrogram,
)
from models.vgg import BirdVGG

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"


class PredictionEntry(TypedDict):
    species: str
    probability: float


class PredictionResult(TypedDict):
    predictions: list[PredictionEntry]
    other: float
    best_match: str


def pad_audio(y: np.ndarray, target_length: int) -> np.ndarray:
    length = len(y)
    if length < target_length:
        padding = target_length - length
        y = np.pad(y, (0, padding), mode="constant")
    elif length > target_length:
        y = y[:target_length]
    return y


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def load_audio(file_path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"librosa\.core\.audio",
        )
        return librosa.load(file_path, sr=target_sr)


def preprocess_audio(
    file_path: Path,
    target_sr: int = 22050,
    *,
    batching_strategy: str = "none",
    fixed_duration: float = DEFAULT_FIXED_DURATION_SEC,
    max_duration: float = DEFAULT_MAX_DURATION_SEC,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if batching_strategy == "none":
        target_samples = int(target_sr * fixed_duration)
        y, sr = load_audio(file_path, target_sr)
        y = pad_audio(y, target_samples)
        spec = compute_mel_spectrogram(y, sr)
        spec_tensor = preprocess_spectrogram(
            spec,
            fixed_time_frames=duration_to_frames(fixed_duration),
            resize_to=DEFAULT_IMAGE_SIZE,
            normalize=True,
        )
        return spec_tensor.unsqueeze(0), None

    if batching_strategy == "masked-gap":
        max_samples = int(target_sr * max_duration)
        y, sr = load_audio(file_path, target_sr)
        if len(y) > max_samples:
            y = y[:max_samples]

        spec = compute_mel_spectrogram(y, sr)
        fixed_frames = duration_to_frames(fixed_duration)
        valid_frames = min(spec.shape[1], fixed_frames)
        spec_tensor = preprocess_spectrogram(
            spec,
            fixed_time_frames=fixed_frames,
            resize_to=None,
            normalize=True,
        )
        return spec_tensor.unsqueeze(0), torch.tensor([valid_frames], dtype=torch.long)

    max_samples = int(target_sr * max_duration)
    y, sr = load_audio(file_path, target_sr)
    if len(y) > max_samples:
        y = y[:max_samples]

    spec = compute_mel_spectrogram(y, sr)
    spec_tensor = preprocess_spectrogram(
        spec,
        fixed_time_frames=None,
        resize_to=None,
        normalize=True,
    )
    return spec_tensor.unsqueeze(0), torch.tensor([spec_tensor.size(2)])


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_predictions(
    probabilities: torch.Tensor,
    classes: list[str],
    top_k: int = 3,
) -> PredictionResult:
    top_prob, top_indices = torch.topk(probabilities, k=min(top_k, len(classes)))
    top_index_set = set(top_indices.tolist())

    predictions: list[PredictionEntry] = []
    for index in top_indices:
        predictions.append(
            {
                "species": classes[index.item()],
                "probability": probabilities[index.item()].item(),
            }
        )

    other_prob = sum(
        probabilities[i].item()
        for i in range(len(classes))
        if i not in top_index_set
    )

    return {
        "predictions": predictions,
        "other": other_prob,
        "best_match": classes[top_indices[0].item()],
    }


def predict_audio(
    audio_file: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
) -> PredictionResult:
    if device is None:
        device = default_device()

    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=True)
    classes = checkpoint_data["classes"]
    pooling_mode = checkpoint_data.get("pooling_mode", "gap")
    batching_strategy = checkpoint_data.get("batching_strategy", "none")
    fixed_duration = checkpoint_data.get("fixed_duration", DEFAULT_FIXED_DURATION_SEC)

    model = BirdVGG(num_classes=len(classes), pooling_mode=pooling_mode).to(torch_device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    input_tensor, lengths = preprocess_audio(
        audio_file,
        batching_strategy=batching_strategy,
        fixed_duration=fixed_duration,
    )
    input_tensor = input_tensor.to(torch_device)
    if lengths is not None:
        lengths = lengths.to(torch_device)

    with torch.no_grad():
        if pooling_mode == "masked-gap":
            if lengths is None:
                lengths = torch.tensor([input_tensor.size(3)], device=torch_device)
            logits = model(input_tensor, lengths)
        else:
            logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    return format_predictions(probabilities, classes)


def print_predictions(result: PredictionResult) -> None:
    print("\n--- Predictions ---")
    for index, entry in enumerate(result["predictions"], start=1):
        print(f"{index}. {entry['species']}: {entry['probability'] * 100:.2f}%")
    print(f"Other: {result['other'] * 100:.2f}%")

    print("\n===========================================")
    print(f"Best Match: {result['best_match']}")
    print("===========================================\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict bird species from an audio file."
    )
    parser.add_argument(
        "audio_file", type=Path, help="Path to the audio file (.wav, .mp3, etc.)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Inference device",
    )

    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    if not args.checkpoint.exists():
        print(
            f"Error: Checkpoint not found: {args.checkpoint}. Please train the model first."
        )
        return 1

    print(f"Using device: {args.device}")
    print(f"Loading checkpoint from {args.checkpoint}...")
    print(f"Processing audio file: {args.audio_file}...")

    try:
        print("Running inference...")
        result = predict_audio(args.audio_file, args.checkpoint, args.device)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1

    print_predictions(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
