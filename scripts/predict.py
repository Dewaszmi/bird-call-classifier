#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import TypedDict

import librosa
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bird-call-classifier"))

from datasets.spectrogram import DEFAULT_IMAGE_SIZE, preprocess_spectrogram
from models.vgg import BirdVGG

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"
DEFAULT_MIN_MARGIN = 0.15
DEFAULT_MIN_TOP_PROBABILITY = 0.0


class PredictionEntry(TypedDict):
    species: str
    probability: float


class ConfidenceMetrics(TypedDict):
    top_probability: float
    margin: float
    entropy: float
    normalized_entropy: float


class PredictionResult(TypedDict):
    predictions: list[PredictionEntry]
    other: float
    best_match: str
    identified: bool
    confidence: ConfidenceMetrics


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


def load_audio(
    file_path: Path,
    target_sr: int,
    target_duration: float,
) -> tuple[np.ndarray, int]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"librosa\.core\.audio",
        )
        return librosa.load(file_path, sr=target_sr, duration=target_duration)


def preprocess_audio(
    file_path: Path,
    target_sr: int = 22050,
    target_duration: float = 30.0,
    resize_to: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    target_samples = int(target_sr * target_duration)

    y, sr = load_audio(file_path, target_sr, target_duration)
    y = pad_audio(y, target_samples)

    spec = compute_mel_spectrogram(y, sr)
    spec_tensor = preprocess_spectrogram(spec, resize_to=resize_to, normalize=True)

    lengths = None
    if resize_to is None:
        lengths = torch.tensor([spec_tensor.size(2)])

    return spec_tensor.unsqueeze(0), lengths


def default_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def assess_confidence(probabilities: torch.Tensor) -> ConfidenceMetrics:
    sorted_probs, _ = torch.sort(probabilities, descending=True)
    top_probability = sorted_probs[0].item()
    second_probability = sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0
    margin = top_probability - second_probability

    eps = 1e-12
    entropy = -(probabilities * torch.log(probabilities + eps)).sum().item()
    num_classes = len(probabilities)
    max_entropy = math.log(num_classes) if num_classes > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "top_probability": top_probability,
        "margin": margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def is_confident(
    confidence: ConfidenceMetrics,
    *,
    min_margin: float,
    min_top_probability: float,
) -> bool:
    return (
        confidence["margin"] >= min_margin
        and confidence["top_probability"] >= min_top_probability
    )


def format_predictions(
    probabilities: torch.Tensor,
    classes: list[str],
    top_k: int = 3,
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    min_top_probability: float = DEFAULT_MIN_TOP_PROBABILITY,
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

    confidence = assess_confidence(probabilities)
    identified = is_confident(
        confidence,
        min_margin=min_margin,
        min_top_probability=min_top_probability,
    )
    best_match = (
        classes[top_indices[0].item()] if identified else "Unable to identify"
    )

    return {
        "predictions": predictions,
        "other": other_prob,
        "best_match": best_match,
        "identified": identified,
        "confidence": confidence,
    }


def predict_audio(
    audio_file: Path,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    min_top_probability: float = DEFAULT_MIN_TOP_PROBABILITY,
) -> PredictionResult:
    if device is None:
        device = default_device()

    torch_device = torch.device(device)
    checkpoint_data = torch.load(checkpoint, map_location=torch_device, weights_only=True)
    classes = checkpoint_data["classes"]
    batching_strategy = checkpoint_data.get("batching_strategy", "naive")

    model = BirdVGG(num_classes=len(classes)).to(torch_device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    resize_to = (
        DEFAULT_IMAGE_SIZE
        if batching_strategy in {"naive", "none"}
        else None
    )
    input_tensor, _lengths = preprocess_audio(audio_file, resize_to=resize_to)
    input_tensor = input_tensor.to(torch_device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    return format_predictions(
        probabilities,
        classes,
        min_margin=min_margin,
        min_top_probability=min_top_probability,
    )


def print_predictions(result: PredictionResult) -> None:
    confidence = result["confidence"]
    print("\n--- Predictions ---")
    for index, entry in enumerate(result["predictions"], start=1):
        print(f"{index}. {entry['species']}: {entry['probability'] * 100:.2f}%")
    print(f"Other: {result['other'] * 100:.2f}%")

    print("\n--- Confidence ---")
    print(f"Top probability: {confidence['top_probability'] * 100:.2f}%")
    print(f"Margin (1st - 2nd): {confidence['margin'] * 100:.2f}%")
    print(f"Normalized entropy: {confidence['normalized_entropy']:.3f}")

    print("\n===========================================")
    print(f"Best Match: {result['best_match']}")
    if not result["identified"]:
        print("(Top class did not meet confidence thresholds)")
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
    parser.add_argument(
        "--min-margin",
        type=float,
        default=DEFAULT_MIN_MARGIN,
        help=(
            "Minimum gap between the top two class probabilities required to "
            f"accept a prediction (default: {DEFAULT_MIN_MARGIN})"
        ),
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=DEFAULT_MIN_TOP_PROBABILITY,
        help=(
            "Minimum top-class probability required to accept a prediction "
            f"(default: {DEFAULT_MIN_TOP_PROBABILITY}, disabled)"
        ),
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
        result = predict_audio(
            args.audio_file,
            args.checkpoint,
            args.device,
            min_margin=args.min_margin,
            min_top_probability=args.min_probability,
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1

    print_predictions(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
