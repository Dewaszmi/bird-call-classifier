#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.spectrogram import DEFAULT_IMAGE_SIZE, preprocess_spectrogram
from models.vgg import BirdVGG

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "best.pt"


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


def preprocess_audio(
    file_path: Path,
    target_sr: int = 22050,
    target_duration: float = 30.0,
    resize_to: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    target_samples = int(target_sr * target_duration)

    y, sr = librosa.load(file_path, sr=target_sr, duration=target_duration)
    y = pad_audio(y, target_samples)

    spec = compute_mel_spectrogram(y, sr)
    spec_tensor = preprocess_spectrogram(spec, resize_to=resize_to, normalize=True)

    lengths = None
    if resize_to is None:
        lengths = torch.tensor([spec_tensor.size(2)])

    return spec_tensor.unsqueeze(0), lengths


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict bird species from an audio file.")
    parser.add_argument("audio_file", type=Path, help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--device",
        default="mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="Inference device",
    )

    args = parser.parse_args()

    if not args.audio_file.exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}. Please train the model first.")
        return 1

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    pooling_mode = checkpoint.get("pooling_mode", "gap")
    batching_strategy = checkpoint.get("batching_strategy", "none")

    model = BirdVGG(num_classes=len(classes), pooling_mode=pooling_mode).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Processing audio file: {args.audio_file}...")
    try:
        resize_to = DEFAULT_IMAGE_SIZE if batching_strategy == "none" else None
        input_tensor, lengths = preprocess_audio(args.audio_file, resize_to=resize_to)
        input_tensor = input_tensor.to(device)
        if lengths is not None:
            lengths = lengths.to(device)
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return 1

    print("Running inference...")
    with torch.no_grad():
        if pooling_mode == "masked-gap":
            if lengths is None:
                lengths = torch.tensor([input_tensor.size(3)], device=device)
            logits = model(input_tensor, lengths)
        else:
            logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    top_prob, top_indices = torch.topk(probabilities, k=min(3, len(classes)))

    print("\n--- Predictions ---")
    for i in range(len(top_indices)):
        species = classes[top_indices[i].item()]
        prob = top_prob[i].item() * 100
        print(f"{i+1}. {species}: {prob:.2f}%")

    print("\n===========================================")
    print(f"Best Match: {classes[top_indices[0].item()]}")
    print("===========================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
