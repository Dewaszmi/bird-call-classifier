import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_DIR = ROOT / "docs" / "presentation"
PROCESSED_DIR = ROOT / "processed_data"
DATA_DIR = ROOT / "data"

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_MELS = 128
N_FFT = 2048


def compute_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)


def trim_spectrogram(spec: np.ndarray, top_db: float = 35.0) -> np.ndarray:
    threshold = spec.max() - top_db
    active = np.where(spec.max(axis=0) > threshold)[0]
    if len(active) == 0:
        return spec
    return spec[:, : active[-1] + 1]


def save_spectrogram_figure(
    spec: np.ndarray,
    output_path: Path,
    *,
    title: str,
    figsize: tuple[float, float] = (10, 4),
) -> None:
    plt.figure(figsize=figsize)
    plt.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    plt.title(title)
    plt.ylabel("Mel Frequency Bins")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def find_sample_source() -> tuple[Path, str]:
    npy_files = sorted(PROCESSED_DIR.rglob("*.npy"))
    if npy_files:
        sample_file = npy_files[0]
        return sample_file, sample_file.parent.name

    audio_files: list[Path] = []
    if DATA_DIR.is_dir():
        for ext in ("*.mp3", "*.wav", "*.flac", "*.ogg"):
            audio_files.extend(sorted(DATA_DIR.rglob(ext)))

    if not audio_files:
        raise FileNotFoundError(
            "No processed_data/*.npy or data audio files found. "
            "Download and preprocess data first."
        )

    sample_file = audio_files[0]
    return sample_file, sample_file.parent.name


def plot_spectrogram() -> None:
    sample_path, label = find_sample_source()

    if sample_path.suffix == ".npy":
        padded_spec = np.load(sample_path)
        trimmed_spec = trim_spectrogram(padded_spec)
    else:
        y, sr = librosa.load(sample_path, sr=SAMPLE_RATE)
        y, _ = librosa.effects.trim(y, top_db=30)

        padded = np.pad(y, (0, int(SAMPLE_RATE * 30) - len(y)), mode="constant")
        padded_spec = compute_mel_spectrogram(padded, sr)
        trimmed_spec = compute_mel_spectrogram(y, sr)

    save_spectrogram_figure(
        padded_spec,
        PRESENTATION_DIR / "spectrogram_plot.png",
        title=f"Log-Mel Spectrogram\n{label}",
    )
    save_spectrogram_figure(
        trimmed_spec,
        PRESENTATION_DIR / "spectrogram_sample.png",
        title=f"Log-Mel Spectrogram\n{label}",
        figsize=(8, 4),
    )
    print(f"Saved spectrogram plots to {PRESENTATION_DIR}")


def plot_history() -> None:
    history_file = ROOT / "checkpoints" / "history.json"
    if not history_file.exists():
        print("No history.json found.")
        return

    with history_file.open(encoding="utf-8") as f:
        data = json.load(f)

    epochs = [item["epoch"] for item in data]
    train_loss = [item["train_loss"] for item in data]
    val_loss = [item["val_loss"] for item in data]
    val_acc = [item["val_accuracy"] for item in data]
    val_f1 = [item["val_macro_f1"] for item in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax1.plot(epochs, val_loss, label="Validation Loss", marker="o")
    ax1.set_title("Learning Curve (Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    ax2.plot(epochs, val_acc, label="Accuracy", marker="s", color="green")
    ax2.plot(epochs, val_f1, label="Macro F1", marker="^", color="orange")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    output_path = PRESENTATION_DIR / "history_plot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved history plot to {output_path}")


if __name__ == "__main__":
    plot_history()
    plot_spectrogram()
