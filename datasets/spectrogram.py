from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Sample = tuple[Path, int]

SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_MELS = 128
DEFAULT_IMAGE_SIZE = (128, 128)


def preprocess_spectrogram(
    spec: np.ndarray | torch.Tensor,
    *,
    resize_to: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert a log-mel spectrogram to model input."""
    if isinstance(spec, np.ndarray):
        spec = torch.from_numpy(spec).float()
    else:
        spec = spec.float()

    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram (freq, time), got shape {tuple(spec.shape)}")

    spec = spec.unsqueeze(0)  # (1, freq, time)

    if resize_to is not None:
        spec = F.interpolate(
            spec.unsqueeze(0),
            size=resize_to,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if normalize:
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

    return spec


def collate_fixed(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    specs, labels = zip(*batch)
    return torch.stack(specs), torch.tensor(labels, dtype=torch.long)


def collate_padded(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad spectrograms to the longest width in the batch."""
    specs, labels = zip(*batch)
    lengths = torch.tensor([spec.size(2) for spec in specs], dtype=torch.long)
    max_time = int(lengths.max())

    padded = [F.pad(spec, (0, max_time - spec.size(2))) for spec in specs]
    return torch.stack(padded), torch.tensor(labels, dtype=torch.long), lengths


def collate_no_pad(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != 1:
        raise ValueError(f"collate_no_pad expects batch size 1, got {len(batch)}")

    spec, label = batch[0]
    return (
        spec.unsqueeze(0),
        torch.tensor([label], dtype=torch.long),
        torch.tensor([spec.size(2)], dtype=torch.long),
    )


def discover_samples(root: Path) -> tuple[list[Sample], list[str]]:
    """Find all .npy spectrograms under root/<species>/ directories."""
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory not found: {root}")

    classes = sorted(
        d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    if not classes:
        raise ValueError(f"No species directories found in {root}")

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    samples: list[Sample] = []

    for species in classes:
        species_dir = root / species
        for path in sorted(species_dir.glob("*.npy")):
            samples.append((path, class_to_idx[species]))

    if not samples:
        raise ValueError(f"No .npy files found under {root}")

    return samples, classes


def train_val_test_split(
    samples: list[Sample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Stratified split by class; splits whole recordings, not random crops."""
    if not 0.0 < val_ratio + test_ratio < 1.0:
        raise ValueError("val_ratio + test_ratio must be between 0 and 1")

    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    for class_samples in by_class.values():
        shuffled = class_samples.copy()
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_samples.extend(shuffled)
            continue

        n_val = max(1, int(len(shuffled) * val_ratio))
        n_test = max(1, int(len(shuffled) * test_ratio))

        if n_val + n_test >= len(shuffled):
            n_val = max(1, len(shuffled) // 3)
            n_test = max(1, len(shuffled) // 3)

        val_samples.extend(shuffled[:n_val])
        test_samples.extend(shuffled[n_val : n_val + n_test])
        train_samples.extend(shuffled[n_val + n_test :])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    return train_samples, val_samples, test_samples


class SpectrogramDataset(Dataset):
    """Loads log-mel spectrograms from processed_data/<species>/*.npy."""

    def __init__(
        self,
        samples: list[Sample],
        classes: list[str],
        resize_to: tuple[int, int] | None = DEFAULT_IMAGE_SIZE,
        normalize: bool = True,
    ):
        self.samples = samples
        self.classes = classes
        self.resize_to = resize_to
        self.normalize = normalize
        self._time_frames: list[int] | None = None

    def get_time_frames(self) -> list[int]:
        if self._time_frames is None:
            self._time_frames = [
                int(np.load(path, mmap_mode="r").shape[1]) for path, _ in self.samples
            ]
        return self._time_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        while True:
            try:
                path, label = self.samples[idx]
                spec = np.load(path).astype(np.float32)
                break
            except Exception as e:
                print(f"Warning: Failed to load {path} ({e}), picking another random sample.")
                idx = random.randint(0, len(self.samples) - 1)

        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram in {path}, got shape {spec.shape}")

        spec = preprocess_spectrogram(
            spec,
            resize_to=self.resize_to,
            normalize=self.normalize,
        )
        return spec, label
