from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

Sample = tuple[Path, int]


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


def train_val_split(
    samples: list[Sample],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Stratified split by class; splits whole recordings, not random crops."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    by_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    for class_samples in by_class.values():
        shuffled = class_samples.copy()
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_samples.extend(shuffled)
            continue

        n_val = max(1, int(len(shuffled) * val_ratio))
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


class SpectrogramDataset(Dataset):
    """Loads log-mel spectrograms from processed_data/<species>/*.npy."""

    def __init__(
        self,
        samples: list[Sample],
        classes: list[str],
        image_size: tuple[int, int] = (128, 128),
        normalize: bool = True,
    ):
        self.samples = samples
        self.classes = classes
        self.image_size = image_size
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        spec = np.load(path).astype(np.float32)

        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram in {path}, got shape {spec.shape}")

        # (freq, time) -> (1, freq, time)
        spec = torch.from_numpy(spec).unsqueeze(0)
        spec = F.interpolate(
            spec.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if self.normalize:
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        return spec, label
