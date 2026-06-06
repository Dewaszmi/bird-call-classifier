from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import BatchSampler, DataLoader

from datasets.spectrogram import (
    SpectrogramDataset,
    collate_fixed,
    collate_fixed_with_lengths,
    collate_no_pad,
    collate_padded,
)

BATCHING_STRATEGIES = ("none", "no-batch", "length-bucketing", "masked-gap")


class LengthBucketBatchSampler(BatchSampler):
    """Group clips with similar spectrogram lengths to minimize batch padding."""

    def __init__(
        self,
        dataset: SpectrogramDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._initial_seed = seed
        self.seed = seed
        self.time_frames = dataset.get_time_frames()

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda idx: self.time_frames[idx])

        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

        if self.shuffle:
            random.Random(self.seed).shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.seed = self._initial_seed + epoch


def build_dataloader(
    dataset: SpectrogramDataset,
    *,
    batching_strategy: str,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if batching_strategy not in BATCHING_STRATEGIES:
        raise ValueError(
            f"Unknown batching strategy {batching_strategy!r}. "
            f"Choose from: {', '.join(BATCHING_STRATEGIES)}"
        )

    if batching_strategy == "none":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fixed,
        )

    if batching_strategy == "no-batch":
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_no_pad,
        )

    if batching_strategy == "length-bucketing":
        batch_sampler = LengthBucketBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
            collate_fn=collate_padded,
        )

    if batching_strategy == "masked-gap":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate_fixed_with_lengths,
        )

    raise ValueError(f"Unhandled batching strategy: {batching_strategy}")


def uses_lengths(batching_strategy: str) -> bool:
    return batching_strategy in {"no-batch", "length-bucketing", "masked-gap"}


def uses_masked_pooling(batching_strategy: str) -> bool:
    return batching_strategy == "masked-gap"
