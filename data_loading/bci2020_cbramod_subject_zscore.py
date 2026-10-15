#!/usr/bin/env python3
"""Within/mixed loaders with leakage-free per-subject train-set z-scoring.

The underlying signals remain the CBraMod-compatible 3-second, 200 Hz data.
For each subject independently, channel mean/std are fitted only on that
subject's official training trials across the epoch and time dimensions. The
same statistics are then applied lazily to that subject's train/val/test data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .bci2020_cbramod_compatible import (
    CBraModCompatibleEpochDataset,
    DEFAULT_CBRAMOD_PREPROCESSED_ROOT,
    EXPECTED_CHANNELS,
)


SUBJECT_NUMBERS = np.arange(1, 16, dtype=np.int16)
DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True)
class SubjectwiseNormalizer:
    subject_numbers: np.ndarray  # (15,)
    mean: np.ndarray  # (15, 64)
    std: np.ndarray  # (15, 64)
    epsilon: float


def fit_subjectwise_normalizer(
    train_dataset: CBraModCompatibleEpochDataset,
    *,
    epsilon: float = DEFAULT_EPSILON,
    expected_trials_per_subject: int | None = 300,
) -> SubjectwiseNormalizer:
    """Fit per-subject/channel statistics from exactly the supplied trials.

    ``expected_trials_per_subject`` remains 300 for the existing full-data
    Within/Mixed loaders. Few-shot callers pass their selected subset size,
    which prevents unselected official training trials from influencing the
    normalizer.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if expected_trials_per_subject is not None and expected_trials_per_subject <= 0:
        raise ValueError("expected_trials_per_subject must be positive or None")
    subject_numbers = np.unique(train_dataset.subject_number).astype(np.int16)
    if subject_numbers.size == 0:
        raise ValueError("training dataset contains no subjects")
    if not np.isin(subject_numbers, SUBJECT_NUMBERS).all():
        raise ValueError(f"unexpected subject numbers: {subject_numbers.tolist()}")

    means = np.empty((subject_numbers.size, EXPECTED_CHANNELS), dtype=np.float32)
    stds = np.empty((subject_numbers.size, EXPECTED_CHANNELS), dtype=np.float32)

    for row, subject_number in enumerate(subject_numbers):
        indices = np.flatnonzero(train_dataset.subject_number == subject_number)
        if (expected_trials_per_subject is not None and
                indices.size != expected_trials_per_subject):
            raise ValueError(
                f"sub-{subject_number:02d}: expected "
                f"{expected_trials_per_subject} training trials, "
                f"got {indices.size}"
            )
        # Use float64 accumulation for stable statistics, then store float32.
        subject_x = np.asarray(train_dataset.x[indices], dtype=np.float64)
        mean = subject_x.mean(axis=(0, 2))
        std = subject_x.std(axis=(0, 2))
        std = np.maximum(std, epsilon)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError(f"sub-{subject_number:02d}: non-finite normalization statistics")
        means[row] = mean.astype(np.float32)
        stds[row] = std.astype(np.float32)

    return SubjectwiseNormalizer(
        subject_numbers=subject_numbers,
        mean=means,
        std=stds,
        epsilon=float(epsilon),
    )


class SubjectwiseZScoreDataset(Dataset):
    """Apply training-derived subject/channel statistics in ``__getitem__``."""

    def __init__(
        self,
        base: CBraModCompatibleEpochDataset,
        normalizer: SubjectwiseNormalizer,
    ) -> None:
        self.base = base
        self.normalizer = normalizer
        self._mean = torch.from_numpy(normalizer.mean[:, :, None])
        self._std = torch.from_numpy(normalizer.std[:, :, None])
        self._subject_to_row = {
            int(subject_number): row
            for row, subject_number in enumerate(normalizer.subject_numbers)
        }

        observed = np.unique(base.subject_number)
        if not np.array_equal(observed, normalizer.subject_numbers):
            raise ValueError(
                f"{base.path}: expected subjects 1..15, got {observed.tolist()}"
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        x, y = item[:2]
        subject_number = int(self.base.subject_number[index])
        row = self._subject_to_row[subject_number]
        x = (x - self._mean[row]) / self._std[row]
        if len(item) == 2:
            return x, y
        return x, y, item[2]

    @property
    def class_counts(self) -> np.ndarray:
        return self.base.class_counts

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return self.base.subject_ids

    @property
    def y(self) -> np.ndarray:
        return self.base.y


@dataclass(frozen=True)
class SubjectZScoreDataLoaderBundle:
    protocol: str
    train: DataLoader
    val: DataLoader
    test: DataLoader
    preprocessed_root: Path
    normalizer: SubjectwiseNormalizer
    subject: str | None = None

    @property
    def fold(self) -> None:
        return None

    @property
    def held_out_subject(self) -> None:
        return None

    @property
    def datasets(self) -> dict[str, SubjectwiseZScoreDataset]:
        return {
            "train": self.train.dataset,
            "val": self.val.dataset,
            "test": self.test.dataset,
        }


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: SubjectwiseZScoreDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
    )


def create_subject_zscore_mixed_dataloaders(
    *,
    preprocessed_root: Path | str = DEFAULT_CBRAMOD_PREPROCESSED_ROOT,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    return_metadata: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> SubjectZScoreDataLoaderBundle:
    """Create mixed loaders with statistics fitted exclusively on train data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    root = Path(preprocessed_root).resolve()
    expected_namespace = (
        Path(__file__).resolve().parents[2]
        / "Data" / "BCI2020" / "processed" / "cbramod_compatible"
    ).resolve()
    if expected_namespace not in root.parents:
        raise ValueError(
            "preprocessed_root must stay inside Data/BCI2020/processed/"
            "cbramod_compatible"
        )

    mixed_root = root / "mixed_subject"
    base = {
        split: CBraModCompatibleEpochDataset(
            mixed_root / f"{split}_pool.npz",
            return_metadata=return_metadata,
        )
        for split in ("train", "val", "test")
    }
    normalizer = fit_subjectwise_normalizer(base["train"], epsilon=epsilon)
    if not np.array_equal(normalizer.subject_numbers, SUBJECT_NUMBERS):
        raise ValueError(
            "mixed training pool must contain subjects 1..15; got "
            f"{normalizer.subject_numbers.tolist()}"
        )
    datasets = {
        split: SubjectwiseZScoreDataset(dataset, normalizer)
        for split, dataset in base.items()
    }
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    return SubjectZScoreDataLoaderBundle(
        protocol="mixed",
        train=_make_loader(
            datasets["train"], batch_size=batch_size, shuffle=True, seed=seed,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        val=_make_loader(
            datasets["val"], batch_size=batch_size, shuffle=False, seed=seed + 1,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        test=_make_loader(
            datasets["test"], batch_size=batch_size, shuffle=False, seed=seed + 2,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        preprocessed_root=root,
        normalizer=normalizer,
        subject=None,
    )


def create_subject_zscore_within_dataloaders(
    subject: int | str,
    *,
    preprocessed_root: Path | str = DEFAULT_CBRAMOD_PREPROCESSED_ROOT,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    return_metadata: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> SubjectZScoreDataLoaderBundle:
    """Create one subject's loaders using only its train-set statistics."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if isinstance(subject, str):
        text = subject.strip().lower()
        if text.startswith("sub-"):
            text = text[4:]
        subject_number = int(text)
    else:
        subject_number = int(subject)
    if subject_number not in SUBJECT_NUMBERS:
        raise ValueError("subject must be in 1..15")
    subject_id = f"sub-{subject_number:02d}"

    root = Path(preprocessed_root).resolve()
    expected_namespace = (
        Path(__file__).resolve().parents[2]
        / "Data" / "BCI2020" / "processed" / "cbramod_compatible"
    ).resolve()
    if expected_namespace not in root.parents:
        raise ValueError(
            "preprocessed_root must stay inside Data/BCI2020/processed/"
            "cbramod_compatible"
        )

    subject_root = root / "by_subject" / subject_id
    base = {
        split: CBraModCompatibleEpochDataset(
            subject_root / f"{split}.npz",
            return_metadata=return_metadata,
        )
        for split in ("train", "val", "test")
    }
    for split, dataset in base.items():
        if dataset.subject_ids != (subject_id,):
            raise ValueError(
                f"{split}: expected only {subject_id}, got {dataset.subject_ids}"
            )

    normalizer = fit_subjectwise_normalizer(base["train"], epsilon=epsilon)
    datasets = {
        split: SubjectwiseZScoreDataset(dataset, normalizer)
        for split, dataset in base.items()
    }
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    return SubjectZScoreDataLoaderBundle(
        protocol="within",
        train=_make_loader(
            datasets["train"], batch_size=batch_size, shuffle=True, seed=seed,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        val=_make_loader(
            datasets["val"], batch_size=batch_size, shuffle=False, seed=seed + 1,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        test=_make_loader(
            datasets["test"], batch_size=batch_size, shuffle=False, seed=seed + 2,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        preprocessed_root=root,
        normalizer=normalizer,
        subject=subject_id,
    )


__all__ = [
    "DEFAULT_EPSILON",
    "SubjectwiseNormalizer",
    "SubjectwiseZScoreDataset",
    "create_subject_zscore_mixed_dataloaders",
    "create_subject_zscore_within_dataloaders",
    "fit_subjectwise_normalizer",
]
