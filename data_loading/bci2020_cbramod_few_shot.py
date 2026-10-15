#!/usr/bin/env python3
"""Few-shot loader with subset-only, leakage-free channel normalization."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .bci2020_cbramod_compatible import (
    CBraModCompatibleEpochDataset,
    DEFAULT_CBRAMOD_PREPROCESSED_ROOT,
)
from .bci2020_cbramod_subject_zscore import (
    DEFAULT_EPSILON,
    SubjectwiseNormalizer,
    SubjectwiseZScoreDataset,
    fit_subjectwise_normalizer,
)


ALLOWED_RATIOS = (5, 10, 20, 30, 50, 100)
ALLOWED_SEEDS = (42, 44, 46)
DEFAULT_FEW_SHOT_INDEX_ROOT = (
    DEFAULT_CBRAMOD_PREPROCESSED_ROOT / "few_shot_indices"
)


def _subject_id(subject: int | str) -> str:
    text = str(subject).strip().lower()
    if text.startswith("sub-"):
        text = text[4:]
    number = int(text)
    if not 1 <= number <= 15:
        raise ValueError("subject must be in 1..15")
    return f"sub-{number:02d}"


class IndexedEpochDataset(Dataset):
    """A metadata-preserving view over selected trials of one epoch dataset."""

    def __init__(
        self,
        base: CBraModCompatibleEpochDataset,
        indices: np.ndarray | list[int],
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.path = base.path
        self.return_metadata = base.return_metadata
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional array")
        if np.any(self.indices < 0) or np.any(self.indices >= len(base)):
            raise IndexError("few-shot index outside the source training dataset")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("few-shot indices contain duplicates")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        item = self.base[source_index]
        if len(item) == 2:
            return item
        x, y, metadata = item
        metadata = dict(metadata)
        metadata["dataset_index"] = source_index
        return x, y, metadata

    @property
    def x(self) -> np.ndarray:
        return self.base.x[self.indices]

    @property
    def y(self) -> np.ndarray:
        return self.base.y[self.indices]

    @property
    def subject_id(self) -> np.ndarray:
        return self.base.subject_id[self.indices]

    @property
    def subject_number(self) -> np.ndarray:
        return self.base.subject_number[self.indices]

    @property
    def original_epoch_index(self) -> np.ndarray:
        return self.base.original_epoch_index[self.indices]

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=5)

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(sorted(np.unique(self.subject_id).tolist()))


@dataclass(frozen=True)
class FewShotDataLoaderBundle:
    protocol: str
    train: DataLoader
    val: DataLoader
    test: DataLoader
    preprocessed_root: Path
    index_root: Path
    index_file: Path
    selected_dataset_indices: np.ndarray
    normalizer: SubjectwiseNormalizer
    subject: str
    ratio_percent: int
    subset_seed: int

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
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
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


def _load_selected_indices(
    *,
    index_root: Path,
    subject_id: str,
    ratio_percent: int,
    subset_seed: int,
    train_base: CBraModCompatibleEpochDataset,
) -> tuple[Path, np.ndarray]:
    index_file = index_root / f"seed_{subset_seed:04d}" / f"{subject_id}.json"
    if not index_file.is_file():
        raise FileNotFoundError(index_file)
    document = json.loads(index_file.read_text(encoding="utf-8"))
    if document.get("subject_id") != subject_id:
        raise ValueError(f"{index_file}: subject mismatch")
    if document.get("seed") != subset_seed:
        raise ValueError(f"{index_file}: seed mismatch")
    subset = document.get("subsets", {}).get(str(ratio_percent))
    if not isinstance(subset, dict):
        raise ValueError(f"{index_file}: missing {ratio_percent}% subset")
    indices = np.asarray(subset.get("dataset_indices", []), dtype=np.int64)
    expected_per_class = 60 * ratio_percent // 100
    expected_total = expected_per_class * 5
    if indices.shape != (expected_total,):
        raise ValueError(
            f"{index_file}: {ratio_percent}% requires {expected_total} indices, "
            f"got {indices.size}"
        )
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{index_file}: duplicate indices")
    if np.any(indices < 0) or np.any(indices >= len(train_base)):
        raise ValueError(f"{index_file}: index outside training dataset")
    selected_labels = train_base.y[indices]
    if np.bincount(selected_labels, minlength=5).tolist() != [expected_per_class] * 5:
        raise ValueError(f"{index_file}: selected labels are not balanced")
    if subset.get("labels") != selected_labels.tolist():
        raise ValueError(f"{index_file}: stored labels differ from source training data")
    selected_original = train_base.original_epoch_index[indices].tolist()
    if subset.get("original_epoch_indices") != selected_original:
        raise ValueError(f"{index_file}: stored original indices differ from source")
    return index_file, indices


def create_few_shot_dataloaders(
    subject: int | str,
    ratio_percent: int,
    subset_seed: int,
    *,
    preprocessed_root: Path | str = DEFAULT_CBRAMOD_PREPROCESSED_ROOT,
    index_root: Path | str = DEFAULT_FEW_SHOT_INDEX_ROOT,
    batch_size: int = 16,
    dataloader_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    return_metadata: bool = False,
    epsilon: float = DEFAULT_EPSILON,
) -> FewShotDataLoaderBundle:
    """Create Within few-shot loaders using subset-only normalization.

    Validation and test remain the complete official 50-trial splits. The
    normalizer is fitted *after* index selection and therefore sees only the
    15/30/60/90/150/300 selected training trials.
    """

    subject_id = _subject_id(subject)
    ratio_percent = int(ratio_percent)
    subset_seed = int(subset_seed)
    if ratio_percent not in ALLOWED_RATIOS:
        raise ValueError(f"ratio_percent must be one of {ALLOWED_RATIOS}")
    if subset_seed not in ALLOWED_SEEDS:
        raise ValueError(f"subset_seed must be one of {ALLOWED_SEEDS}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    root = Path(preprocessed_root).resolve()
    indices_root = Path(index_root).resolve()
    expected_namespace = (
        Path(__file__).resolve().parents[2]
        / "Data" / "BCI2020" / "processed" / "cbramod_compatible"
    ).resolve()
    if expected_namespace not in root.parents:
        raise ValueError("preprocessed_root is outside cbramod_compatible")
    if expected_namespace not in indices_root.parents:
        raise ValueError("index_root is outside cbramod_compatible")

    subject_root = root / "by_subject" / subject_id
    base = {
        split: CBraModCompatibleEpochDataset(
            subject_root / f"{split}.npz", return_metadata=return_metadata
        )
        for split in ("train", "val", "test")
    }
    for split, dataset in base.items():
        if dataset.subject_ids != (subject_id,):
            raise ValueError(f"{split}: expected only {subject_id}")

    index_file, selected_indices = _load_selected_indices(
        index_root=indices_root,
        subject_id=subject_id,
        ratio_percent=ratio_percent,
        subset_seed=subset_seed,
        train_base=base["train"],
    )
    selected_train = IndexedEpochDataset(base["train"], selected_indices)
    normalizer = fit_subjectwise_normalizer(
        selected_train,
        epsilon=epsilon,
        expected_trials_per_subject=len(selected_train),
    )
    datasets = {
        "train": SubjectwiseZScoreDataset(selected_train, normalizer),
        "val": SubjectwiseZScoreDataset(base["val"], normalizer),
        "test": SubjectwiseZScoreDataset(base["test"], normalizer),
    }
    loader_seed = subset_seed if dataloader_seed is None else int(dataloader_seed)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
    return FewShotDataLoaderBundle(
        protocol="few_shot",
        train=_make_loader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            seed=loader_seed, num_workers=num_workers, pin_memory=pin_memory,
        ),
        val=_make_loader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            seed=loader_seed + 1, num_workers=num_workers, pin_memory=pin_memory,
        ),
        test=_make_loader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            seed=loader_seed + 2, num_workers=num_workers, pin_memory=pin_memory,
        ),
        preprocessed_root=root,
        index_root=indices_root,
        index_file=index_file,
        selected_dataset_indices=selected_indices.copy(),
        normalizer=normalizer,
        subject=subject_id,
        ratio_percent=ratio_percent,
        subset_seed=subset_seed,
    )


__all__ = [
    "ALLOWED_RATIOS",
    "ALLOWED_SEEDS",
    "DEFAULT_FEW_SHOT_INDEX_ROOT",
    "FewShotDataLoaderBundle",
    "IndexedEpochDataset",
    "create_few_shot_dataloaders",
]
