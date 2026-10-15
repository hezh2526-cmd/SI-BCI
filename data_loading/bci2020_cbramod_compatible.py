#!/usr/bin/env python3
"""DataLoader for the separately stored CBraMod-compatible BCIC2020-3 data.

Only ``within`` and official ``mixed`` protocols are supported here. This
module intentionally does not fall back to the legacy 512-point preprocessing
tree, which prevents experiments from silently mixing the two pipelines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CBRAMOD_PREPROCESSED_ROOT = (
    PROJECT_ROOT
    / "Data"
    / "BCI2020"
    / "processed"
    / "cbramod_compatible"
    / "preprocessed_data"
)

EXPECTED_CHANNELS = 64
EXPECTED_TIMES = 600
EXPECTED_SFREQ = 200.0
EXPECTED_CLASSES = 5
EXPECTED_PREPROCESSING_ID = "bcic2020_cbramod_compatible_v1"
Protocol = Literal["within", "mixed"]


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def _subject_id(subject: int | str) -> str:
    if isinstance(subject, str):
        value = subject.strip().lower()
        if value.startswith("sub-"):
            value = value[4:]
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError(f"cannot parse subject: {subject!r}") from exc
    else:
        number = int(subject)
    if not 1 <= number <= 15:
        raise ValueError("subject must be in 1..15")
    return f"sub-{number:02d}"


class CBraModCompatibleEpochDataset(Dataset):
    """In-memory epochs with shape ``(trial, 64, 600)``."""

    def __init__(self, path: Path | str, *, return_metadata: bool = False) -> None:
        self.path = Path(path).resolve()
        self.return_metadata = return_metadata
        self.input_field = "X"
        self.normalized_on_load = False
        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        with np.load(self.path, allow_pickle=False) as data:
            required = {
                "X", "y", "times", "sfreq", "ch_names", "class_names",
                "subject_id", "subject_number", "original_epoch_index",
                "is_flagged", "preprocessing_id", "split",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{self.path}: missing fields {missing}")

            self.x = np.array(data["X"], dtype=np.float32, copy=True, order="C")
            self.y = np.asarray(data["y"], dtype=np.int64).copy()
            self.times = np.asarray(data["times"], dtype=np.float64).copy()
            self.sfreq = float(data["sfreq"])
            self.ch_names = np.asarray(data["ch_names"], dtype="U5").copy()
            self.class_names = np.asarray(data["class_names"], dtype="U10").copy()
            self.subject_id = np.asarray(data["subject_id"], dtype="U6").copy()
            self.subject_number = np.asarray(data["subject_number"], dtype=np.int16).copy()
            self.original_epoch_index = np.asarray(
                data["original_epoch_index"], dtype=np.int64
            ).copy()
            self.is_flagged = np.asarray(data["is_flagged"], dtype=bool).copy()
            self.preprocessing_id = _scalar_text(data["preprocessing_id"])
            self.split = _scalar_text(data["split"])

        n_epochs = self.y.size
        if self.x.shape != (n_epochs, EXPECTED_CHANNELS, EXPECTED_TIMES):
            raise ValueError(f"{self.path}: unexpected X shape {self.x.shape}")
        if not np.isfinite(self.x).all():
            raise ValueError(f"{self.path}: X contains NaN or Inf")
        if not np.all((0 <= self.y) & (self.y < EXPECTED_CLASSES)):
            raise ValueError(f"{self.path}: labels must be in 0..4")
        if self.times.shape != (EXPECTED_TIMES,):
            raise ValueError(f"{self.path}: unexpected times shape {self.times.shape}")
        if not np.isclose(self.sfreq, EXPECTED_SFREQ):
            raise ValueError(f"{self.path}: expected 200 Hz, got {self.sfreq}")
        if self.ch_names.shape != (EXPECTED_CHANNELS,):
            raise ValueError(f"{self.path}: unexpected channel metadata")
        if self.class_names.tolist() != ["Hello", "Helpme", "Stop", "Thankyou", "Yes"]:
            raise ValueError(f"{self.path}: unexpected class mapping {self.class_names.tolist()}")
        if self.preprocessing_id != EXPECTED_PREPROCESSING_ID:
            raise ValueError(
                f"{self.path}: expected preprocessing_id={EXPECTED_PREPROCESSING_ID}, "
                f"got {self.preprocessing_id}"
            )
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"{self.path}: invalid split {self.split!r}")

        for name, values in (
            ("subject_id", self.subject_id),
            ("subject_number", self.subject_number),
            ("original_epoch_index", self.original_epoch_index),
            ("is_flagged", self.is_flagged),
        ):
            if values.shape != (n_epochs,):
                raise ValueError(f"{self.path}: unexpected {name} shape {values.shape}")
        if np.any(self.is_flagged):
            raise ValueError(f"{self.path}: CBraMod-compatible data must not flag epochs")

    def __len__(self) -> int:
        return int(self.y.size)

    def __getitem__(self, index: int):
        x = torch.from_numpy(self.x[index])
        y = torch.tensor(int(self.y[index]), dtype=torch.long)
        if not self.return_metadata:
            return x, y
        metadata = {
            "subject_id": str(self.subject_id[index]),
            "subject_number": int(self.subject_number[index]),
            "original_epoch_index": int(self.original_epoch_index[index]),
            "is_flagged": bool(self.is_flagged[index]),
        }
        return x, y, metadata

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.y, minlength=EXPECTED_CLASSES)

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(sorted(np.unique(self.subject_id).tolist()))


@dataclass(frozen=True)
class CBraModCompatibleDataLoaderBundle:
    protocol: Protocol
    train: DataLoader
    val: DataLoader
    test: DataLoader
    preprocessed_root: Path
    subject: str | None = None

    @property
    def datasets(self) -> dict[str, CBraModCompatibleEpochDataset]:
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
    dataset: CBraModCompatibleEpochDataset,
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


def create_cbramod_compatible_dataloaders(
    protocol: Protocol,
    *,
    preprocessed_root: Path | str = DEFAULT_CBRAMOD_PREPROCESSED_ROOT,
    subject: int | str | None = None,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    return_metadata: bool = False,
) -> CBraModCompatibleDataLoaderBundle:
    """Create strictly separated CBraMod-compatible within or mixed loaders."""

    protocol = protocol.lower()  # type: ignore[assignment]
    if protocol not in {"within", "mixed"}:
        raise ValueError("protocol must be 'within' or 'mixed'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    root = Path(preprocessed_root).resolve()
    expected_namespace = (
        PROJECT_ROOT / "Data" / "BCI2020" / "processed" / "cbramod_compatible"
    ).resolve()
    if expected_namespace not in root.parents:
        raise ValueError(
            "preprocessed_root must stay inside Data/BCI2020/processed/"
            "cbramod_compatible"
        )
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    selected_subject: str | None = None
    paths: dict[str, Path] = {}
    if protocol == "within":
        if subject is None:
            raise ValueError("within protocol requires subject")
        selected_subject = _subject_id(subject)
        subject_dir = root / "by_subject" / selected_subject
        paths = {split: subject_dir / f"{split}.npz" for split in ("train", "val", "test")}
    else:
        if subject is not None:
            raise ValueError("mixed protocol does not accept subject")
        mixed_dir = root / "mixed_subject"
        paths = {
            split: mixed_dir / f"{split}_pool.npz"
            for split in ("train", "val", "test")
        }

    datasets = {
        split: CBraModCompatibleEpochDataset(path, return_metadata=return_metadata)
        for split, path in paths.items()
    }
    if protocol == "within":
        for split, dataset in datasets.items():
            if dataset.subject_ids != (selected_subject,):
                raise AssertionError(f"{split}: data do not belong only to {selected_subject}")

    return CBraModCompatibleDataLoaderBundle(
        protocol=protocol,  # type: ignore[arg-type]
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
        subject=selected_subject,
    )


__all__ = [
    "CBraModCompatibleDataLoaderBundle",
    "CBraModCompatibleEpochDataset",
    "DEFAULT_CBRAMOD_PREPROCESSED_ROOT",
    "create_cbramod_compatible_dataloaders",
]
