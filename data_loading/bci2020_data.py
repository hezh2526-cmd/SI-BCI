#!/usr/bin/env python3
"""为 BCI2020 Within/Mixed/LOSO 提供统一的 PyTorch 数据接口。

公共数据形状始终为 ``(batch, channel, time)``。模型负责自己的输入适配：
EEGNet/ShallowConvNet 可在 forward 中增加单通道图像维，LNN/LSTM 可在
forward 中把后两维交换为 ``(batch, time, channel)``。

归一化规则由实验协议决定：

* within/mixed 读取已经使用训练统计量标准化的 ``X``；
* LOSO 读取 ``X_volts``，并应用该 fold 的 ``normalizer.npz``。

这样可避免 within/mixed 重复归一化，也可避免 LOSO 使用留出被试自己的
within-subject 标准化参数。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "Data" / "BCI2020" / "processed"
EXPECTED_CHANNELS = 64
EXPECTED_TIMES = 512
EXPECTED_CLASSES = 5
Protocol = Literal["within", "mixed", "loso"]
Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class _Source:
    path: Path
    subject_id: str | None = None


@dataclass
class _LoadedSource:
    path: Path
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    subject_number: np.ndarray
    original_epoch_index: np.ndarray
    is_flagged: np.ndarray


class BCIEpochDataset(Dataset):
    """内存中的 BCI2020 epoch 数据集。

    参数 ``input_field`` 与 ``normalizer`` 通常不应由训练脚本手工指定；请优先
    使用 :func:`create_dataloaders`，由实验协议自动选择。
    """

    def __init__(
        self,
        sources: Iterable[_Source],
        *,
        input_field: Literal["X", "X_volts"],
        normalizer: tuple[np.ndarray, np.ndarray] | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.input_field = input_field
        self.return_metadata = return_metadata
        self.normalized_on_load = normalizer is not None
        self._mean = None
        self._std = None
        if normalizer is not None:
            self._mean = np.asarray(normalizer[0], dtype=np.float32)
            self._std = np.asarray(normalizer[1], dtype=np.float32)
            if self._mean.shape != (1, EXPECTED_CHANNELS, 1):
                raise ValueError(f"normalizer mean 形状异常：{self._mean.shape}")
            if self._std.shape != (1, EXPECTED_CHANNELS, 1):
                raise ValueError(f"normalizer std 形状异常：{self._std.shape}")
            if not np.isfinite(self._mean).all() or not np.isfinite(self._std).all():
                raise ValueError("normalizer 包含 NaN/Inf")
            if np.any(self._std <= 0):
                raise ValueError("normalizer std 必须全部大于 0")

        loaded = [self._load_source(source) for source in sources]
        if not loaded:
            raise ValueError("数据源不能为空")

        # Mixed/Within 通常只有一个来源文件，直接复用数组可避免额外复制数百 MB。
        if len(loaded) == 1:
            only = loaded[0]
            self.x = only.x
            self.y = only.y
            self.subject_id = only.subject_id
            self.subject_number = only.subject_number
            self.original_epoch_index = only.original_epoch_index
            self.is_flagged = only.is_flagged
        else:
            self.x = np.ascontiguousarray(
                np.concatenate([item.x for item in loaded], axis=0)
            )
            self.y = np.ascontiguousarray(
                np.concatenate([item.y for item in loaded], axis=0)
            )
            self.subject_id = np.concatenate([item.subject_id for item in loaded])
            self.subject_number = np.ascontiguousarray(
                np.concatenate([item.subject_number for item in loaded])
            )
            self.original_epoch_index = np.ascontiguousarray(
                np.concatenate([item.original_epoch_index for item in loaded])
            )
            self.is_flagged = np.ascontiguousarray(
                np.concatenate([item.is_flagged for item in loaded])
            )
        self.source_paths = tuple(str(item.path.resolve()) for item in loaded)

        if self.x.shape != (len(self.y), EXPECTED_CHANNELS, EXPECTED_TIMES):
            raise AssertionError(f"拼接后的 EEG 形状异常：{self.x.shape}")
        if not np.isfinite(self.x).all():
            raise ValueError("拼接后的 EEG 包含 NaN/Inf")
        if not np.all((0 <= self.y) & (self.y < EXPECTED_CLASSES)):
            raise ValueError("标签必须位于 0..4")

    def _load_source(self, source: _Source) -> _LoadedSource:
        path = source.path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            required = {self.input_field, "y"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"{path} 缺少字段：{missing}")

            x = np.array(data[self.input_field], dtype=np.float32, copy=True, order="C")
            y = np.asarray(data["y"], dtype=np.int64)
            if x.ndim != 3 or x.shape[1:] != (EXPECTED_CHANNELS, EXPECTED_TIMES):
                raise ValueError(f"{path.name}/{self.input_field} 形状异常：{x.shape}")
            if y.shape != (x.shape[0],):
                raise ValueError(f"{path.name} 的 X 与 y 数量不一致")
            if not np.isfinite(x).all():
                raise ValueError(f"{path.name}/{self.input_field} 包含 NaN/Inf")

            if self._mean is not None and self._std is not None:
                x -= self._mean
                x /= self._std

            n_epochs = x.shape[0]
            if "subject_id" in data.files:
                subject_id = np.asarray(data["subject_id"], dtype="U6")
            elif source.subject_id is not None:
                subject_id = np.full(n_epochs, source.subject_id, dtype="U6")
            else:
                raise ValueError(f"{path.name} 没有 subject_id，数据源也未提供")

            if "subject_number" in data.files:
                subject_number = np.asarray(data["subject_number"], dtype=np.int16)
            else:
                subject_number = np.asarray(
                    [int(value.split("-")[-1]) for value in subject_id], dtype=np.int16
                )

            if "original_epoch_index" in data.files:
                original_epoch_index = np.asarray(
                    data["original_epoch_index"], dtype=np.int64
                )
            elif "kept_original_epoch_indices" in data.files:
                original_epoch_index = np.asarray(
                    data["kept_original_epoch_indices"], dtype=np.int64
                )
            else:
                original_epoch_index = np.arange(n_epochs, dtype=np.int64)

            if "is_flagged" in data.files:
                is_flagged = np.asarray(data["is_flagged"], dtype=bool)
            elif "flagged_original_epoch_indices" in data.files:
                flagged = np.asarray(data["flagged_original_epoch_indices"], dtype=np.int64)
                is_flagged = np.isin(original_epoch_index, flagged)
            else:
                is_flagged = np.zeros(n_epochs, dtype=bool)

        for name, values in (
            ("subject_id", subject_id),
            ("subject_number", subject_number),
            ("original_epoch_index", original_epoch_index),
            ("is_flagged", is_flagged),
        ):
            if values.shape != (n_epochs,):
                raise ValueError(f"{path.name}/{name} 形状异常：{values.shape}")

        return _LoadedSource(
            path=path,
            x=x,
            y=y,
            subject_id=subject_id,
            subject_number=subject_number,
            original_epoch_index=original_epoch_index,
            is_flagged=is_flagged,
        )

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
class DataLoaderBundle:
    """一个实验协议的三个 DataLoader 及其可审计信息。"""

    protocol: Protocol
    train: DataLoader
    val: DataLoader
    test: DataLoader
    processed_root: Path
    subject: str | None = None
    fold: str | None = None
    held_out_subject: str | None = None

    @property
    def datasets(self) -> dict[str, BCIEpochDataset]:
        return {
            "train": self.train.dataset,
            "val": self.val.dataset,
            "test": self.test.dataset,
        }


def _subject_id(subject: int | str) -> str:
    if isinstance(subject, str):
        value = subject.strip().lower()
        if value.startswith("sub-"):
            value = value[4:]
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError(f"无法识别被试编号：{subject!r}") from exc
    else:
        number = int(subject)
    if not 1 <= number <= 15:
        raise ValueError("被试编号必须位于 1..15")
    return f"sub-{number:02d}"


def _fold_name(fold: int | str) -> str:
    if isinstance(fold, str):
        value = fold.strip().lower()
        if value.startswith("fold_"):
            value = value[5:]
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError(f"无法识别 LOSO fold：{fold!r}") from exc
    else:
        number = int(fold)
    if not 1 <= number <= 15:
        raise ValueError("LOSO fold 必须位于 1..15")
    return f"fold_{number:02d}"


def _load_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["mean"], dtype=np.float32),
            np.asarray(data["std"], dtype=np.float32),
        )


def _worker_init_fn(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(
    dataset: BCIEpochDataset,
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


def create_dataloaders(
    protocol: Protocol,
    *,
    processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
    subject: int | str | None = None,
    fold: int | str | None = None,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    return_metadata: bool = False,
) -> DataLoaderBundle:
    """构建 Within、Mixed 或 LOSO 的 train/val/test DataLoader。

    ``train`` 会确定性打乱，``val/test`` 保持顺序。Windows 下默认
    ``num_workers=0``，避免每个 worker 复制数百 MB 的内存数据。
    """

    protocol = protocol.lower()  # type: ignore[assignment]
    if protocol not in {"within", "mixed", "loso"}:
        raise ValueError("protocol 必须是 within、mixed 或 loso")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if num_workers < 0:
        raise ValueError("num_workers 不能小于 0")
    processed_root = Path(processed_root).resolve()
    within_root = processed_root / "preprocessed_data" / "within_subject"
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    dataset_args: dict[str, dict[str, Any]] = {}
    selected_subject = None
    selected_fold = None
    held_out_subject = None

    if protocol == "within":
        if subject is None:
            raise ValueError("within 协议必须指定 subject")
        selected_subject = _subject_id(subject)
        subject_dir = within_root / selected_subject
        for split in ("train", "val", "test"):
            dataset_args[split] = {
                "sources": [_Source(subject_dir / f"{split}.npz", selected_subject)],
                "input_field": "X",
            }

    elif protocol == "mixed":
        mixed_root = processed_root / "preprocessed_data" / "mixed_subject"
        for split in ("train", "val", "test"):
            dataset_args[split] = {
                "sources": [_Source(mixed_root / f"{split}_pool.npz")],
                "input_field": "X",
            }

    else:
        if fold is None:
            raise ValueError("loso 协议必须指定 fold")
        selected_fold = _fold_name(fold)
        fold_dir = processed_root / "preprocessed_data" / "cross_subject_LOSO" / selected_fold
        manifest_path = fold_dir / "fold_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fold_name") != selected_fold:
            raise ValueError(f"{manifest_path} 的 fold_name 不匹配")
        held_out_subject = str(manifest["held_out_subject"])
        normalization = manifest["normalization"]
        if normalization.get("input_field") != "X_volts":
            raise ValueError("LOSO manifest 必须指定 X_volts 作为归一化输入")
        if normalization.get("excluded_subject") != held_out_subject:
            raise ValueError("LOSO normalizer 未排除留出被试")
        normalizer = _load_normalizer(fold_dir / normalization["file"])
        for split in ("train", "val", "test"):
            entries = manifest["splits"][split]["sources"]
            sources = [
                _Source(
                    (fold_dir / entry["path"]).resolve(),
                    str(entry["subject_id"]),
                )
                for entry in entries
            ]
            dataset_args[split] = {
                "sources": sources,
                "input_field": "X_volts",
                "normalizer": normalizer,
            }

    datasets = {
        split: BCIEpochDataset(return_metadata=return_metadata, **dataset_args[split])
        for split in ("train", "val", "test")
    }

    if protocol == "loso":
        if held_out_subject in datasets["train"].subject_ids:
            raise AssertionError("LOSO 留出被试泄漏到了训练集")
        if held_out_subject in datasets["val"].subject_ids:
            raise AssertionError("LOSO 留出被试泄漏到了验证集")
        if datasets["test"].subject_ids != (held_out_subject,):
            raise AssertionError("LOSO 测试集不是唯一的留出被试")

    return DataLoaderBundle(
        protocol=protocol,  # type: ignore[arg-type]
        train=_make_loader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            seed=seed, num_workers=num_workers, pin_memory=pin_memory,
        ),
        val=_make_loader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            seed=seed + 1, num_workers=num_workers, pin_memory=pin_memory,
        ),
        test=_make_loader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            seed=seed + 2, num_workers=num_workers, pin_memory=pin_memory,
        ),
        processed_root=processed_root,
        subject=selected_subject,
        fold=selected_fold,
        held_out_subject=held_out_subject,
    )


def _summary(bundle: DataLoaderBundle) -> dict[str, Any]:
    splits = {}
    for split, dataset in bundle.datasets.items():
        first = next(iter(getattr(bundle, split)))
        x, y = first[:2]
        splits[split] = {
            "epochs": len(dataset),
            "batch_shape": list(x.shape),
            "batch_x_dtype": str(x.dtype),
            "batch_y_dtype": str(y.dtype),
            "class_counts": dataset.class_counts.astype(int).tolist(),
            "subjects": list(dataset.subject_ids),
            "input_field": dataset.input_field,
            "normalizer_applied_on_load": dataset.normalized_on_load,
            "flagged_epochs": int(dataset.is_flagged.sum()),
        }
    return {
        "protocol": bundle.protocol,
        "subject": bundle.subject,
        "fold": bundle.fold,
        "held_out_subject": bundle.held_out_subject,
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("within", "mixed", "loso"), required=True)
    parser.add_argument("--subject", help="within 被试，例如 1 或 sub-01")
    parser.add_argument("--fold", help="LOSO 折，例如 1 或 fold_01")
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    bundle = create_dataloaders(
        args.protocol,
        processed_root=args.processed_root,
        subject=args.subject,
        fold=args.fold,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(json.dumps(_summary(bundle), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
