#!/usr/bin/env python3
"""Verify subset-only normalization and few-shot DataLoader invariants."""

from __future__ import annotations

import json

import numpy as np
import torch

from .bci2020_cbramod_few_shot import create_few_shot_dataloaders
from .bci2020_cbramod_subject_zscore import (
    create_subject_zscore_within_dataloaders,
)


def main() -> int:
    few = create_few_shot_dataloaders(
        "sub-01", 5, 42, batch_size=16, return_metadata=True
    )
    if len(few.train.dataset) != 15:
        raise AssertionError("5% training split must contain 15 trials")
    if len(few.val.dataset) != 50 or len(few.test.dataset) != 50:
        raise AssertionError("validation and test must remain complete")
    if few.train.dataset.class_counts.tolist() != [3, 3, 3, 3, 3]:
        raise AssertionError("5% training split is not class-balanced")

    selected = few.train.dataset.base
    raw_selected = np.asarray(selected.x, dtype=np.float64)
    expected_mean = raw_selected.mean(axis=(0, 2))
    expected_std = raw_selected.std(axis=(0, 2))
    if not np.allclose(few.normalizer.mean[0], expected_mean, rtol=1e-6, atol=1e-6):
        raise AssertionError("normalizer mean was not fitted from selected trials")
    if not np.allclose(few.normalizer.std[0], expected_std, rtol=1e-6, atol=1e-6):
        raise AssertionError("normalizer std was not fitted from selected trials")

    normalized = (
        raw_selected - few.normalizer.mean[0, :, None]
    ) / few.normalizer.std[0, :, None]
    if np.max(np.abs(normalized.mean(axis=(0, 2)))) > 2e-5:
        raise AssertionError("selected normalized train mean is not approximately zero")
    if np.max(np.abs(normalized.std(axis=(0, 2)) - 1.0)) > 2e-5:
        raise AssertionError("selected normalized train std is not approximately one")

    batch = next(iter(few.train))
    x, y, metadata = batch
    if tuple(x.shape[1:]) != (64, 600) or x.dtype != torch.float32:
        raise AssertionError("few-shot batch input is invalid")
    if y.dtype != torch.int64 or not torch.isfinite(x).all():
        raise AssertionError("few-shot batch labels or values are invalid")
    if "dataset_index" not in metadata or "original_epoch_index" not in metadata:
        raise AssertionError("few-shot provenance metadata is incomplete")

    full_few = create_few_shot_dataloaders("sub-01", 100, 42, batch_size=16)
    full_within = create_subject_zscore_within_dataloaders(
        "sub-01", batch_size=16
    )
    if not np.array_equal(
        np.sort(full_few.selected_dataset_indices), np.arange(300)
    ):
        raise AssertionError("100% few-shot subset is not the complete train split")
    if not np.allclose(full_few.normalizer.mean, full_within.normalizer.mean):
        raise AssertionError("100% few-shot and full Within means disagree")
    if not np.allclose(full_few.normalizer.std, full_within.normalizer.std):
        raise AssertionError("100% few-shot and full Within stds disagree")

    report = {
        "status": "PASS",
        "checked_subject": "sub-01",
        "checked_subset": {"ratio_percent": 5, "seed": 42, "trials": 15},
        "validation_trials": 50,
        "test_trials": 50,
        "subset_only_normalization": True,
        "metadata_provenance": True,
        "full_ratio_matches_within_normalizer": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
