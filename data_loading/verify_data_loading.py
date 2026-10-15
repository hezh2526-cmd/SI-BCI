#!/usr/bin/env python3
"""验证 BCI2020 PyTorch 数据适配、归一化和 LOSO 被试隔离。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from bci2020_data import DEFAULT_PROCESSED_ROOT, create_dataloaders


def _verify_bundle(bundle) -> dict:
    result = {
        "protocol": bundle.protocol,
        "subject": bundle.subject,
        "fold": bundle.fold,
        "held_out_subject": bundle.held_out_subject,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        loader = getattr(bundle, split)
        dataset = loader.dataset
        batch = next(iter(loader))
        x, y = batch[:2]
        if x.ndim != 3 or tuple(x.shape[1:]) != (64, 512):
            raise AssertionError(f"{bundle.protocol}/{split} batch 形状异常：{tuple(x.shape)}")
        if x.dtype != torch.float32 or y.dtype != torch.int64:
            raise AssertionError(f"{bundle.protocol}/{split} Tensor dtype 异常")
        if not torch.isfinite(x).all():
            raise AssertionError(f"{bundle.protocol}/{split} batch 含 NaN/Inf")
        if int(y.min()) < 0 or int(y.max()) > 4:
            raise AssertionError(f"{bundle.protocol}/{split} 标签超出 0..4")
        result["splits"][split] = {
            "epochs": len(dataset),
            "first_batch_shape": list(x.shape),
            "class_counts": dataset.class_counts.astype(int).tolist(),
            "subjects": list(dataset.subject_ids),
            "input_field": dataset.input_field,
            "normalizer_applied_on_load": dataset.normalized_on_load,
            "flagged_epochs": int(dataset.is_flagged.sum()),
            "finite": True,
        }

    train = bundle.train.dataset
    # 训练集归一化按通道、跨 epoch/time 拟合；允许 float32 舍入误差。
    mean = train.x.mean(axis=(0, 2), dtype=np.float64)
    std = train.x.std(axis=(0, 2), dtype=np.float64)
    max_abs_mean = float(np.max(np.abs(mean)))
    max_abs_std_error = float(np.max(np.abs(std - 1.0)))
    result["train_normalization"] = {
        "max_abs_channel_mean": max_abs_mean,
        "max_abs_channel_std_error": max_abs_std_error,
    }
    if max_abs_mean > 2e-4 or max_abs_std_error > 2e-4:
        raise AssertionError(
            f"{bundle.protocol} 训练集归一化异常：mean={max_abs_mean}, std_error={max_abs_std_error}"
        )

    if bundle.protocol in {"within", "mixed"}:
        if any(ds.input_field != "X" or ds.normalized_on_load for ds in bundle.datasets.values()):
            raise AssertionError("Within/Mixed 应直接读取 X，不能再次归一化")
    else:
        if any(ds.input_field != "X_volts" or not ds.normalized_on_load
               for ds in bundle.datasets.values()):
            raise AssertionError("LOSO 应读取 X_volts 并应用 fold normalizer")
        held_out = bundle.held_out_subject
        if held_out in train.subject_ids or held_out in bundle.val.dataset.subject_ids:
            raise AssertionError("LOSO 留出被试进入了 train/val")
        if bundle.test.dataset.subject_ids != (held_out,):
            raise AssertionError("LOSO test 不是留出被试")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--protocols", nargs="+", choices=("within", "mixed", "loso"),
        default=("within", "mixed", "loso"),
    )
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--fold", default="fold_01")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    reports = []
    for protocol in args.protocols:
        bundle = create_dataloaders(
            protocol,
            processed_root=args.processed_root,
            subject=args.subject if protocol == "within" else None,
            fold=args.fold if protocol == "loso" else None,
            batch_size=args.batch_size,
            seed=42,
            num_workers=0,
        )
        reports.append(_verify_bundle(bundle))
        del bundle
        gc.collect()

    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
