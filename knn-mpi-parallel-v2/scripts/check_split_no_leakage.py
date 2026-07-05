#!/usr/bin/env python
"""
Sanity checks for split-first augmentation.

Percentage mode is for accuracy/correctness experiments. Fixed-size mode is for
performance/scaling experiments. In both modes augmentation happens only after
the original sklearn digits train/test split, so train/test origin ids must be
disjoint.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.data_loader import load_scaled_digits


def _assert_reasonable_distribution(labels, name):
    counts = np.bincount(labels, minlength=10)
    if np.any(counts == 0):
        raise AssertionError(f"{name}: missing classes, counts={counts.tolist()}")
    ratio = counts.max() / counts.min()
    if ratio > 2.0:
        raise AssertionError(f"{name}: class imbalance too high, counts={counts.tolist()}")


def _assert_no_leakage(meta):
    train_origins = set(meta["train_origin_ids"])
    test_origins = set(meta["test_origin_ids"])
    overlap = train_origins.intersection(test_origins)
    if overlap:
        preview = sorted(overlap)[:10]
        raise AssertionError(f"origin leakage detected: {preview}")


def check_percentage_mode():
    X_train, X_test, y_train, y_test, meta = load_scaled_digits(
        n_total=10000,
        test_size=0.2,
        return_metadata=True,
    )
    if len(X_train) != 8000 or len(X_test) != 2000:
        raise AssertionError(f"percentage sizes: train={len(X_train)} test={len(X_test)}")
    if X_train.dtype != np.float64 or X_test.dtype != np.float64:
        raise AssertionError("X arrays must be float64")
    if y_train.dtype != np.int64 or y_test.dtype != np.int64:
        raise AssertionError("y arrays must be int64")
    _assert_no_leakage(meta)
    _assert_reasonable_distribution(y_train, "percentage train")
    _assert_reasonable_distribution(y_test, "percentage test")
    print("OK percentage mode: train=8000 test=2000 no leakage")


def check_fixed_mode():
    X_train, X_test, y_train, y_test, meta = load_scaled_digits(
        n_train=25000,
        n_test=5000,
        orig_test_size=0.2,
        return_metadata=True,
    )
    if len(X_train) != 25000 or len(X_test) != 5000:
        raise AssertionError(f"fixed sizes: train={len(X_train)} test={len(X_test)}")
    if X_train.dtype != np.float64 or X_test.dtype != np.float64:
        raise AssertionError("X arrays must be float64")
    if y_train.dtype != np.int64 or y_test.dtype != np.int64:
        raise AssertionError("y arrays must be int64")
    _assert_no_leakage(meta)
    _assert_reasonable_distribution(y_train, "fixed train")
    _assert_reasonable_distribution(y_test, "fixed test")
    print("OK fixed mode: train=25000 test=5000 no leakage")


def main():
    check_percentage_mode()
    check_fixed_mode()


if __name__ == "__main__":
    main()
