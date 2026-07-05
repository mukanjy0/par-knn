#!/usr/bin/env python
"""
Valida el desempate determinista del Top-K:
los candidatos se ordenan por (distancia, índice global).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.knn_kernel import local_topk_batch, merge_topk


def assert_equal(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise AssertionError(f"{name}: esperado {expected}, obtenido {actual}")


def main():
    X_test = np.array([[0.0]], dtype=np.float64)
    X_train = np.array([[1.0], [-1.0], [2.0]], dtype=np.float64)
    dist, idx = local_topk_batch(X_test, X_train, k=2)
    assert_equal(idx, np.array([[0, 1]], dtype=np.int64), "local_topk_batch idx")
    assert_equal(dist, np.array([[1.0, 1.0]], dtype=np.float64), "local_topk_batch dist")

    dist_a = np.array([[1.0, 1.0, 2.0]], dtype=np.float64)
    idx_a = np.array([[10, 4, 9]], dtype=np.int64)
    dist_b = np.array([[1.0, 0.5, 1.0]], dtype=np.float64)
    idx_b = np.array([[3, 20, 5]], dtype=np.int64)

    expected_dist = np.array([[0.5, 1.0, 1.0]], dtype=np.float64)
    expected_idx = np.array([[20, 3, 4]], dtype=np.int64)

    merged_dist, merged_idx = merge_topk(dist_a, idx_a, dist_b, idx_b, k=3)
    assert_equal(merged_dist, expected_dist, "merge_topk dist")
    assert_equal(merged_idx, expected_idx, "merge_topk idx")

    rev_dist, rev_idx = merge_topk(dist_b, idx_b, dist_a, idx_a, k=3)
    assert_equal(rev_dist, expected_dist, "merge_topk reversed dist")
    assert_equal(rev_idx, expected_idx, "merge_topk reversed idx")

    print("OK: Top-K tie-breaking is deterministic by (distance, global_index).")


if __name__ == "__main__":
    main()
