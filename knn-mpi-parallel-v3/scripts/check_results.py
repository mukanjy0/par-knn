#!/usr/bin/env python
"""Lightweight validation for raw or aggregated KNN result CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED = {
    "n", "p", "k", "n_train", "n_test", "n_features",
    "t_total", "t_comm", "accuracy", "flops", "flops_per_sec",
}


def to_float(row: dict[str, str], col: str) -> float:
    try:
        return float(row[col])
    except Exception as exc:
        raise SystemExit(f"Invalid numeric value for {col}: {row.get(col)!r}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Check KNN result CSV")
    parser.add_argument("csv", help="CSV path")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED - fieldnames)
        if missing:
            raise SystemExit(f"Missing columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise SystemExit(f"No data rows found in {path}")

    accuracies = []
    for row in rows:
        t_total = to_float(row, "t_total")
        t_comm = to_float(row, "t_comm")
        accuracy = to_float(row, "accuracy")
        flops = to_float(row, "flops")
        flops_per_sec = to_float(row, "flops_per_sec")
        if t_total < 0 or t_comm < 0:
            raise SystemExit("Found negative timing values")
        if not 0 <= accuracy <= 1:
            raise SystemExit("Found accuracy outside [0, 1]")
        if flops <= 0:
            raise SystemExit("Found non-positive FLOP count")
        if flops_per_sec < 0:
            raise SystemExit("Found negative FLOPs/s")
        accuracies.append(accuracy)

    print(f"OK: {path}")
    print(f"Rows: {len(rows)}")
    print(f"Columns: {', '.join(reader.fieldnames or [])}")
    print(f"Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}")


if __name__ == "__main__":
    main()
