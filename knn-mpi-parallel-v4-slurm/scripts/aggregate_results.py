#!/usr/bin/env python
"""Aggregate raw KNN CSV rows and compute speedup/efficiency.

MPI rows use MPI p=1 as the baseline when available. That reports parallel
speedup for the same implementation instead of mixing in scalar-vs-vectorized
algorithmic speedup from the sequential baseline.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


GROUP_FIELDS = ["mode", "n", "p", "k", "n_train", "n_test", "n_features"]
SUMMARY_FIELDS = [
    *GROUP_FIELDS,
    "runs",
    "t_total_median", "t_total_mean", "t_total_std",
    "t_comp_median", "t_comm_median",
    "accuracy_mean", "accuracy_min",
    "flops", "flops_per_sec_median",
    "speedup", "efficiency",
]


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def fmt(value: float | str) -> float | str:
    if isinstance(value, str):
        return value
    return f"{value:.12g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate KNN benchmark CSV")
    parser.add_argument("csv", help="Raw result CSV")
    parser.add_argument("--out-dir", default=None, help="Output directory")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    comp_col = "t_comp_max" if "t_comp_max" in fieldnames else "t_comp"
    if comp_col not in fieldnames:
        raise SystemExit("Missing computation time column: expected t_comp_max or t_comp")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in GROUP_FIELDS)
        groups[key].append(row)

    summaries = []
    for key, items in groups.items():
        totals = [f(row, "t_total") for row in items]
        comps = [f(row, comp_col) for row in items]
        comms = [f(row, "t_comm") for row in items]
        accs = [f(row, "accuracy") for row in items]
        flop_rates = [f(row, "flops_per_sec") for row in items]
        summary = dict(zip(GROUP_FIELDS, key))
        summary.update({
            "runs": len(items),
            "t_total_median": median(totals),
            "t_total_mean": mean(totals),
            "t_total_std": stdev(totals) if len(totals) > 1 else 0.0,
            "t_comp_median": median(comps),
            "t_comm_median": median(comms),
            "accuracy_mean": mean(accs),
            "accuracy_min": min(accs),
            "flops": max(f(row, "flops") for row in items),
            "flops_per_sec_median": median(flop_rates),
            "speedup": "",
            "efficiency": "",
        })
        summaries.append(summary)

    sequential_baselines: dict[tuple[int, int], float] = {}
    mpi_baselines: dict[tuple[int, int], float] = {}
    for row in summaries:
        n = int(float(row["n"]))
        k = int(float(row["k"]))
        p = int(float(row["p"]))
        mode = row["mode"]
        if mode == "sequential":
            sequential_baselines[(n, k)] = float(row["t_total_median"])
        elif mode == "mpi" and p == 1:
            mpi_baselines[(n, k)] = float(row["t_total_median"])

    for row in summaries:
        n = int(float(row["n"]))
        k = int(float(row["k"]))
        p = int(float(row["p"]))
        if row["mode"] == "mpi":
            base = mpi_baselines.get((n, k), sequential_baselines.get((n, k)))
        else:
            base = sequential_baselines.get((n, k))
        total = float(row["t_total_median"])
        if base and total > 0:
            row["speedup"] = base / total
            row["efficiency"] = (base / total) / p

    raw_out = out_dir / "raw_with_speedup.csv"
    summary_out = out_dir / "summary.csv"

    with raw_out.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_out.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in sorted(summaries, key=lambda r: (r["mode"], float(r["n"]), float(r["p"]))):
            writer.writerow({key: fmt(row[key]) for key in SUMMARY_FIELDS})

    print(f"Raw rows: {len(rows)}")
    print(f"Summary rows: {len(summaries)}")
    print(f"Wrote: {raw_out}")
    print(f"Wrote: {summary_out}")


if __name__ == "__main__":
    main()
