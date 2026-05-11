#!/usr/bin/env python
"""Generate standard plots for the KNN MPI report."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Wrote: {path}")


def fit_theory_curve(p_values, measured):
    """Fit T(p) = a / p + b to measured seconds."""
    p_arr = np.asarray(p_values, dtype=float)
    y = np.asarray(measured, dtype=float)
    if len(p_arr) < 2:
        return y
    a, b = np.polyfit(1.0 / p_arr, y, deg=1)
    return a / p_arr + b


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KNN MPI results")
    parser.add_argument("summary_csv", help="summary.csv from aggregate_results.py")
    parser.add_argument("--out-dir", default=None, help="Figure output directory")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent / "figures"
    rows = read_rows(summary_path)
    mpi = [row for row in rows if row.get("mode") == "mpi"]

    if not mpi:
        raise SystemExit("No MPI rows found in summary CSV")

    n_values = sorted({int(num(row, "n")) for row in mpi})
    for n in n_values:
        part = sorted([row for row in mpi if int(num(row, "n")) == n], key=lambda row: num(row, "p"))
        p = [num(row, "p") for row in part]
        total = [num(row, "t_total_median") for row in part]
        comp = [num(row, "t_comp_median") for row in part]
        comm = [num(row, "t_comm_median") for row in part]

        plt.figure()
        plt.plot(p, total, marker="o", label="total measured")
        plt.plot(p, fit_theory_curve(p, total), marker="x", label="fitted a/p + b")
        plt.xlabel("MPI processes p")
        plt.ylabel("seconds")
        plt.title(f"Total time vs p (n={n})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig(out_dir / f"total_time_vs_p_n{n}.png")

        plt.figure()
        plt.plot(p, comp, marker="o", label="computation")
        plt.plot(p, comm, marker="o", label="communication")
        plt.xlabel("MPI processes p")
        plt.ylabel("seconds")
        plt.title(f"Computation and communication vs p (n={n})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig(out_dir / f"comp_comm_vs_p_n{n}.png")

        speedup = [num(row, "speedup") for row in part]
        if not all(np.isnan(speedup)):
            plt.figure()
            plt.plot(p, speedup, marker="o", label="measured speedup")
            plt.plot(p, p, linestyle="--", label="ideal")
            plt.xlabel("MPI processes p")
            plt.ylabel("speedup")
            plt.title(f"Speedup vs p (n={n})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            savefig(out_dir / f"speedup_vs_p_n{n}.png")

        efficiency = [num(row, "efficiency") for row in part]
        if not all(np.isnan(efficiency)):
            plt.figure()
            plt.plot(p, efficiency, marker="o")
            plt.xlabel("MPI processes p")
            plt.ylabel("efficiency")
            plt.title(f"Efficiency vs p (n={n})")
            plt.grid(True, alpha=0.3)
            savefig(out_dir / f"efficiency_vs_p_n{n}.png")

        flop_rates = [num(row, "flops_per_sec_median") for row in part]
        plt.figure()
        plt.plot(p, flop_rates, marker="o")
        plt.xlabel("MPI processes p")
        plt.ylabel("FLOPs/s")
        plt.title(f"FLOPs/s vs p (n={n})")
        plt.grid(True, alpha=0.3)
        savefig(out_dir / f"flops_per_sec_vs_p_n{n}.png")

    p_values = sorted({int(num(row, "p")) for row in mpi})
    for p_value in p_values:
        part = sorted([row for row in mpi if int(num(row, "p")) == p_value], key=lambda row: num(row, "n"))
        if len(part) < 2:
            continue
        plt.figure()
        plt.plot([num(row, "n") for row in part], [num(row, "t_total_median") for row in part], marker="o")
        plt.xlabel("samples n")
        plt.ylabel("seconds")
        plt.title(f"Total time vs n (p={p_value})")
        plt.grid(True, alpha=0.3)
        savefig(out_dir / f"total_time_vs_n_p{p_value}.png")

    acc_path = out_dir / "accuracy_table.csv"
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    with acc_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "n", "p", "k", "runs", "accuracy_mean", "accuracy_min"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    print(f"Wrote: {acc_path}")


if __name__ == "__main__":
    main()
