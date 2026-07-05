"""
Sequential KNN baseline for the digits project.

This keeps the original scalar Euclidean-distance implementation, but adds a
small CLI and machine-readable outputs so it can be used as a reproducible
baseline for MPI experiments.
"""
from collections import Counter
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits


CSV_FIELDS = [
    "timestamp", "hostname", "git_commit", "command",
    "mode", "n", "p", "k", "rep",
    "n_train", "n_test", "n_features",
    "t_total", "t_bcast", "t_scatter", "t_gather", "t_comm",
    "t_comp", "t_comp_max", "t_comp_mean", "t_comp_min", "t_comp_std",
    "accuracy", "flops", "flops_per_sec",
    "speedup", "efficiency",
]


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def append_rows_to_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def find_repo_root(start):
    path = Path(start).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def get_git_commit():
    repo_root = find_repo_root(__file__)
    if repo_root is None:
        return "unknown"
    try:
        safe_dir = repo_root.as_posix()
        result = subprocess.run(
            ["git", "-c", f"safe.directory={safe_dir}", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_one_rep(X_train, y_train, X_test, y_test, k):
    t_start = time.perf_counter()
    t0 = time.perf_counter()
    y_pred = np.array([knn_predict(x, X_train, y_train, k) for x in X_test], dtype=np.int64)
    t_comp = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start
    accuracy = float(np.mean(y_pred == y_test))
    return y_pred, {
        "t_total": t_total,
        "t_comp": t_comp,
        "t_comm": 0.0,
        "accuracy": accuracy,
    }


def maybe_plot(X_test, y_test, y_pred, output_path=None):
    import matplotlib.pyplot as plt

    n_show = min(10, len(X_test))
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n_show:
            ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
            ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    plt.suptitle("Sample Predictions (Sequential KNN)")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Sequential KNN baseline")
    parser.add_argument("--n", type=int, default=None, help="Total dataset size")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors")
    parser.add_argument("--reps", type=int, default=1, help="Measured repetitions")
    parser.add_argument("--csv", type=str, default=None, help="Append raw rows to CSV")
    parser.add_argument("--json", type=str, default=None, help="Write final row to JSON")
    parser.add_argument("--pred-out", type=str, default=None, help="Write final predictions to a .npy file")
    parser.add_argument("--plot", action="store_true", help="Show or save prediction plot")
    parser.add_argument("--plot-out", type=str, default=None, help="Optional PNG path for --plot")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_scaled_digits(args.n)
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    n_total = args.n if args.n is not None else n_train + n_test
    flops = (3 * n_features + 1) * n_train * n_test
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "git_commit": get_git_commit(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "mode": "sequential",
        "n": n_total,
        "p": 1,
        "k": args.k,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "flops": flops,
    }

    rows = []
    last_pred = None
    print(f"[sequential] n_train={n_train} n_test={n_test} d={n_features} p=1 k={args.k} reps={args.reps}")
    for rep in range(args.reps):
        last_pred, result = run_one_rep(X_train, y_train, X_test, y_test, args.k)
        row = {
            **metadata,
            "rep": rep,
            **result,
            "t_bcast": 0.0,
            "t_scatter": 0.0,
            "t_gather": 0.0,
            "t_comp_max": result["t_comp"],
            "t_comp_mean": result["t_comp"],
            "t_comp_min": result["t_comp"],
            "t_comp_std": 0.0,
            "flops_per_sec": flops / result["t_comp"] if result["t_comp"] > 0 else 0.0,
            "speedup": "",
            "efficiency": "",
        }
        rows.append(row)
        print(
            f"  [rep {rep}] t_total={result['t_total']*1000:7.2f} ms  "
            f"t_comp={result['t_comp']*1000:7.2f} ms  acc={result['accuracy']:.4f}"
        )

    if args.csv:
        append_rows_to_csv(args.csv, rows)
        print(f"  -> {len(rows)} rows appended to {args.csv}")

    if args.json and rows:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows[-1], indent=2), encoding="utf-8")
        print(f"  -> JSON written to {args.json}")

    if args.pred_out and last_pred is not None:
        pred_path = Path(args.pred_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, last_pred)
        print(f"  -> predictions written to {args.pred_out}")

    if args.plot and last_pred is not None:
        maybe_plot(X_test, y_test, last_pred, args.plot_out)


if __name__ == "__main__":
    main()
