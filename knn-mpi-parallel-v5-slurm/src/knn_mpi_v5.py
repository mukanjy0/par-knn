"""
KNN-MPI v5 Slurm-ready benchmark implementation.

This version preserves the clean v5 algorithm:
  1. distribute the training set with MPI Scatterv,
  2. replicate the full test set with MPI Bcast,
  3. compute local partial Top-K candidates,
  4. combine partial Top-K candidates with MPI Reduce and a custom MergeTopK op.

The surrounding CLI and result format mirror the established Slurm/Khipu workflow:
measured repetitions, optional warm-up, CSV append output, JSON output, timing
by phase, FLOPs, reproducible dataset arguments, and prediction dumps for
correctness checks.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import argparse
import csv
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits
from utils.knn_kernel import local_topk_batch, majority_vote, merge_topk


CSV_FIELDS = [
    "timestamp", "hostname", "git_commit", "command",
    "mode", "n", "p", "k", "rep",
    "n_train", "n_test", "n_features",
    "t_total", "t_bcast", "t_scatter", "t_gather", "t_comm",
    "t_comp", "t_comp_max", "t_comp_mean", "t_comp_min", "t_comp_std",
    "accuracy", "flops", "flops_per_sec",
    "speedup", "efficiency",
]


def add_dataset_args(parser):
    parser.add_argument("--n", type=int, default=None, help="Alias for --n-total")
    parser.add_argument("--n-total", type=int, default=None,
                        help="Final total dataset size; uses percentage split")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Exact final train split size")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Exact final test split size")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Final test fraction for --n-total mode")
    parser.add_argument("--orig-test-size", type=float, default=0.2,
                        help="Original holdout fraction before augmentation in fixed-size mode")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for stratified original split")
    parser.add_argument("--augment-seed", type=int, default=123,
                        help="Seed for deterministic augmentation")


def dataset_kwargs_from_args(parser, args):
    if args.n is not None and args.n_total is not None:
        parser.error("--n and --n-total are aliases; use only one")
    return {
        "n_total": args.n_total if args.n_total is not None else args.n,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "test_size": args.test_size,
        "orig_test_size": args.orig_test_size,
        "split_seed": args.split_seed,
        "augment_seed": args.augment_seed,
    }


def reported_n(args, n_train, n_test):
    if args.n_train is not None and args.n_test is not None:
        return args.n_train + args.n_test
    if args.n_total is not None:
        return args.n_total
    if args.n is not None:
        return args.n
    return n_train + n_test


def block_counts_displs(n, size):
    counts = np.array([n // size + (1 if r < n % size else 0)
                       for r in range(size)], dtype=np.int32)
    displs = np.array([int(np.sum(counts[:r])) for r in range(size)], dtype=np.int32)
    return counts, displs


def append_rows_to_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
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


def make_merge_topk_op(k):
    def _merge(invec, inoutvec, datatype):
        a = np.frombuffer(invec, dtype=np.float64).reshape(-1, k, 2)
        b = np.frombuffer(inoutvec, dtype=np.float64).reshape(-1, k, 2)
        merged_dist, merged_idx = merge_topk(a[:, :, 0], a[:, :, 1],
                                             b[:, :, 0], b[:, :, 1], k)
        b[:, :, 0] = merged_dist
        b[:, :, 1] = merged_idx

    return MPI.Op.Create(_merge, commute=True)


def local_topk_test_batched(x_test, local_x, k, batch_size):
    if batch_size <= 0 or batch_size >= x_test.shape[0]:
        return local_topk_batch(x_test, local_x, k)

    n_test = x_test.shape[0]
    topk_dist = np.empty((n_test, k), dtype=np.float64)
    topk_idx = np.empty((n_test, k), dtype=np.int64)
    for start in range(0, n_test, batch_size):
        stop = min(start + batch_size, n_test)
        batch_dist, batch_idx = local_topk_batch(x_test[start:stop], local_x, k)
        topk_dist[start:stop] = batch_dist
        topk_idx[start:stop] = batch_idx
    return topk_dist, topk_idx


def run_one_rep(comm, X_train, y_train, X_test, y_test,
                n_train, n_features, n_test, k, test_batch_size):
    rank = comm.Get_rank()
    size = comm.Get_size()

    counts, displs = block_counts_displs(n_train, size)
    local_n = int(counts[rank])
    offset = int(displs[rank])

    comm.Barrier()
    t_total_start = MPI.Wtime()

    local_x = np.empty((local_n, n_features), dtype=np.float64)
    local_y = np.empty(local_n, dtype=np.int64)
    send_x = np.ascontiguousarray(X_train, dtype=np.float64) if rank == 0 else None
    send_y = np.ascontiguousarray(y_train, dtype=np.int64) if rank == 0 else None

    comm.Barrier()
    t0 = MPI.Wtime()
    comm.Scatterv(
        [send_x, counts * n_features, displs * n_features, MPI.DOUBLE],
        local_x,
        root=0,
    )
    comm.Scatterv(
        [send_y, counts, displs, MPI.INT64_T],
        local_y,
        root=0,
    )
    comm.Barrier()
    t_scatter_local = MPI.Wtime() - t0

    if rank == 0:
        x_test = np.ascontiguousarray(X_test, dtype=np.float64)
    else:
        x_test = np.empty((n_test, n_features), dtype=np.float64)

    comm.Barrier()
    t0 = MPI.Wtime()
    comm.Bcast(x_test, root=0)
    comm.Barrier()
    t_bcast_local = MPI.Wtime() - t0

    comm.Barrier()
    t0 = MPI.Wtime()
    topk_dist, topk_idx_local = local_topk_test_batched(
        x_test, local_x, k, test_batch_size
    )
    topk_idx = np.where(topk_idx_local >= 0, topk_idx_local + offset, topk_idx_local)
    comm.Barrier()
    t_comp_local = MPI.Wtime() - t0

    sendbuf = np.empty((n_test, k, 2), dtype=np.float64)
    sendbuf[:, :, 0] = topk_dist
    sendbuf[:, :, 1] = topk_idx.astype(np.float64)
    recvbuf = np.empty((n_test, k, 2), dtype=np.float64) if rank == 0 else None
    merge_op = make_merge_topk_op(k)

    comm.Barrier()
    t0 = MPI.Wtime()
    comm.Reduce(sendbuf, recvbuf, op=merge_op, root=0)
    comm.Barrier()
    t_reduce_local = MPI.Wtime() - t0
    merge_op.Free()

    t_total_local = MPI.Wtime() - t_total_start

    t_scatter = comm.reduce(t_scatter_local, op=MPI.MAX, root=0)
    t_bcast = comm.reduce(t_bcast_local, op=MPI.MAX, root=0)
    t_reduce = comm.reduce(t_reduce_local, op=MPI.MAX, root=0)
    t_total = comm.reduce(t_total_local, op=MPI.MAX, root=0)
    all_t_comp = comm.gather(t_comp_local, root=0)

    if rank == 0:
        global_dist = recvbuf[:, :, 0]
        global_idx = recvbuf[:, :, 1].astype(np.int64)
        y_pred = majority_vote(y_train[global_idx], global_dist)
        t_comp_arr = np.array(all_t_comp)
        accuracy = float(np.mean(y_pred == y_test))
        return y_pred, {
            "t_total": float(t_total),
            "t_bcast": float(t_bcast),
            "t_scatter": float(t_scatter),
            "t_gather": float(t_reduce),
            "t_comp": float(t_comp_arr.max()),
            "t_comp_max": float(t_comp_arr.max()),
            "t_comp_mean": float(t_comp_arr.mean()),
            "t_comp_min": float(t_comp_arr.min()),
            "t_comp_std": float(t_comp_arr.std()),
            "accuracy": accuracy,
        }

    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="KNN-MPI v5: Scatterv + Bcast + Reduce(custom MergeTopK)"
    )
    add_dataset_args(parser)
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors")
    parser.add_argument("--reps", type=int, default=5, help="Measured repetitions")
    parser.add_argument("--no-warmup", action="store_true", help="Do not discard a warm-up repetition")
    parser.add_argument("--csv", type=str, default=None, help="Append raw benchmark rows to CSV")
    parser.add_argument("--json", type=str, default=None, help="Write final measured row to JSON")
    parser.add_argument("--pred-out", type=str, default=None, help="Write final predictions to a .npy file")
    parser.add_argument("--test-batch-size", type=int, default=0,
                        help="Number of test rows per local Top-K batch; 0 means all at once")
    args = parser.parse_args()
    dataset_kwargs = dataset_kwargs_from_args(parser, args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        X_train, X_test, y_train, y_test = load_scaled_digits(**dataset_kwargs)
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]
        if args.k > n_train:
            raise SystemExit(f"k={args.k} > n_train={n_train}: impossible")
        if n_train >= 2**53:
            raise SystemExit(
                f"n_train={n_train} >= 2**53: indices cannot be packed exactly "
                "in the float64 reduction buffer"
            )
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": socket.gethostname(),
            "git_commit": get_git_commit(),
            "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "mode": "mpi",
        }
        print(
            f"[rank 0] n_train={n_train} n_test={n_test} d={n_features} "
            f"p={size} k={args.k} reps={args.reps} "
            f"warmup={'no' if args.no_warmup else 'yes'} "
            f"test_batch_size={args.test_batch_size}"
        )
    else:
        X_train = y_train = X_test = y_test = None
        n_train = n_features = n_test = None
        metadata = None

    n_train = comm.bcast(n_train, root=0)
    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)

    total_reps = args.reps + (0 if args.no_warmup else 1)
    rows_for_csv = []
    last_pred = None

    for rep_idx in range(total_reps):
        y_pred, result = run_one_rep(
            comm, X_train, y_train, X_test, y_test,
            n_train, n_features, n_test, args.k, args.test_batch_size
        )
        is_warmup = (not args.no_warmup) and rep_idx == 0
        measured_idx = rep_idx if args.no_warmup else rep_idx - 1

        if rank == 0:
            last_pred = y_pred
            t_comm = result["t_bcast"] + result["t_scatter"] + result["t_gather"]
            tag = "[warmup]" if is_warmup else f"[rep {measured_idx}]"
            print(
                f"  {tag} t_total={result['t_total']*1000:7.2f} ms  "
                f"t_comp_max={result['t_comp_max']*1000:7.2f} ms  "
                f"t_comm={t_comm*1000:6.2f} ms  acc={result['accuracy']:.4f}"
            )

            if not is_warmup:
                flops = (3 * n_features + 1) * n_train * n_test
                rows_for_csv.append({
                    **metadata,
                    "n": reported_n(args, n_train, n_test),
                    "p": size,
                    "k": args.k,
                    "rep": measured_idx,
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_features": n_features,
                    "t_total": result["t_total"],
                    "t_bcast": result["t_bcast"],
                    "t_scatter": result["t_scatter"],
                    "t_gather": result["t_gather"],
                    "t_comm": t_comm,
                    "t_comp": result["t_comp"],
                    "t_comp_max": result["t_comp_max"],
                    "t_comp_mean": result["t_comp_mean"],
                    "t_comp_min": result["t_comp_min"],
                    "t_comp_std": result["t_comp_std"],
                    "accuracy": result["accuracy"],
                    "flops": flops,
                    "flops_per_sec": flops / result["t_comp_max"] if result["t_comp_max"] > 0 else 0.0,
                    "speedup": "",
                    "efficiency": "",
                })

    if rank == 0 and args.csv and rows_for_csv:
        append_rows_to_csv(args.csv, rows_for_csv)
        print(f"  -> {len(rows_for_csv)} rows appended to {args.csv}")

    if rank == 0 and args.json and rows_for_csv:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows_for_csv[-1], indent=2), encoding="utf-8")
        print(f"  -> JSON written to {args.json}")

    if rank == 0 and args.pred_out and last_pred is not None:
        pred_path = Path(args.pred_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, last_pred)
        print(f"  -> predictions written to {args.pred_out}")


if __name__ == "__main__":
    main()
