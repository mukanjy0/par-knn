"""
KNN-MPI v2 Slurm-ready benchmark implementation.

This version preserves the v2 algorithm:
  1. split the training set across MPI ranks with explicit Send/Recv,
  2. broadcast the full test set with an explicit recursive-doubling tree,
  3. compute local partial Top-K candidates,
  4. reduce partial Top-K candidates with an explicit binary tree.

The surrounding CLI and result format mirror the working v3 Khipu workflow:
measured repetitions, optional warm-up, CSV append output, JSON output, timing
by phase, rank compute statistics, FLOPs, and reproducible metadata.
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
import math
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


TAG_TRAIN_X = 10
TAG_TRAIN_Y = 11
TAG_TEST = 20
TAG_TOPK_DIST = 30
TAG_TOPK_IDX = 31

CSV_FIELDS = [
    "timestamp", "hostname", "git_commit", "command",
    "mode", "n", "p", "k", "rep",
    "n_train", "n_test", "n_features",
    "t_total", "t_bcast", "t_scatter", "t_gather", "t_comm",
    "t_comp", "t_comp_max", "t_comp_mean", "t_comp_min", "t_comp_std",
    "accuracy", "flops", "flops_per_sec",
    "speedup", "efficiency",
]


def block_counts_displs(n, size):
    counts = np.array([n // size + (1 if r < n % size else 0)
                       for r in range(size)], dtype=np.int64)
    displs = np.array([int(np.sum(counts[:r])) for r in range(size)], dtype=np.int64)
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


def scatter_train_explicit(comm, X_train_full, y_train_full, n_train, n_features):
    rank = comm.Get_rank()
    size = comm.Get_size()
    counts, displs = block_counts_displs(n_train, size)
    local_n = int(counts[rank])
    offset = int(displs[rank])

    local_x = np.empty((local_n, n_features), dtype=np.float64)
    local_y = np.empty(local_n, dtype=np.int64)

    if rank == 0:
        local_x[:] = X_train_full[offset:offset + local_n]
        local_y[:] = y_train_full[offset:offset + local_n]
        for r in range(1, size):
            r_lo = int(displs[r])
            r_hi = r_lo + int(counts[r])
            comm.Send(np.ascontiguousarray(X_train_full[r_lo:r_hi]), dest=r, tag=TAG_TRAIN_X)
            comm.Send(np.ascontiguousarray(y_train_full[r_lo:r_hi]), dest=r, tag=TAG_TRAIN_Y)
    else:
        comm.Recv(local_x, source=0, tag=TAG_TRAIN_X)
        comm.Recv(local_y, source=0, tag=TAG_TRAIN_Y)

    return local_x, local_y, offset


def bcast_test_tree(comm, X_test_full, n_test, n_features):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        x_test = X_test_full
    else:
        x_test = np.empty((n_test, n_features), dtype=np.float64)

    n_steps = math.ceil(math.log2(size)) if size > 1 else 0
    for h in range(1, n_steps + 1):
        half = 1 << (h - 1)
        if rank < half:
            dest = rank + half
            if dest < size:
                comm.Send(x_test, dest=dest, tag=TAG_TEST)
        elif half <= rank < (1 << h):
            src = rank - half
            comm.Recv(x_test, source=src, tag=TAG_TEST)

    return x_test


def reduce_topk_tree(comm, topk_dist, topk_idx, n_test, k):
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_steps = math.ceil(math.log2(size)) if size > 1 else 0

    for h in range(1, n_steps + 1):
        block = 1 << h
        half = 1 << (h - 1)
        if rank % block == half:
            parent = rank - half
            comm.Send(topk_dist, dest=parent, tag=TAG_TOPK_DIST)
            comm.Send(topk_idx, dest=parent, tag=TAG_TOPK_IDX)
            break
        if rank % block == 0:
            child = rank + half
            if child < size:
                recv_dist = np.empty((n_test, k), dtype=np.float64)
                recv_idx = np.empty((n_test, k), dtype=np.int64)
                comm.Recv(recv_dist, source=child, tag=TAG_TOPK_DIST)
                comm.Recv(recv_idx, source=child, tag=TAG_TOPK_IDX)
                topk_dist, topk_idx = merge_topk(
                    topk_dist, topk_idx, recv_dist, recv_idx, k
                )

    return topk_dist, topk_idx


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

    comm.Barrier()
    t_total_start = MPI.Wtime()

    comm.Barrier()
    t0 = MPI.Wtime()
    local_x, local_y, offset = scatter_train_explicit(
        comm, X_train, y_train, n_train, n_features
    )
    comm.Barrier()
    t_scatter_local = MPI.Wtime() - t0

    comm.Barrier()
    t0 = MPI.Wtime()
    x_test = bcast_test_tree(comm, X_test, n_test, n_features)
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

    comm.Barrier()
    t0 = MPI.Wtime()
    topk_dist, topk_idx = reduce_topk_tree(comm, topk_dist, topk_idx, n_test, k)
    comm.Barrier()
    t_reduce_local = MPI.Wtime() - t0

    t_total_local = MPI.Wtime() - t_total_start

    t_scatter = comm.reduce(t_scatter_local, op=MPI.MAX, root=0)
    t_bcast = comm.reduce(t_bcast_local, op=MPI.MAX, root=0)
    t_reduce = comm.reduce(t_reduce_local, op=MPI.MAX, root=0)
    t_total = comm.reduce(t_total_local, op=MPI.MAX, root=0)
    all_t_comp = comm.gather(t_comp_local, root=0)

    if rank == 0:
        y_pred = majority_vote(y_train[topk_idx], topk_dist)
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
        description="KNN-MPI v2: training decomposition + explicit Top-K tree reduction"
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
