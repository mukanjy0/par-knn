"""
KNN-MPI Beta v3 — Versión instrumentada para benchmarks reproducibles.

Cambios respecto a v2:
  1. PINNING DE BLAS A 1 THREAD POR PROCESO.
     Resuelve la oversubscription detectada en los datos de v2: con BLAS
     multi-threaded + p procesos MPI, teníamos p×8 threads en 8 cores y
     varianza descontrolada. Ahora cada rank tiene su thread BLAS dedicado.
     Las variables de entorno se fijan ANTES de importar numpy.

  2. SALIDA A CSV con todas las métricas necesarias para el análisis.
     Una fila por repetición. El notebook de análisis lee este CSV directamente.

  3. REPETICIONES INTERNAS con warm-up.
     Un solo `mpirun` corre N+1 reps (la primera se descarta como warm-up,
     que resuelve el efecto de cold-start de BLAS visto en v2 a n=40000).

  4. BARRIERS antes de cada fase para mediciones consistentes.

  5. ESTADÍSTICAS POR RANK del cómputo (max, mean, min, std) para detectar
     desbalance de carga.

Uso (una sola corrida):
    mpirun -n 4 python src/knn_mpi_v3.py --n 10000 --reps 5

Uso (con CSV append para benchmark agregado):
    mpirun -n 4 python src/knn_mpi_v3.py --n 10000 --reps 5 \\
        --csv experiments/results/v3_results.csv

Validación: accuracy debe coincidir con v1 y v2 para el mismo n.
"""
# ─────────────────────────────────────────────────────────────────────────
# CRÍTICO: pinning de BLAS antes de importar numpy.
# Esto previene el oversubscription detectado en v2.
# ─────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
from mpi4py import MPI
import argparse
import csv
import json
import sys
from pathlib import Path
import socket
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits
from utils.knn_kernel import knn_predict_batch


CSV_FIELDS = [
    "timestamp", "hostname", "git_commit", "command",
    "mode",
    "n", "p", "k", "rep",
    "n_train", "n_test", "n_features",
    "t_total", "t_bcast", "t_scatter", "t_gather", "t_comm",
    "t_comp",
    "t_comp_max", "t_comp_mean", "t_comp_min", "t_comp_std",
    "accuracy", "flops", "flops_per_sec",
    "speedup", "efficiency",
]


def run_one_rep(comm, X_train, y_train, X_test, y_test,
                n_train, n_features, n_test, k):
    """
    Ejecuta una iteración completa del pipeline paralelo y retorna métricas
    en el rank 0 (None en los demás).
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    t_start = MPI.Wtime()

    # ─── BCAST entrenamiento ────────────────────────────────────────
    comm.Barrier()
    t0 = MPI.Wtime()
    if rank != 0:
        X_train = np.empty((n_train, n_features), dtype=np.float64)
        y_train = np.empty(n_train, dtype=np.int64)
    comm.Bcast(X_train, root=0)
    comm.Bcast(y_train, root=0)
    t_bcast = MPI.Wtime() - t0

    # ─── SCATTERV testeo ────────────────────────────────────────────
    comm.Barrier()
    t0 = MPI.Wtime()
    counts = np.array([n_test // size + (1 if i < n_test % size else 0)
                       for i in range(size)], dtype=np.int32)
    displs = np.array([int(np.sum(counts[:i])) for i in range(size)], dtype=np.int32)
    local_n = int(counts[rank])

    X_test_local = np.empty((local_n, n_features), dtype=np.float64)
    y_test_local = np.empty(local_n, dtype=np.int64)

    comm.Scatterv(
        [X_test if rank == 0 else None, counts * n_features, displs * n_features, MPI.DOUBLE],
        X_test_local, root=0
    )
    comm.Scatterv(
        [y_test if rank == 0 else None, counts, displs, MPI.LONG],
        y_test_local, root=0
    )
    t_scatter = MPI.Wtime() - t0

    # ─── CÓMPUTO LOCAL ──────────────────────────────────────────────
    comm.Barrier()
    t0 = MPI.Wtime()
    y_pred_local = knn_predict_batch(X_test_local, X_train, y_train, k)
    t_comp_local = MPI.Wtime() - t0

    # ─── GATHERV predicciones ───────────────────────────────────────
    comm.Barrier()
    t0 = MPI.Wtime()
    if rank == 0:
        y_pred = np.empty(n_test, dtype=np.int64)
    else:
        y_pred = None
    comm.Gatherv(y_pred_local, [y_pred, counts, displs, MPI.LONG], root=0)
    t_gather = MPI.Wtime() - t0

    comm.Barrier()
    t_total = MPI.Wtime() - t_start

    # ─── Recolección de t_comp por rank ─────────────────────────────
    all_t_comp = comm.gather(t_comp_local, root=0)

    if rank == 0:
        t_comp_arr = np.array(all_t_comp)
        accuracy = float(np.mean(y_pred == y_test))
        return {
            "t_total": t_total,
            "t_bcast": t_bcast,
            "t_scatter": t_scatter,
            "t_gather": t_gather,
            "t_comp_max": float(t_comp_arr.max()),
            "t_comp_mean": float(t_comp_arr.mean()),
            "t_comp_min": float(t_comp_arr.min()),
            "t_comp_std": float(t_comp_arr.std()),
            "accuracy": accuracy,
        }
    return None


def append_rows_to_csv(csv_path, rows):
    """Añade filas al CSV; escribe encabezado solo si el archivo es nuevo."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def main():
    parser = argparse.ArgumentParser(description="KNN-MPI v3 (instrumentado)")
    parser.add_argument("--n", type=int, default=None, help="Tamaño total del dataset")
    parser.add_argument("--k", type=int, default=3, help="Número de vecinos")
    parser.add_argument("--reps", type=int, default=5, help="Repeticiones medidas")
    parser.add_argument("--no-warmup", action="store_true",
                        help="No descartar la primera repetición")
    parser.add_argument("--csv", type=str, default=None,
                        help="Ruta del CSV (modo append). Si no se da, solo imprime.")
    parser.add_argument("--json", type=str, default=None,
                        help="Ruta JSON para escribir la última repetición medida.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ─── Carga de datos en rank 0 ───────────────────────────────────
    if rank == 0:
        X_train, X_test, y_train, y_test = load_scaled_digits(args.n)
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": socket.gethostname(),
            "git_commit": get_git_commit(),
            "command": " ".join([Path(sys.executable).name, *sys.argv]),
            "mode": "mpi",
        }
        print(f"[rank 0] n_train={n_train} n_test={n_test} d={n_features} "
              f"p={size} k={args.k} reps={args.reps} "
              f"warmup={'no' if args.no_warmup else 'sí'}")
    else:
        X_train = y_train = X_test = y_test = None
        n_train = n_features = n_test = None
        metadata = None

    n_train = comm.bcast(n_train, root=0)
    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)

    # ─── Loop de repeticiones ───────────────────────────────────────
    total_reps = args.reps + (0 if args.no_warmup else 1)
    rows_for_csv = []

    for rep_idx in range(total_reps):
        result = run_one_rep(
            comm, X_train, y_train, X_test, y_test,
            n_train, n_features, n_test, args.k
        )

        is_warmup = (not args.no_warmup) and rep_idx == 0
        measured_idx = rep_idx if args.no_warmup else rep_idx - 1

        if rank == 0:
            tag = "[warmup]" if is_warmup else f"[rep {measured_idx}]"
            t_comm = result["t_bcast"] + result["t_scatter"] + result["t_gather"]
            print(f"  {tag} t_total={result['t_total']*1000:7.2f} ms  "
                  f"t_comp_max={result['t_comp_max']*1000:7.2f} ms  "
                  f"t_comm={t_comm*1000:6.2f} ms  acc={result['accuracy']:.4f}")

            if not is_warmup:
                # FLOPs según la fórmula del enunciado: (3d + 1) por par
                d = n_features
                flops = (3 * d + 1) * n_train * n_test
                flops_per_sec = flops / result["t_comp_max"] if result["t_comp_max"] > 0 else 0

                rows_for_csv.append({
                    **metadata,
                    "n": args.n if args.n else n_train + n_test,
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
                    "t_comp": result["t_comp_max"],
                    "t_comp_max": result["t_comp_max"],
                    "t_comp_mean": result["t_comp_mean"],
                    "t_comp_min": result["t_comp_min"],
                    "t_comp_std": result["t_comp_std"],
                    "accuracy": result["accuracy"],
                    "flops": flops,
                    "flops_per_sec": flops_per_sec,
                    "speedup": "",
                    "efficiency": "",
                })

    # ─── Escritura final del CSV ────────────────────────────────────
    if rank == 0 and args.csv and rows_for_csv:
        append_rows_to_csv(args.csv, rows_for_csv)
        print(f"  → {len(rows_for_csv)} filas añadidas a {args.csv}")

    if rank == 0 and args.json and rows_for_csv:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rows_for_csv[-1], indent=2), encoding="utf-8")
        print(f"  → JSON escrito en {args.json}")


if __name__ == "__main__":
    main()
