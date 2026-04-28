"""
KNN-MPI Beta v2 — Vectorización NumPy del cómputo local.

Cambios respecto a v1:
  1. Distancias por bloques con la identidad ||a-b||² = ||a||² + ||b||² - 2·a·b
     → reemplaza dos `for` Python por un BLAS GEMM (matriz × matriz).
  2. argpartition en lugar de argsort completo: O(n) vs O(n log n) para top-k.
  3. Mayoría vectorizada con np.add.at sobre matriz one-hot de votos.

La estructura MPI (bcast/scatter/gather) se mantiene IDÉNTICA a v1 — solo
cambia el cómputo local. Esto aísla el efecto de la vectorización y nos
deja medir limpiamente el cambio en t_comp (esperado ~30-100×).

Importante para el inciso (d): los FLOPs reportables siguen siendo los de
la fórmula clásica (3d+1) · n_tr · n_te aunque internamente usemos GEMM,
porque la consigna pide "basada en la fórmula de la distancia euclidiana".

Uso:
    mpirun -n 4 python src/knn_mpi_v2.py
    mpirun -n 4 python src/knn_mpi_v2.py --n 10000   # escalar dataset
"""
import numpy as np
from mpi4py import MPI
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits


def knn_predict_batch(X_test_local, X_train, y_train, k, n_classes=10):
    """
    Predicción KNN vectorizada para un bloque completo de puntos de testeo.

    Parameters
    ----------
    X_test_local : (n_local, d) float64
        Bloque de puntos a clasificar asignado a este rank.
    X_train : (n_train, d) float64
        Conjunto de entrenamiento (replicado en todos los ranks).
    y_train : (n_train,) int64
        Etiquetas de entrenamiento.
    k : int
        Número de vecinos.
    n_classes : int
        Número de clases (10 para digits).

    Returns
    -------
    predictions : (n_local,) int64
    """
    # ─── Distancias al cuadrado por GEMM ────────────────────────────
    # ||a-b||² = ||a||² + ||b||² - 2·a·b
    # No tomamos sqrt: preserva el orden y ahorra n_test*n_train operaciones.
    # El sqrt se cuenta en los FLOPs teóricos del reporte, no en el cómputo real.
    test_sq = np.sum(X_test_local ** 2, axis=1, keepdims=True)   # (n_local, 1)
    train_sq = np.sum(X_train ** 2, axis=1)                       # (n_train,)
    cross = X_test_local @ X_train.T                              # (n_local, n_train)  ← BLAS
    dists_sq = test_sq + train_sq - 2.0 * cross                   # (n_local, n_train)

    # ─── Top-k vecinos con argpartition ─────────────────────────────
    # argpartition es O(n_train) por fila; argsort sería O(n_train · log n_train).
    # Para k=3 y n_train grande, esta diferencia es notable.
    k_indices = np.argpartition(dists_sq, k, axis=1)[:, :k]       # (n_local, k)
    k_labels = y_train[k_indices]                                 # (n_local, k)

    # ─── Voto mayoritario vectorizado ───────────────────────────────
    # Construimos matriz de votos (n_local × n_classes) y tomamos argmax.
    # np.add.at hace incremento "por dispersión" (scatter-add atómico).
    votes = np.zeros((len(k_labels), n_classes), dtype=np.int32)
    rows = np.arange(len(k_labels))[:, None]
    np.add.at(votes, (rows, k_labels), 1)
    predictions = votes.argmax(axis=1).astype(np.int64)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None,
                        help='Tamaño total del dataset (None = original 1797)')
    parser.add_argument('--k', type=int, default=3, help='Número de vecinos')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ─────────────────────────────────────────────────────────────────
    # PASO 1: rank 0 carga datos
    # ─────────────────────────────────────────────────────────────────
    if rank == 0:
        X_train, X_test, y_train, y_test = load_scaled_digits(args.n)
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]
        print(f"[rank 0] Dataset: n_train={n_train}, n_test={n_test}, "
              f"features={n_features}, p={size}, k={args.k}")
    else:
        X_train = y_train = X_test = y_test = None
        n_train = n_features = n_test = None

    comm.Barrier()
    t_start = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 2: BCAST entrenamiento  →  O(log(p) · (α + n_tr·β))
    # ─────────────────────────────────────────────────────────────────
    t_bcast_start = MPI.Wtime()

    n_train = comm.bcast(n_train, root=0)
    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)

    if rank != 0:
        X_train = np.empty((n_train, n_features), dtype=np.float64)
        y_train = np.empty(n_train, dtype=np.int64)

    comm.Bcast(X_train, root=0)
    comm.Bcast(y_train, root=0)

    t_bcast_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 3: SCATTERV testeo  →  O(p · (α + n_te/p · β))
    # ─────────────────────────────────────────────────────────────────
    t_scatter_start = MPI.Wtime()

    counts = np.array([n_test // size + (1 if i < n_test % size else 0)
                       for i in range(size)])
    displs = np.array([sum(counts[:i]) for i in range(size)])
    local_n = counts[rank]

    sendcounts_X = counts * n_features
    displs_X = displs * n_features

    X_test_local = np.empty((local_n, n_features), dtype=np.float64)
    y_test_local = np.empty(local_n, dtype=np.int64)

    comm.Scatterv([X_test if rank == 0 else None, sendcounts_X, displs_X, MPI.DOUBLE],
                  X_test_local, root=0)
    comm.Scatterv([y_test if rank == 0 else None, counts, displs, MPI.LONG],
                  y_test_local, root=0)

    t_scatter_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 4: CÓMPUTO LOCAL VECTORIZADO  →  O(n_tr · n_te / p)
    # ─────────────────────────────────────────────────────────────────
    t_comp_start = MPI.Wtime()

    y_pred_local = knn_predict_batch(X_test_local, X_train, y_train, args.k)

    t_comp_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 5: GATHERV predicciones  →  O(p · (α + k·β))
    # ─────────────────────────────────────────────────────────────────
    t_gather_start = MPI.Wtime()

    if rank == 0:
        y_pred = np.empty(n_test, dtype=np.int64)
    else:
        y_pred = None

    comm.Gatherv(y_pred_local, [y_pred, counts, displs, MPI.LONG], root=0)

    t_gather_end = MPI.Wtime()
    t_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 6: Reporte de rank 0
    # ─────────────────────────────────────────────────────────────────
    # Para análisis de carga, recolectamos t_comp de TODOS los ranks
    # (en v3 esto se hará sistemáticamente).
    t_comp_local = t_comp_end - t_comp_start
    all_t_comp = comm.gather(t_comp_local, root=0)

    if rank == 0:
        accuracy = float(np.mean(y_pred == y_test))

        t_total = t_end - t_start
        t_bcast = t_bcast_end - t_bcast_start
        t_scatter = t_scatter_end - t_scatter_start
        t_comp = max(all_t_comp)        # cuello de botella: el rank más lento
        t_comp_mean = sum(all_t_comp) / len(all_t_comp)
        t_gather = t_gather_end - t_gather_start
        t_comm = t_bcast + t_scatter + t_gather

        # FLOPs según la fórmula del enunciado: 3d + 1 por par de puntos
        d = n_features
        flops_total = (3 * d + 1) * n_train * n_test
        flops_per_sec = flops_total / t_comp if t_comp > 0 else 0

        print(f"\n{'='*60}")
        print(f"KNN-MPI v2 (vectorizado) — p={size}, k={args.k}")
        print(f"{'='*60}")
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Tiempo total:       {t_total:.4f} s")
        print(f"  ├─ Broadcast:     {t_bcast:.4f} s ({100*t_bcast/t_total:.1f}%)")
        print(f"  ├─ Scatter:       {t_scatter:.4f} s ({100*t_scatter/t_total:.1f}%)")
        print(f"  ├─ Cómputo (max): {t_comp:.4f} s ({100*t_comp/t_total:.1f}%)")
        print(f"  ├─ Cómputo (med): {t_comp_mean:.4f} s")
        print(f"  └─ Gather:        {t_gather:.4f} s ({100*t_gather/t_total:.1f}%)")
        print(f"Comunicación total: {t_comm:.4f} s ({100*t_comm/t_total:.1f}%)")
        print(f"FLOPs región paralel.: {flops_total:.3e}")
        print(f"FLOPs/seg (cómputo):   {flops_per_sec:.3e}")


if __name__ == "__main__":
    main()
