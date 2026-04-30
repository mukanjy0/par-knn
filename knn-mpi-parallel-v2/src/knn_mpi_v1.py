"""
KNN-MPI Beta v1 — Paralelización básica siguiendo el DAG del enunciado.

Estrategia (mapeo al DAG):
  1. Rank 0 carga datos                          → "Initial parameters"
  2. comm.bcast(X_train, y_train) a todos        → O(log(p)·(α + n_tr·β))
  3. comm.Scatterv(X_test) en chunks por rank    → O(p·(α + n_te/p·β))
  4. Cada rank: distancias + sort + k-vecinos    → O(n_tr · n_te / p)  [paralelo]
  5. comm.Gatherv(y_pred_local) hacia rank 0     → O(p·(α + k·β))
  6. Rank 0 imprime accuracy y tiempo

Modelo PRAM: CREW
  - Concurrent Read: todos los procesos leen el mismo X_train (replicado vía bcast)
  - Exclusive Write: cada proceso escribe sus predicciones en posiciones disjuntas

Uso:
    mpirun -n 4 python src/knn_mpi_v1.py

Validación: la accuracy debe ser idéntica al secuencial (~0.9861 para k=3).
"""
import numpy as np
from mpi4py import MPI
from collections import Counter
import sys
import os

# Permitir importar desde src/utils/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits


def euclidean_distance(a, b):
    """Distancia euclidiana entre dos vectores. Misma fórmula que el secuencial."""
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    """Predice la etiqueta de un punto usando KNN (versión escalar, igual que el original)."""
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    k = 3
    n_target = None  # None = dataset original. Cambiar para experimentos de escalabilidad.

    # ─────────────────────────────────────────────────────────────────
    # PASO 1: rank 0 carga los datos
    # ─────────────────────────────────────────────────────────────────
    if rank == 0:
        X_train, X_test, y_train, y_test = load_scaled_digits(n_target)
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]
        print(f"[rank 0] Dataset cargado: n_train={n_train}, n_test={n_test}, "
              f"features={n_features}, p={size}")
    else:
        X_train = y_train = X_test = y_test = None
        n_train = n_features = n_test = None

    comm.Barrier()
    t_start = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 2: BCAST del entrenamiento (todos los procesos lo necesitan)
    # ─────────────────────────────────────────────────────────────────
    t_bcast_start = MPI.Wtime()

    n_train = comm.bcast(n_train, root=0)
    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)

    if rank != 0:
        X_train = np.empty((n_train, n_features), dtype=np.float64)
        y_train = np.empty(n_train, dtype=np.int64)

    # Bcast de buffers grandes con la versión optimizada (mayúscula)
    comm.Bcast(X_train, root=0)
    comm.Bcast(y_train, root=0)

    t_bcast_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 3: SCATTERV del testeo (cada rank recibe una fracción)
    # Usamos Scatterv porque n_test puede no ser divisible entre size
    # ─────────────────────────────────────────────────────────────────
    t_scatter_start = MPI.Wtime()

    # Calcular cuántos puntos recibe cada rank (distribución balanceada)
    counts = np.array([n_test // size + (1 if i < n_test % size else 0)
                       for i in range(size)])
    displs = np.array([sum(counts[:i]) for i in range(size)])
    local_n = counts[rank]

    # Para Scatterv necesitamos contar elementos (no filas), así que multiplicamos por n_features
    sendcounts_X = counts * n_features
    displs_X = displs * n_features

    X_test_local = np.empty((local_n, n_features), dtype=np.float64)
    y_test_local = np.empty(local_n, dtype=np.int64)

    if rank == 0:
        X_test_send = X_test
        y_test_send = y_test
    else:
        X_test_send = None
        y_test_send = None

    comm.Scatterv([X_test_send, sendcounts_X, displs_X, MPI.DOUBLE],
                  X_test_local, root=0)
    comm.Scatterv([y_test_send, counts, displs, MPI.LONG],
                  y_test_local, root=0)

    t_scatter_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 4: CÓMPUTO LOCAL — cada rank predice su fracción
    # ─────────────────────────────────────────────────────────────────
    t_comp_start = MPI.Wtime()

    y_pred_local = np.array(
        [knn_predict(x, X_train, y_train, k) for x in X_test_local],
        dtype=np.int64
    )

    t_comp_end = MPI.Wtime()

    # ─────────────────────────────────────────────────────────────────
    # PASO 5: GATHERV de las predicciones hacia rank 0
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
    # PASO 6: Rank 0 reporta resultados
    # ─────────────────────────────────────────────────────────────────
    if rank == 0:
        accuracy = np.mean(y_pred == y_test)

        t_total = t_end - t_start
        t_bcast = t_bcast_end - t_bcast_start
        t_scatter = t_scatter_end - t_scatter_start
        t_comp = t_comp_end - t_comp_start
        t_gather = t_gather_end - t_gather_start
        t_comm = t_bcast + t_scatter + t_gather

        print(f"\n{'='*60}")
        print(f"KNN-MPI v1 — p={size}, k={k}")
        print(f"{'='*60}")
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Tiempo total:       {t_total:.4f} s")
        print(f"  ├─ Broadcast:     {t_bcast:.4f} s ({100*t_bcast/t_total:.1f}%)")
        print(f"  ├─ Scatter:       {t_scatter:.4f} s ({100*t_scatter/t_total:.1f}%)")
        print(f"  ├─ Cómputo:       {t_comp:.4f} s ({100*t_comp/t_total:.1f}%)")
        print(f"  └─ Gather:        {t_gather:.4f} s ({100*t_gather/t_total:.1f}%)")
        print(f"Comunicación total: {t_comm:.4f} s ({100*t_comm/t_total:.1f}%)")


if __name__ == "__main__":
    main()
