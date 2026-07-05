"""
KNN-MPI — Clasificación K-Nearest Neighbors paralela con descomposición por
el conjunto de ENTRENAMIENTO y reducción de Top-K en árbol binario.

El algoritmo
------------
El trabajo dominante de KNN es O(n_te · n_tr · d): para cada punto de prueba hay
que medir su distancia a todos los puntos de entrenamiento. Por eso se reparte el
ENTRENAMIENTO entre los p procesos y se replica el conjunto de prueba:

  Fase 1 — Repartir el entrenamiento.
      El proceso raíz parte X_train en p bloques de ~n_tr/p puntos y envía a cada
      proceso su bloque (con sus etiquetas).

  Fase 2 — Difundir el test.
      Todo el conjunto de prueba se hace llegar a cada proceso mediante un árbol
      de difusión (recursive doubling): el número de procesos que ya tienen el
      test se duplica en cada paso, así que la difusión cuesta log2(p) pasos.

  Fase 3 — Top-K parcial local.
      Cada proceso calcula, para cada punto de prueba, sus K vecinos más cercanos
      DENTRO de su bloque local. El resultado es una lista Top-K *parcial* por
      punto de prueba: las K menores distancias y los índices (globales) de esos
      vecinos.

  Fase 4 — Reducir los Top-K en árbol.
      Las listas Top-K parciales se combinan con una reducción en árbol binario
      (log2(p) pasos) mediante la operación MergeTopK(A, B, K) = los K mejores
      candidatos de A ∪ B. Esto es correcto porque los K vecinos globales más
      cercanos están necesariamente en la unión de los K mejores locales de cada
      proceso; y como MergeTopK es asociativa, puede aplicarse en árbol. La
      reducción termina en el proceso raíz, que tiene ya el Top-K global y emite
      la predicción por voto mayoritario.

Las tres comunicaciones (repartir el train, difundir el test, reducir los Top-K)
se implementan con Send/Recv punto a punto, de modo que los árboles log(p)
quedan explícitos en el código.

Manejo de casos no divisibles
-----------------------------
  - n_tr no múltiplo de p: el bloque se reparte con counts/displs (los primeros
    n_tr % p procesos reciben un punto extra). Cada proceso conoce su tamaño y
    su offset global de forma determinista, sin necesidad de comunicarlos.
  - p no potencia de 2: los árboles usan ceil(log2(p)) pasos y guardas
    (dest < p, child < p), así que funcionan para cualquier p.
  - bloque con < K puntos: local_topk_batch rellena con (+inf, -1); el relleno
    nunca sobrevive al merge global mientras n_tr >= K.

Uso:
    mpirun -n 4 python src/knn_mpi_v4.py --n 10000 --k 3

La accuracy debe coincidir con la versión secuencial para el mismo n y ser
independiente de p (la reducción no cambia el resultado).
"""
# ─────────────────────────────────────────────────────────────────────────
# CRÍTICO: pinning de BLAS antes de importar numpy (evita oversubscription:
# p procesos MPI × varios threads BLAS compitiendo por los mismos cores).
# ─────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import argparse
import math
import sys

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits
from utils.knn_kernel import local_topk_batch, merge_topk, majority_vote


def add_dataset_args(parser):
    parser.add_argument("--n", type=int, default=None,
                        help="Alias historico de --n-total")
    parser.add_argument("--n-total", type=int, default=None,
                        help="Tamano total final; usa split porcentual")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Tamano final exacto del train split")
    parser.add_argument("--n-test", type=int, default=None,
                        help="Tamano final exacto del test split")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proporcion final de test en modo --n-total")
    parser.add_argument("--orig-test-size", type=float, default=0.2,
                        help="Holdout original usado antes de aumentar en modo fijo")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Semilla del split original estratificado")
    parser.add_argument("--augment-seed", type=int, default=123,
                        help="Semilla del aumento deterministico")


def dataset_kwargs_from_args(parser, args):
    if args.n is not None and args.n_total is not None:
        parser.error("--n y --n-total son alias; use solo uno")
    return {
        "n_total": args.n_total if args.n_total is not None else args.n,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "test_size": args.test_size,
        "orig_test_size": args.orig_test_size,
        "split_seed": args.split_seed,
        "augment_seed": args.augment_seed,
    }


# Etiquetas (tags) MPI para distinguir los mensajes de cada fase.
TAG_TRAIN_X = 10
TAG_TRAIN_Y = 11
TAG_TEST = 20
TAG_TOPK_DIST = 30
TAG_TOPK_IDX = 31


def block_counts_displs(n, size):
    """
    Reparto balanceado de `n` elementos entre `size` procesos (maneja resto).
    Los primeros (n % size) procesos reciben un elemento extra.

    Returns
    -------
    counts : (size,) int   nº de elementos por proceso
    displs : (size,) int   offset global del bloque de cada proceso
    """
    counts = np.array([n // size + (1 if r < n % size else 0)
                       for r in range(size)], dtype=np.int64)
    displs = np.array([int(np.sum(counts[:r])) for r in range(size)], dtype=np.int64)
    return counts, displs


def knn_predict_mpi(comm, X_train_full, y_train_full, X_test_full,
                    n_train, n_features, n_test, k):
    """
    Ejecuta el pipeline KNN distribuido. Devuelve el vector de predicciones
    (n_test,) en el proceso raíz (rank 0) y None en los demás.

    Solo el rank 0 trae datos completos en X_train_full / y_train_full /
    X_test_full; en el resto pueden ser None.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Reparto determinista del entrenamiento: todos los procesos calculan el
    # mismo counts/displs, así que cada uno sabe su tamaño y su offset global
    # sin que se los comuniquen.
    counts, displs = block_counts_displs(n_train, size)
    local_n = int(counts[rank])
    offset = int(displs[rank])   # índice global del primer punto del bloque

    # ─── FASE 1: repartir el entrenamiento (scatter explícito) ──────────────
    # El rank 0 envía a cada proceso su bloque [displs[r] : displs[r]+counts[r]]
    # y se queda con el suyo por copia local (no se auto-envía: bloquearía).
    LocalX = np.empty((local_n, n_features), dtype=np.float64)
    LocalY = np.empty(local_n, dtype=np.int64)
    if rank == 0:
        LocalX[:] = X_train_full[offset:offset + local_n]
        LocalY[:] = y_train_full[offset:offset + local_n]
        for r in range(1, size):
            r_lo = int(displs[r])
            r_hi = r_lo + int(counts[r])
            # ascontiguousarray garantiza buffers contiguos para el envío.
            comm.Send(np.ascontiguousarray(X_train_full[r_lo:r_hi]), dest=r, tag=TAG_TRAIN_X)
            comm.Send(np.ascontiguousarray(y_train_full[r_lo:r_hi]), dest=r, tag=TAG_TRAIN_Y)
    else:
        comm.Recv(LocalX, source=0, tag=TAG_TRAIN_X)
        comm.Recv(LocalY, source=0, tag=TAG_TRAIN_Y)

    # ─── FASE 2: difundir el test en ÁRBOL (broadcast explícito) ────────────
    # Recursive doubling: en el paso h, los procesos que YA tienen el test
    # (0 .. half-1) se lo envían a (half .. 2·half-1). El conjunto con el dato
    # se duplica cada paso -> log2(p) pasos.
    if rank == 0:
        X_test = X_test_full          # la raíz ya tiene el test
    else:
        X_test = np.empty((n_test, n_features), dtype=np.float64)

    n_steps = math.ceil(math.log2(size)) if size > 1 else 0
    for h in range(1, n_steps + 1):
        half = 1 << (h - 1)           # 2^(h-1)
        if rank < half:
            dest = rank + half
            if dest < size:           # guarda para p no potencia de 2
                comm.Send(X_test, dest=dest, tag=TAG_TEST)
        elif half <= rank < (1 << h):
            src = rank - half
            comm.Recv(X_test, source=src, tag=TAG_TEST)

    # ─── FASE 3: Top-K parcial local ────────────────────────────────────────
    # K vecinos más cercanos de cada punto de prueba DENTRO del bloque local.
    # Pasamos los índices locales a GLOBALES sumando el offset del bloque
    # (los rellenos -1 se dejan intactos).
    topk_dist, topk_idx_local = local_topk_batch(X_test, LocalX, k)
    topk_idx = np.where(topk_idx_local >= 0, topk_idx_local + offset, topk_idx_local)

    # ─── FASE 4: reducir los Top-K en ÁRBOL (reduce explícito) ──────────────
    # Árbol binomial inverso que termina en el rank 0. En el paso h:
    #   - un proceso "hijo" (rank % 2^h == 2^(h-1)) envía su Top-K al padre y sale,
    #   - un proceso "padre" (rank % 2^h == 0) recibe el del hijo y hace MergeTopK.
    for h in range(1, n_steps + 1):
        block = 1 << h                # 2^h
        half = 1 << (h - 1)           # 2^(h-1)
        if rank % block == half:
            parent = rank - half
            comm.Send(topk_dist, dest=parent, tag=TAG_TOPK_DIST)
            comm.Send(topk_idx, dest=parent, tag=TAG_TOPK_IDX)
            break                     # ya reenvió su Top-K hacia arriba: termina
        elif rank % block == 0:
            child = rank + half
            if child < size:          # guarda para p no potencia de 2
                recv_dist = np.empty((n_test, k), dtype=np.float64)
                recv_idx = np.empty((n_test, k), dtype=np.int64)
                comm.Recv(recv_dist, source=child, tag=TAG_TOPK_DIST)
                comm.Recv(recv_idx, source=child, tag=TAG_TOPK_IDX)
                topk_dist, topk_idx = merge_topk(topk_dist, topk_idx,
                                                 recv_dist, recv_idx, k)

    # ─── Predicción final en la raíz ────────────────────────────────────────
    # El rank 0 conserva y_train_full completo (fue la raíz del scatter), así
    # que puede mapear índice global -> etiqueta y votar.
    if rank == 0:
        k_labels = y_train_full[topk_idx]   # (n_test, k)
        return majority_vote(k_labels, topk_dist)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="KNN-MPI: descomposición por entrenamiento + reducción Top-K en árbol")
    add_dataset_args(parser)
    parser.add_argument("--k", type=int, default=3, help="Número de vecinos")
    parser.add_argument("--pred-out", type=str, default=None,
                        help="Ruta .npy para volcar las predicciones (para verificación).")
    args = parser.parse_args()
    dataset_kwargs = dataset_kwargs_from_args(parser, args)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ─── Carga de datos en rank 0 ───────────────────────────────────────────
    if rank == 0:
        X_train, X_test, y_train, y_test = load_scaled_digits(**dataset_kwargs)
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]
        print(f"[rank 0] n_train={n_train} n_test={n_test} d={n_features} "
              f"p={size} k={args.k}")
        if args.k > n_train:
            raise SystemExit(f"k={args.k} > n_train={n_train}: imposible")
    else:
        X_train = y_train = X_test = y_test = None
        n_train = n_features = n_test = None

    # Metadata escalar (3 enteros): broadcast trivial, NO es el test-set.
    # El test-set y la reducción de Top-K sí usan los árboles explícitos.
    n_train = comm.bcast(n_train, root=0)
    n_features = comm.bcast(n_features, root=0)
    n_test = comm.bcast(n_test, root=0)

    y_pred = knn_predict_mpi(comm, X_train, y_train, X_test,
                             n_train, n_features, n_test, args.k)

    if rank == 0:
        accuracy = float(np.mean(y_pred == y_test))
        print(f"[rank 0] accuracy={accuracy:.4f}")
        if args.pred_out:
            from pathlib import Path
            pred_path = Path(args.pred_out)
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_path, y_pred)
            print(f"  → predicciones escritas en {args.pred_out}")


if __name__ == "__main__":
    main()
