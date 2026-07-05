"""
KNN-MPI — Clasificación K-Nearest Neighbors paralela con descomposición por el
conjunto de ENTRENAMIENTO, usando colectivas MPI y una operación de reducción
PERSONALIZADA para combinar listas Top-K.

El algoritmo
------------
El trabajo dominante de KNN es O(n_te · n_tr · d): para cada punto de prueba hay
que medir su distancia a todos los puntos de entrenamiento. Por eso se reparte el
ENTRENAMIENTO entre los p procesos y se replica el conjunto de prueba. El cómputo
se apoya íntegramente en colectivas MPI:

  Fase 1 — Repartir el entrenamiento  (comm.Scatterv).
      El conjunto de entrenamiento se parte en p bloques de ~n_tr/p puntos y se
      reparte: cada proceso recibe su bloque de puntos y sus etiquetas. Scatterv
      (en vez de Scatter) permite bloques de tamaño distinto cuando n_tr no es
      múltiplo de p.

  Fase 2 — Difundir el test  (comm.Bcast).
      Todo el conjunto de prueba se difunde a cada proceso. MPI implementa el
      broadcast con un árbol interno de coste log(p).

  Fase 3 — Top-K parcial local.
      Cada proceso calcula, para cada punto de prueba, sus K vecinos más cercanos
      DENTRO de su bloque local: las K menores distancias y los índices globales
      de esos vecinos.

  Fase 4 — Reducir los Top-K  (comm.Reduce con operación PERSONALIZADA).
      Aquí está el núcleo de esta versión. MPI trae reducciones predefinidas
      (MPI.SUM, MPI.MAX, ...), pero no una para "quedarse con los K mejores". Se
      define entonces una operación propia con MPI.Op.Create cuya regla de
      combinación es MergeTopK(A, B, K) = los K mejores candidatos de A ∪ B por
      orden lexicográfico (distancia, índice global), y se la pasa a comm.Reduce.
      MPI la aplica en su árbol de reducción de coste
      log(p), igual que haría con una suma. La operación es válida como reducción
      porque MergeTopK es asociativa y conmutativa, y porque los K vecinos
      globales más cercanos están necesariamente en la unión de los K mejores
      locales de cada proceso. El resultado (el Top-K global por punto de prueba)
      queda en el proceso raíz, que emite la predicción por voto mayoritario.

Empaquetado para la operación personalizada
--------------------------------------------
Una operación de reducción de MPI opera sobre un único buffer de un tipo de dato.
Como cada candidato es un par (distancia, índice) que debe viajar junto, se
empaquetan ambos en un buffer float64 de forma (n_test, K, 2): el plano [...,0]
guarda las distancias y el plano [...,1] los índices. Los índices se representan
exactamente mientras n_train < 2**53, condición verificada en rank 0. La función
de combinación reinterpreta el buffer, mezcla los 2K candidatos de cada punto de
prueba y deja los K mejores en sitio.

Manejo de casos no divisibles
-----------------------------
  - n_tr no múltiplo de p: Scatterv reparte bloques desiguales con counts/displs
    (los primeros n_tr % p procesos reciben un punto extra).
  - bloque con < K puntos: local_topk_batch rellena con (+inf, -1); el relleno
    nunca sobrevive a la reducción global mientras n_tr >= K.

Uso:
    mpirun -n 4 python src/knn_mpi_v3.py --n 10000 --k 3

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


def block_counts_displs(n, size):
    """
    Reparto balanceado de `n` elementos entre `size` procesos (maneja resto).
    Los primeros (n % size) procesos reciben un elemento extra.

    Returns
    -------
    counts : (size,) int32   nº de elementos por proceso
    displs : (size,) int32   offset global del bloque de cada proceso
    """
    counts = np.array([n // size + (1 if r < n % size else 0)
                       for r in range(size)], dtype=np.int32)
    displs = np.array([int(np.sum(counts[:r])) for r in range(size)], dtype=np.int32)
    return counts, displs


def make_merge_topk_op(k):
    """
    Crea la operación de reducción PERSONALIZADA MergeTopK para comm.Reduce.

    La función de combinación recibe los dos operandos como buffers crudos
    (invec, inoutvec) de doubles. Cada buffer codifica (n_test, K, 2): el plano
    [...,0] son distancias y el plano [...,1] son índices. Combina los 2K
    candidatos de cada punto de prueba y escribe los K mejores EN SITIO sobre
    inoutvec (convención de MPI: inoutvec <- op(invec, inoutvec)).

    `commute=True`: MergeTopK ordena por (distancia, índice global), así que es
    determinista, asociativa y conmutativa; MPI puede ordenar libremente el árbol
    de reducción sin cambiar el Top-K resultante.
    """
    def _merge(invec, inoutvec, datatype):
        # frombuffer reinterpreta la memoria del buffer sin copiar; inoutvec es
        # escribible, así que las asignaciones impactan el buffer real de MPI.
        a = np.frombuffer(invec, dtype=np.float64).reshape(-1, k, 2)
        b = np.frombuffer(inoutvec, dtype=np.float64).reshape(-1, k, 2)
        merged_dist, merged_idx = merge_topk(a[:, :, 0], a[:, :, 1],
                                             b[:, :, 0], b[:, :, 1], k)
        b[:, :, 0] = merged_dist
        b[:, :, 1] = merged_idx

    return MPI.Op.Create(_merge, commute=True)


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

    counts, displs = block_counts_displs(n_train, size)
    local_n = int(counts[rank])
    offset = int(displs[rank])   # índice global del primer punto del bloque

    # ─── FASE 1: repartir el entrenamiento (Scatterv) ───────────────────────
    # Cada proceso recibe su bloque de puntos y etiquetas. Los counts/displs van
    # en unidades de FILAS para y, y de ELEMENTOS (filas·d) para X.
    LocalX = np.empty((local_n, n_features), dtype=np.float64)
    LocalY = np.empty(local_n, dtype=np.int64)
    sendX = np.ascontiguousarray(X_train_full) if rank == 0 else None
    sendY = np.ascontiguousarray(y_train_full, dtype=np.int64) if rank == 0 else None
    comm.Scatterv(
        [sendX, counts * n_features, displs * n_features, MPI.DOUBLE],
        LocalX, root=0
    )
    comm.Scatterv(
        [sendY, counts, displs, MPI.INT64_T],
        LocalY, root=0
    )

    # ─── FASE 2: difundir el test (Bcast) ───────────────────────────────────
    if rank == 0:
        X_test = np.ascontiguousarray(X_test_full)
    else:
        X_test = np.empty((n_test, n_features), dtype=np.float64)
    comm.Bcast(X_test, root=0)

    # ─── FASE 3: Top-K parcial local ────────────────────────────────────────
    # K vecinos más cercanos de cada punto de prueba DENTRO del bloque local.
    # Índices locales -> globales sumando el offset (los rellenos -1 se dejan).
    topk_dist, topk_idx_local = local_topk_batch(X_test, LocalX, k)
    topk_idx = np.where(topk_idx_local >= 0, topk_idx_local + offset, topk_idx_local)

    # ─── FASE 4: reducir los Top-K (Reduce con operación personalizada) ─────
    # Empaquetamos (dist, idx) en un buffer (n_test, K, 2) float64 y reducimos
    # con MergeTopK. El Top-K global queda en recvbuf, solo en el rank 0.
    sendbuf = np.empty((n_test, k, 2), dtype=np.float64)
    sendbuf[:, :, 0] = topk_dist
    sendbuf[:, :, 1] = topk_idx.astype(np.float64)
    recvbuf = np.empty((n_test, k, 2), dtype=np.float64) if rank == 0 else None

    merge_op = make_merge_topk_op(k)
    comm.Reduce(sendbuf, recvbuf, op=merge_op, root=0)
    merge_op.Free()

    # ─── Predicción final en la raíz ────────────────────────────────────────
    # El rank 0 conserva y_train_full completo, así que mapea índice global ->
    # etiqueta y vota (desempatando por el vecino más cercano vía las distancias).
    if rank == 0:
        global_dist = recvbuf[:, :, 0]
        global_idx = recvbuf[:, :, 1].astype(np.int64)
        k_labels = y_train_full[global_idx]   # (n_test, k)
        return majority_vote(k_labels, global_dist)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="KNN-MPI: descomposición por entrenamiento + Reduce con operación Top-K personalizada")
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
        if n_train >= 2**53:
            raise SystemExit(
                f"n_train={n_train} >= 2**53: los índices no caben exactamente "
                "en el empaquetado float64 de la reducción"
            )
    else:
        X_train = y_train = X_test = y_test = None
        n_train = n_features = n_test = None

    # Metadata escalar (3 enteros) para que cada proceso pueda dimensionar sus
    # buffers antes de las colectivas.
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
