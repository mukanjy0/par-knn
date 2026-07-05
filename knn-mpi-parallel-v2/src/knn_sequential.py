"""
KNN secuencial — baseline de referencia para verificar la versión MPI.

Implementación escalar con distancia euclidiana: para cada punto de prueba
calcula la distancia a todos los puntos de entrenamiento, toma los K más
cercanos y predice por voto mayoritario. La opción `--pred-out` vuelca las
predicciones a un .npy para comparar contra la versión MPI.
"""
from collections import Counter
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_scaled_digits


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


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def main():
    parser = argparse.ArgumentParser(description="KNN secuencial (baseline)")
    add_dataset_args(parser)
    parser.add_argument("--k", type=int, default=3, help="Número de vecinos")
    parser.add_argument("--pred-out", type=str, default=None,
                        help="Ruta .npy para volcar las predicciones (para verificación).")
    args = parser.parse_args()
    dataset_kwargs = dataset_kwargs_from_args(parser, args)

    X_train, X_test, y_train, y_test = load_scaled_digits(**dataset_kwargs)
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]
    print(f"[sequential] n_train={n_train} n_test={n_test} d={n_features} k={args.k}")

    y_pred = np.array([knn_predict(x, X_train, y_train, args.k) for x in X_test],
                      dtype=np.int64)
    accuracy = float(np.mean(y_pred == y_test))
    print(f"[sequential] accuracy={accuracy:.4f}")

    if args.pred_out:
        from pathlib import Path
        pred_path = Path(args.pred_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, y_pred)
        print(f"  -> predicciones escritas en {args.pred_out}")


if __name__ == "__main__":
    main()
