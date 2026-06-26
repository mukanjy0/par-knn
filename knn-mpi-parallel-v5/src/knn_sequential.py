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
    parser.add_argument("--n", type=int, default=None, help="Tamaño total del dataset")
    parser.add_argument("--k", type=int, default=3, help="Número de vecinos")
    parser.add_argument("--pred-out", type=str, default=None,
                        help="Ruta .npy para volcar las predicciones (para verificación).")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_scaled_digits(args.n)
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
