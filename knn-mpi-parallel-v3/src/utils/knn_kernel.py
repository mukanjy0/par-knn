"""
Kernel KNN vectorizado — extraído de v2 para reuso en v3.

La implementación es idéntica a la documentada en knn_mpi_v2.py, sección
'knn_predict_batch'. Se aísla aquí para que v3 (instrumentado) no duplique
código y para facilitar pruebas unitarias.

NOTA IMPORTANTE: los env vars de threading BLAS deben fijarse ANTES de
importar este módulo (porque importa numpy). Ver el header de knn_mpi_v3.py.
"""
import numpy as np


def knn_predict_batch(X_test_local, X_train, y_train, k, n_classes=10):
    """
    Predicción KNN vectorizada para un bloque de puntos de testeo.

    Distancias: ||a-b||² = ||a||² + ||b||² - 2·a·b   (BLAS GEMM)
    Top-k:      np.argpartition  (O(n) por fila vs O(n log n) de argsort)
    Voto:       matriz one-hot de votos + argmax  (sin Python loops)
    """
    # Distancias al cuadrado por GEMM
    test_sq = np.sum(X_test_local ** 2, axis=1, keepdims=True)
    train_sq = np.sum(X_train ** 2, axis=1)
    cross = X_test_local @ X_train.T
    dists_sq = test_sq + train_sq - 2.0 * cross

    # Top-k vecinos (sin sqrt: preserva orden)
    k_indices = np.argpartition(dists_sq, k, axis=1)[:, :k]
    k_labels = y_train[k_indices]

    # Voto mayoritario vectorizado
    votes = np.zeros((len(k_labels), n_classes), dtype=np.int32)
    rows = np.arange(len(k_labels))[:, None]
    np.add.at(votes, (rows, k_labels), 1)
    return votes.argmax(axis=1).astype(np.int64)
