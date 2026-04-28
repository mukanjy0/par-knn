"""
Data loader para experimentos KNN-MPI.

load_digits es fijo (1797 muestras). Para los experimentos de escalabilidad
necesitamos variar n, así que replicamos con perturbación gaussiana ligera
para que las muestras no sean idénticas (lo cual rompería la diversidad
del KNN sin afectar la complejidad del cómputo, que es lo que queremos medir).
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_scaled_digits(n_samples_target=None, test_size=0.2, random_state=42, noise_std=0.5):
    """
    Carga load_digits y opcionalmente lo replica hasta n_samples_target.

    Parameters
    ----------
    n_samples_target : int or None
        Si None, retorna el dataset original (1797 muestras).
        Si > 1797, replica con ruido gaussiano hasta alcanzar ese tamaño.
    test_size : float
        Proporción para test split.
    random_state : int
        Semilla para reproducibilidad.
    noise_std : float
        Desviación estándar del ruido gaussiano agregado a las réplicas.
        Los pixeles originales están en [0, 16], así que 0.5 es ~3% de ruido.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Arrays float64 (X) e int64 (y) listos para mpi4py.
    """
    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target.astype(np.int64)

    if n_samples_target is not None and n_samples_target > len(X):
        rng = np.random.default_rng(random_state)
        n_replicas = int(np.ceil(n_samples_target / len(X)))

        X_replicated = np.tile(X, (n_replicas, 1))
        y_replicated = np.tile(y, n_replicas)

        # Ruido solo en las réplicas (no en el original) para preservar la señal
        noise = rng.normal(0, noise_std, X_replicated.shape)
        noise[:len(X)] = 0  # original intacto
        X_replicated = X_replicated + noise

        # Recortar al tamaño exacto
        X = X_replicated[:n_samples_target]
        y = y_replicated[:n_samples_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Sanity check
    for n in [None, 5000, 10000]:
        X_tr, X_te, y_tr, y_te = load_scaled_digits(n)
        print(f"n_target={n}: train={X_tr.shape}, test={X_te.shape}, "
              f"dtype={X_tr.dtype}, classes={len(np.unique(y_tr))}")
