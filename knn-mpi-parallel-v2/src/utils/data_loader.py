"""
Data loader para experimentos KNN-MPI.

La regla importante es evitar fuga entre train/test: primero se divide el
dataset ORIGINAL de sklearn digits y solo despues se aumenta cada split por
separado. Cada muestra aumentada conserva el id del digito original del que
proviene, lo que permite validar que los origenes de train y test son disjuntos.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def _check_positive_int(name, value):
    if value is not None and value <= 0:
        raise ValueError(f"{name} debe ser positivo o None")


def _split_original(test_size, split_seed):
    digits = load_digits()
    x_images = digits.images.astype(np.float64)
    y = digits.target.astype(np.int64)
    origin_ids = np.arange(len(y), dtype=np.int64)
    return train_test_split(
        x_images,
        y,
        origin_ids,
        test_size=test_size,
        stratify=y,
        random_state=split_seed,
    )


def _resize_split(x_images, y, origin_ids, target_size, rng, noise_std):
    if target_size is None:
        return (
            x_images.astype(np.float64, copy=True),
            y.astype(np.int64, copy=True),
            origin_ids.astype(np.int64, copy=True),
        )

    if target_size <= len(y):
        chosen = rng.choice(len(y), size=target_size, replace=False)
        return (
            x_images[chosen].astype(np.float64, copy=True),
            y[chosen].astype(np.int64, copy=True),
            origin_ids[chosen].astype(np.int64, copy=True),
        )

    extra_count = target_size - len(y)
    extra_idx = rng.integers(0, len(y), size=extra_count)
    extra_images = x_images[extra_idx].astype(np.float64, copy=True)
    noise = rng.normal(0.0, noise_std, size=extra_images.shape)
    extra_images += noise

    return (
        np.concatenate([x_images, extra_images], axis=0).astype(np.float64, copy=False),
        np.concatenate([y, y[extra_idx]], axis=0).astype(np.int64, copy=False),
        np.concatenate([origin_ids, origin_ids[extra_idx]], axis=0).astype(np.int64, copy=False),
    )


def _flatten(x_images):
    return np.ascontiguousarray(x_images.reshape(x_images.shape[0], -1), dtype=np.float64)


def _metadata(mode, train_origin_ids, test_origin_ids, **info):
    return {
        "mode": mode,
        **info,
        "train_origin_ids": train_origin_ids,
        "test_origin_ids": test_origin_ids,
    }


def load_scaled_digits(
    n_samples_target=None,
    *,
    n_total=None,
    n_train=None,
    n_test=None,
    test_size=0.2,
    orig_test_size=0.2,
    split_seed=42,
    augment_seed=123,
    random_state=None,
    noise_std=0.5,
    return_metadata=False,
):
    """
    Carga sklearn digits con division original-primero y aumento sin fuga.

    Modos:
    - porcentaje: `n_total` (o el alias historico posicional `n_samples_target`)
      usa `test_size` para definir los tamanos finales.
    - fijo: `n_train` y `n_test` definen tamanos finales exactos; la division
      del dataset original usa `orig_test_size`, no la razon final n_test/total.
    - original: sin tamanos, devuelve el dataset original dividido por
      `test_size`, sin aumento.
    """
    if random_state is not None:
        split_seed = random_state

    if n_samples_target is not None:
        if n_total is not None:
            raise ValueError("Use solo uno de n_samples_target/--n y n_total/--n-total")
        n_total = n_samples_target

    _check_positive_int("n_total", n_total)
    _check_positive_int("n_train", n_train)
    _check_positive_int("n_test", n_test)

    fixed_mode = n_train is not None or n_test is not None
    if fixed_mode and (n_train is None or n_test is None):
        raise ValueError("n_train y n_test deben proporcionarse juntos")
    if fixed_mode and n_total is not None:
        raise ValueError("Use n_total o n_train/n_test, no ambos")

    if fixed_mode:
        split_fraction = orig_test_size
        mode = "fixed"
    else:
        split_fraction = test_size
        mode = "percentage" if n_total is not None else "original"

    if not 0.0 < float(split_fraction) < 1.0:
        raise ValueError("test_size/orig_test_size debe estar entre 0 y 1")

    x_train_orig, x_test_orig, y_train_orig, y_test_orig, train_orig_ids, test_orig_ids = (
        _split_original(split_fraction, split_seed)
    )

    if fixed_mode:
        final_n_train = int(n_train)
        final_n_test = int(n_test)
    elif n_total is not None:
        final_n_train = int(round(n_total * (1.0 - test_size)))
        final_n_test = int(n_total) - final_n_train
        if final_n_train <= 0 or final_n_test <= 0:
            raise ValueError("n_total y test_size deben producir train/test no vacios")
    else:
        final_n_train = final_n_test = None

    rng_train = np.random.default_rng(augment_seed)
    rng_test = np.random.default_rng(augment_seed + 1)
    x_train_img, y_train, train_origin_ids = _resize_split(
        x_train_orig, y_train_orig, train_orig_ids, final_n_train, rng_train, noise_std
    )
    x_test_img, y_test, test_origin_ids = _resize_split(
        x_test_orig, y_test_orig, test_orig_ids, final_n_test, rng_test, noise_std
    )

    assert set(train_origin_ids).isdisjoint(set(test_origin_ids))

    X_train = _flatten(x_train_img)
    X_test = _flatten(x_test_img)
    y_train = np.ascontiguousarray(y_train, dtype=np.int64)
    y_test = np.ascontiguousarray(y_test, dtype=np.int64)

    if return_metadata:
        info = {
            "test_size": test_size,
            "orig_test_size": orig_test_size,
            "split_seed": split_seed,
            "augment_seed": augment_seed,
            "n_total": None if n_total is None else int(n_total),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        }
        return X_train, X_test, y_train, y_test, _metadata(
            mode, train_origin_ids, test_origin_ids, **info
        )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    for kwargs in [{}, {"n_total": 5000}, {"n_train": 2500, "n_test": 500}]:
        X_tr, X_te, y_tr, y_te, meta = load_scaled_digits(return_metadata=True, **kwargs)
        print(
            f"{meta['mode']}: train={X_tr.shape} test={X_te.shape} "
            f"dtype={X_tr.dtype} leakage="
            f"{not set(meta['train_origin_ids']).isdisjoint(set(meta['test_origin_ids']))}"
        )
