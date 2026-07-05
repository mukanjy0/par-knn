"""
Kernel KNN vectorizado para la descomposición por entrenamiento.

Como cada proceso solo tiene un BLOQUE del entrenamiento, su Top-K es *parcial*
y debe combinarse con el de los demás procesos. Este módulo provee las piezas
de cómputo local y de combinación:

  - local_topk_batch : Top-K parcial (distancias + índices) sobre un bloque.
  - merge_topk       : combina dos listas Top-K → la mejor Top-K de la unión.
                       Es la operación asociativa que justifica la reducción.
  - majority_vote    : voto mayoritario final sobre los K vecinos globales.

NOTA IMPORTANTE: los env vars de threading BLAS deben fijarse ANTES de importar
este módulo (porque importa numpy). Ver el header de knn_mpi_v4.py.
"""
import numpy as np


# ── Sentinela de relleno (padding) ──────────────────────────────────────────
# Cuando un bloque tiene menos de K puntos, rellenamos los huecos del Top-K
# con distancia +inf e índice -1. Como +inf nunca gana en una comparación de
# mínimos, estos rellenos jamás sobreviven al merge global mientras exista al
# menos K candidatos reales en total (n_tr >= K, siempre cierto aquí).
PAD_DIST = np.inf
PAD_IDX = -1


def _lex_topk(dist, idx, k):
    """
    Devuelve los K mejores candidatos por orden lexicográfico:
    primero menor distancia, y ante empate menor índice global/local.
    """
    order = np.lexsort((idx, dist), axis=1)[:, :k]
    rows = np.arange(dist.shape[0])[:, None]
    return dist[rows, order], idx[rows, order]


def local_topk_batch(X_test, X_train_local, k):
    """
    Top-K PARCIAL de un bloque local de entrenamiento.

    Para cada punto de prueba calcula las K distancias (al cuadrado) más
    pequeñas frente a los puntos del bloque local y devuelve, junto con ellas,
    el índice LOCAL del vecino dentro del bloque (el llamador le suma el offset
    global del bloque para obtener el índice global de entrenamiento).

    Devuelve SIEMPRE exactamente K columnas. Si el bloque tiene menos de K
    puntos, las columnas sobrantes se rellenan con (PAD_DIST, PAD_IDX) para que
    el buffer tenga tamaño fijo (n_test, K) y la reducción en árbol sea uniforme.

    Returns
    -------
    dist : (n_test, K) float64   distancias al cuadrado (sin sqrt: preserva orden)
    idx  : (n_test, K) int64     índices LOCALES (o PAD_IDX en los rellenos)
    """
    n_test = X_test.shape[0]
    n_local = X_train_local.shape[0]

    # Buffers de salida ya rellenos con el sentinela.
    out_dist = np.full((n_test, k), PAD_DIST, dtype=np.float64)
    out_idx = np.full((n_test, k), PAD_IDX, dtype=np.int64)

    # Bloque vacío (puede pasar si p > n_tr): solo relleno.
    if n_local == 0:
        return out_dist, out_idx

    # Distancias al cuadrado por GEMM: (n_test, n_local)
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
    train_sq = np.sum(X_train_local ** 2, axis=1)
    cross = X_test @ X_train_local.T
    dists_sq = test_sq + train_sq - 2.0 * cross

    # Top-kk vecinos del bloque (kk = K salvo que el bloque sea más chico).
    # Se ordena lexicográficamente por (distancia, índice local) para que los
    # empates exactos sean reproducibles e independientes del número de ranks.
    kk = min(k, n_local)
    local_idx = np.broadcast_to(np.arange(n_local, dtype=np.int64), dists_sq.shape)
    top_dist, top_idx = _lex_topk(dists_sq, local_idx, kk)
    out_dist[:, :kk] = top_dist
    out_idx[:, :kk] = top_idx   # índice LOCAL; el offset global lo añade el caller
    return out_dist, out_idx


def merge_topk(dist_a, idx_a, dist_b, idx_b, k):
    """
    MergeTopK(A, B, K): los K mejores candidatos de A ∪ B.

    Esta es la operación de combinación de la reducción en árbol. Es asociativa
    y conmutativa, por eso puede aplicarse en un árbol binario de log(p) pasos:
    concatena los 2K candidatos de ambas listas y se queda con los K de menor
    distancia y, ante empate exacto, menor índice global. Esta regla hace que
    la operación sea determinista, asociativa y conmutativa: el resultado solo
    depende del conjunto de candidatos, no del orden del árbol de reducción.

    Todos los arrays tienen forma (n_test, K). Devuelve (n_test, K).
    """
    dist = np.concatenate([dist_a, dist_b], axis=1)   # (n_test, 2K)
    idx = np.concatenate([idx_a, idx_b], axis=1)      # (n_test, 2K)
    return _lex_topk(dist, idx, k)


def majority_vote(k_labels, k_dist, n_classes=10):
    """
    Voto mayoritario sobre las etiquetas de los K vecinos, desempatando por el
    vecino MÁS CERCANO (convención estándar de KNN).

    Cuando varias clases empatan en número de votos, gana aquella cuyo vecino
    más cercano tiene la menor distancia. Esto reproduce el comportamiento de
    la versión secuencial (Counter.most_common sobre los vecinos ordenados por
    distancia), no el desempate arbitrario por índice de clase de un argmax.

    Parameters
    ----------
    k_labels : (n_test, K) int   etiquetas de los K vecinos
    k_dist   : (n_test, K) float distancias (al cuadrado) de esos K vecinos
    """
    n_test, k = k_labels.shape
    rows = np.arange(n_test)[:, None]

    # Ordenar los K vecinos por distancia ascendente (estable). Así la posición
    # j=0 es el vecino más cercano de cada punto de prueba.
    order = np.argsort(k_dist, axis=1, kind="stable")
    labels_sorted = k_labels[rows, order]            # (n_test, K) por distancia

    # Conteo de votos por clase.
    votes = np.zeros((n_test, n_classes), dtype=np.int64)
    np.add.at(votes, (rows, labels_sorted), 1)

    # Primera posición (en orden de distancia) en que aparece cada clase; las
    # clases ausentes quedan en k. Recorremos de la última a la primera para que
    # la posición más cercana (j menor) sea la que prevalece.
    first_pos = np.full((n_test, n_classes), k, dtype=np.int64)
    flat_rows = rows[:, 0]
    for j in range(k - 1, -1, -1):
        first_pos[flat_rows, labels_sorted[:, j]] = j

    # Puntuación: más votos siempre gana; entre empates, menor first_pos (vecino
    # más cercano) gana. Las clases sin votos se descartan con -infinito.
    score = votes * (k + 1) - first_pos
    score[votes == 0] = np.iinfo(np.int64).min
    return score.argmax(axis=1).astype(np.int64)
