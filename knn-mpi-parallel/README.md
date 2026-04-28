# KNN-MPI: Paralelización de K-Nearest Neighbors con MPI

Proyecto del curso de Programación Paralela. Paralelización del algoritmo KNN
sobre el dataset `load_digits` de scikit-learn usando **mpi4py**.

## Estrategia de paralelización

Sigue el DAG del enunciado:

```
                    ┌──────────────────┐
                    │ Initial parameters │  (rank 0)
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │  bcast(X_train)             │ scatter(X_test)
              │  O(log(p)·(α + n_tr·β))     │ O(p·(α + n_te/p·β))
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────┐
         │ rank 0 │    │ rank 1 │... │ rank p │
         │ dist+  │    │ dist+  │    │ dist+  │
         │ k-NN   │    │ k-NN   │    │ k-NN   │
         └────┬───┘    └────┬───┘    └────┬───┘
              │             │             │
              └─────────────┼─────────────┘
                            │ gather(y_pred)
                            │ O(p·(α + k·β))
                            ▼
                    ┌──────────────┐
                    │ majority vote │  (rank 0 ensambla)
                    └──────────────┘
```

**Modelo PRAM:** CREW (Concurrent Read, Exclusive Write).

## Instalación

```bash
# Linux
sudo apt install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

# Python deps
pip install -r requirements.txt
```

## Uso

```bash
# Versión secuencial (baseline)
python src/knn_sequential.py

# Versión paralela v1 con 4 procesos
mpirun -n 4 python src/knn_mpi_v1.py
```

## Versiones

| Versión | Descripción | Estado |
|---------|-------------|--------|
| v1 | bcast + scatter + gather básico, distancias escalares | ✅ |
| v2 | Vectorización NumPy de las distancias | 🚧 |
| v3 | Instrumentación completa + benchmarks | 🚧 |

## Estructura del repo

- `src/` — Código fuente (secuencial + 3 versiones paralelas)
- `experiments/` — Scripts de benchmark y resultados crudos
- `analysis/` — Notebooks con análisis y gráficos
- `docs/` — Informe técnico y documentación del modelo PRAM

## Resultados clave

_Por completar tras los experimentos._
