# KNN-MPI: Paralelización de K-Nearest Neighbors con MPI

Proyecto del curso de Programación Paralela. Paralelización del algoritmo KNN
sobre el dataset `load_digits` de scikit-learn usando **mpi4py**, con tres
versiones beta documentadas, benchmarks reproducibles y análisis automático.

## Estrategia de paralelización (DAG)

```
                    ┌──────────────────┐
                    │ Initial parameters│  rank 0
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │ bcast(X_train)              │ scatter(X_test)
              │ O(log p · (α + n_tr·β))     │ O(p · (α + n_te/p · β))
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────┐
         │ rank 0 │    │ rank 1 │... │ rank p │   O(n_tr · n_te / p)
         │ dist+  │    │ dist+  │    │ dist+  │
         │ k-NN   │    │ k-NN   │    │ k-NN   │
         └────┬───┘    └────┬───┘    └────┬───┘
              │             │             │
              └─────────────┼─────────────┘
                            │ gather(y_pred)
                            │ O(p · (α + k·β))
                            ▼
                    ┌──────────────┐
                    │  majority    │  rank 0 ensambla
                    └──────────────┘
```

**Modelo PRAM:** CREW (Concurrent Read, Exclusive Write).

## Versiones beta

| # | Archivo | Cambio principal | Hallazgo |
|---|---------|------------------|----------|
| v1 | `src/knn_mpi_v1.py` | bcast + scatter + gather; cómputo escalar | Correcto, eficiencia 89-97% pero cómputo dominado por loops Python |
| v2 | `src/knn_mpi_v2.py` | Vectorización NumPy (GEMM, argpartition) | 207× speedup vs v1 en cómputo, pero exhibe varianza alta — detectada **oversubscription de threads BLAS** |
| v3 | `src/knn_mpi_v3.py` | BLAS pinning (1 thread/proc) + CSV + warm-up | Mediciones limpias y reproducibles para el informe |

## Instalación

```bash
# Linux
sudo apt install libopenmpi-dev openmpi-bin

# macOS
brew install open-mpi

pip install -r requirements.txt
```

## Uso rápido

```bash
make validate          # corre las 3 versiones, verifica accuracy idéntica
make benchmark-quick   # barrido pequeño (~1 min): 3 procs × 2 sizes
make benchmark         # barrido completo (~5-15 min según hardware)
make analyze           # abre el notebook con los gráficos del informe
```

O sin make:

```bash
mpirun -n 4 python src/knn_mpi_v3.py --n 10000 --reps 5 \
    --csv experiments/results/v3_results.csv

bash experiments/run_v3_benchmark.sh
```

## Estructura

```
.
├── src/
│   ├── knn_sequential.py        # Baseline original (no modificado)
│   ├── knn_mpi_v1.py            # Beta 1
│   ├── knn_mpi_v2.py            # Beta 2
│   ├── knn_mpi_v3.py            # Beta 3 (versión final instrumentada)
│   └── utils/
│       ├── data_loader.py       # Carga y replicación de dataset
│       └── knn_kernel.py        # Predicción vectorizada (compartida v3)
│
├── experiments/
│   ├── run_v2_benchmark.sh      # Benchmark v2 (referencia histórica)
│   ├── run_v3_benchmark.sh      # Benchmark final → CSV
│   └── results/                 # Salidas
│
├── analysis/
│   └── 01_results_analysis.ipynb  # Notebook que produce las figuras
│
├── docs/
│   ├── figures/                 # PNG generados por el notebook
│   └── summary_table.csv        # Tabla resumen para el informe
│
├── Makefile
├── requirements.txt
└── README.md
```

## Configuración del entorno (importante)

Para que las mediciones de v3 sean correctas, el script fija automáticamente
las variables de entorno que limitan BLAS a 1 thread por proceso:

```python
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1   # macOS Accelerate
NUMEXPR_NUM_THREADS=1
```

Esto evita la oversubscription que detectamos en v2 (donde p procesos × 8 threads
de BLAS competían por los mismos 8 cores físicos).

## Métricas reportadas

El CSV de v3 contiene una fila por repetición con: `n, p, k, rep, t_total,
t_bcast, t_scatter, t_gather, t_comm, t_comp_max, t_comp_mean, t_comp_min,
t_comp_std, accuracy, flops, flops_per_sec`.

Los **FLOPs** se calculan según la fórmula del enunciado:
`(3·d + 1) · n_tr · n_te` donde `d` es la dimensión de los vectores
(64 para `load_digits`).
