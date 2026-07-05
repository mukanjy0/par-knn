# KNN-MPI v2 Slurm

This folder contains the Slurm-ready version of the v2 KNN-MPI implementation.
The original `knn-mpi-parallel-v2` folder is left unchanged as the algorithm
reference.

## What changed

- `src/knn_mpi_v2.py` keeps the v2 training decomposition and explicit Top-K
  tree reduction, with benchmark instrumentation added.
- `src/knn_sequential.py` writes baseline CSV rows compatible with the MPI rows.
- `scripts/` contains correctness, smoke-test, aggregation, validation, plotting,
  and Khipu environment setup helpers.
- `slurm/knn_benchmark.sbatch` runs the benchmark matrix on Khipu/Slurm.

## Local checks

From the repository root:

```bash
cd knn-mpi-parallel-v2-slurm
PYTHON=../.venv/bin/python N=120 PROCS="1 2 4" bash scripts/verify_correctness.sh
PYTHON=../.venv/bin/python N=120 PROCS="1 2" REPS=1 bash scripts/local_smoke_test.sh
```

## Khipu

Follow `README_KHIPU.md` for transfer, environment setup, debug submission, full
benchmark submission, and result retrieval.
