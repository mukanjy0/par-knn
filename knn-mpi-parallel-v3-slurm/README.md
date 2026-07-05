# KNN-MPI v3 Slurm

This folder contains the Slurm-ready version of the v3 KNN-MPI implementation.
The original `knn-mpi-parallel-v3` folder is left unchanged as the algorithm
reference.

## What changed

- `src/knn_mpi_v3.py` keeps the v3 collective implementation
  (`Scatterv + Bcast + Reduce(custom MergeTopK)`), with benchmark
  instrumentation added.
- `src/knn_sequential.py` writes baseline CSV rows compatible with the MPI rows.
- `scripts/` contains correctness, smoke-test, aggregation, validation, plotting,
  and Khipu environment setup helpers.
- `slurm/knn_benchmark.sbatch` runs the benchmark matrix on Khipu/Slurm.

## Local checks

From the repository root:

```bash
cd knn-mpi-parallel-v3-slurm
PYTHON=../.venv/bin/python N=120 PROCS="1 2 4" bash scripts/verify_correctness.sh
PYTHON=../.venv/bin/python N=120 PROCS="1 2" REPS=1 bash scripts/local_smoke_test.sh
```

## Khipu

Follow `README_KHIPU.md` for transfer, environment setup, debug submission, full
benchmark submission, and result retrieval.
