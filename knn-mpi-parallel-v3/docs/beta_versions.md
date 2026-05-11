# Beta Versions

This file records a plausible and honest development progression based on the current repository files. It does not fabricate performance results.

## Beta 1: Sequential baseline and first MPI decomposition

Files:

- `src/knn_sequential.py`
- `src/knn_mpi_v1.py`

Description:

- Started from scalar KNN using Euclidean distance on `load_digits`.
- Added MPI distribution with rank 0 loading data, broadcasting training data, scattering test chunks, local KNN prediction, and gathering predictions.
- Used the required bcast/scatter/gather style communication pattern.

Validation status:

- TODO: include actual accuracy/timing rows from a completed run.

## Beta 2: Vectorized local KNN computation

Files:

- `src/knn_mpi_v2.py`
- `src/utils/data_loader.py`

Description:

- Kept the MPI data distribution from Beta 1.
- Replaced scalar Python distance loops with NumPy vectorized squared-distance computation.
- Used `argpartition` for top-k selection.
- Added optional dataset scaling through deterministic replication for larger `n`.

Validation status:

- TODO: include actual accuracy/timing rows from a completed run.

## Beta 3: Instrumentation and benchmark readiness

Files:

- `src/knn_mpi_v3.py`
- `src/utils/knn_kernel.py`
- `scripts/*.sh`
- `scripts/*.py`
- `slurm/knn_benchmark.sbatch`
- `README_KHIPU.md`
- `docs/*.md`

Description:

- Pinned BLAS/OpenMP thread counts to reduce oversubscription.
- Added repeated measured runs with optional warm-up.
- Recorded total, computation, and communication timing.
- Added FLOP count and FLOPs/s based on the Euclidean-distance region.
- Added CSV/JSON outputs, metadata, aggregation, plotting, and Khipu Slurm workflow.

Validation status:

- Local sequential smoke test can be run without MPI.
- TODO: run MPI smoke test where an MPI launcher is available.
- TODO: run full Slurm benchmarks on Khipu and paste summary tables/figures into the report.
