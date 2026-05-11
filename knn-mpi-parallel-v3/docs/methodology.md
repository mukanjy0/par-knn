# Methodology

## Problem

The project implements K-Nearest Neighbors (KNN) classification for the `load_digits` dataset from scikit-learn. Each image is represented by a vector of `d = 64` features. For each test sample, KNN computes distances to all training samples, chooses the `k` closest labels, and predicts the majority class.

## Sequential baseline

The baseline is `src/knn_sequential.py`. It uses the scalar Euclidean distance formula:

```text
sqrt(sum_j((x_test[j] - x_train[j])^2))
```

Timing starts after data loading and train/test splitting. The reported computation time is the prediction loop over the test set. Communication time is zero for the sequential baseline.

## Parallel MPI version

The final MPI implementation is `src/knn_mpi_v3.py`. Rank 0 loads and splits the data. The training set is replicated on all ranks, while the test set is partitioned across ranks. Each rank predicts labels for its local test chunk. Rank 0 gathers predictions and computes accuracy.

The required MPI communication operations are present:

- `comm.bcast`: broadcasts metadata such as dimensions.
- `comm.Bcast`: broadcasts the training arrays.
- `comm.Scatterv`: scatters test chunks and labels.
- `comm.Gatherv`: gathers local predictions.
- `comm.gather`: gathers local computation times for statistics.

## Timing

Each MPI repetition records:

- total time for the distributed prediction pipeline,
- broadcast time,
- scatter time,
- gather time,
- communication time as `broadcast + scatter + gather`,
- local computation time per rank summarized as max/mean/min/std,
- accuracy.

The maximum rank computation time is used as the parallel computation time because total progress is limited by the slowest rank. Repeated runs should be used for serious measurements; the scripts default to at least 3 repetitions for benchmark runs.

## FLOP counting

FLOPs are counted for the parallelizable KNN distance region, not for data loading or plotting. The convention is:

- one subtraction per dimension,
- one multiplication/square per dimension,
- one accumulation per dimension,
- one final operation per train-test pair to represent the distance finalization convention.

Thus each train-test pair costs `3d + 1` FLOPs, and total FLOPs are:

```text
F(n_train, n_test, d) = (3d + 1) * n_train * n_test
```

The implementation may compare squared distances and avoid `sqrt`, because the ordering is unchanged. The report should keep the FLOP convention explicit and use it consistently for all configurations.

## Data-size scaling

The original digits dataset has 1797 samples. For `n <= 1797`, `data_loader.py` takes a deterministic subsample. For `n > 1797`, it tiles the dataset and adds small deterministic Gaussian noise to replicated feature vectors. This is meant for scalability experiments, not for claiming a new real-world dataset.

## Raw data and reproducibility

Raw measurements are written to CSV with timestamp, hostname, git commit when available, command, `n`, `p`, `k`, train/test sizes, timing fields, accuracy, FLOPs, and FLOPs/s. Result directories are timestamped to avoid overwriting previous runs.
