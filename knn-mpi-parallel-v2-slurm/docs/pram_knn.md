# PRAM and DAG Model for MPI KNN

## Decomposition

The KNN work is decomposed over test samples. Every rank receives the full training set and an exclusive chunk of the test set. This makes the prediction tasks independent after the initial communication.

```text
rank 0 loads data
  -> broadcast metadata and training set
  -> scatter test chunks
  -> each rank computes local KNN predictions
  -> gather predictions
  -> rank 0 computes accuracy
```

## PRAM interpretation

The model is CREW:

- Concurrent Read: all ranks read the replicated training data.
- Exclusive Write: each rank writes predictions for a disjoint test subset.

There are no write conflicts in the KNN computation because a test sample is assigned to exactly one rank.

## Sequential complexity

For `n_train` training samples, `n_test` test samples, dimension `d`, and small fixed `k`, the dominant sequential cost is distance computation:

```text
T_seq = Theta(n_test * n_train * d)
```

Sorting or top-k selection adds extra work. The vectorized implementation uses `argpartition`, so the practical top-k step is closer to linear in `n_train` per test row, but distance computation remains the dominant model used for FLOP reporting.

## Parallel complexity

With `p` MPI processes and balanced test chunks:

```text
T_comp(p) = Theta((n_test / p) * n_train * d)
```

Communication consists of:

- metadata broadcast: small,
- training-set broadcast: proportional to `n_train * d`,
- test-set scatter: proportional to `n_test * d`,
- prediction gather: proportional to `n_test`,
- synchronization and MPI overhead.

A simple model is:

```text
T_par(p) = A / p + T_comm(p) + T_overhead(p)
```

where `A` represents the parallelizable distance work. In the plotting script, a normalized empirical theory curve is fitted as:

```text
T_fit(p) = a / p + b
```

The fitted `b` absorbs communication, synchronization, memory effects, and fixed overheads.

## Speedup limit and optimal p

Ideal speedup is `S(p) = p`. Real speedup is limited by:

- fixed training-set broadcast cost,
- scatter/gather overhead,
- load imbalance when `n_test` is not divisible by `p`,
- memory bandwidth contention,
- process startup and synchronization costs,
- insufficient work per rank when `n` is small.

The expected optimal process count is the smallest `p` near the minimum measured total time for a fixed `n`. After that point, added processes no longer reduce total time because communication and overhead dominate the saved computation.
