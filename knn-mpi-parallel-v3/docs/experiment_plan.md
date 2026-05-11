# Experiment Plan

## Matrix

Recommended process counts:

```text
p = 1, 2, 4, 8, 16, 32
```

This matrix matches the Khipu account limit of `cpu=32`. Do not use `p = 40`.

Recommended sample sizes:

```text
n = 1797, 5000, 10000
```

The first uses the original digits dataset. Larger sizes use deterministic replication with light noise, so they are appropriate for scalability measurements rather than dataset-quality claims.

Use at least 3 measured repetitions for serious results:

```text
REPS=3
```

The MPI script includes a warm-up repetition by default and excludes it from the CSV.

## Required measurements

Every configuration should record:

- total execution time,
- computation time,
- communication time,
- accuracy,
- `p`,
- `n`,
- `k`,
- train/test sizes,
- feature dimension,
- FLOP count,
- FLOPs/s,
- speedup relative to the `p=1` or sequential baseline,
- efficiency,
- timestamp, hostname, git commit, command.

## Commands

Local smoke test:

```bash
bash scripts/local_smoke_test.sh
```

Khipu debug job:

```bash
mkdir -p logs
sbatch --partition=debug --time=00:05:00 --export=ALL,KHIPU_MODULES="python3/3.10.2",PYTHON="$PWD/.venv/bin/python",MPI_LAUNCHER=mpirun,PROCS="1 2",SAMPLES="120",REPS=1 slurm/knn_benchmark.sbatch
```

Khipu main job:

```bash
mkdir -p logs
sbatch --partition=standard --time=01:00:00 --export=ALL,KHIPU_MODULES="python3/3.10.2",PYTHON="$PWD/.venv/bin/python",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",SAMPLES="1797 5000 10000",REPS=3 slurm/knn_benchmark.sbatch
```

If the selected matrix needs more wall time, use at most the account limit:

```bash
sbatch --partition=standard --time=08:00:00 --export=ALL,KHIPU_MODULES="python3/3.10.2",PYTHON="$PWD/.venv/bin/python",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",SAMPLES="1797 5000 10000",REPS=3 slurm/knn_benchmark.sbatch
```

## Graphs

Produce these figures from `summary.csv`:

- total time vs `p`,
- computation time vs `p`,
- communication time vs `p`,
- speedup vs `p`,
- efficiency vs `p`,
- FLOPs/s vs `p`,
- total time vs `n` for fixed `p`,
- fitted theoretical curve `a/p + b` overlaid against measured time,
- accuracy table.

## Partial report

For the partial report, include:

- explanation of sequential and MPI algorithms,
- PRAM/DAG decomposition,
- proof that required MPI operations are used,
- smoke-test results,
- planned experiment matrix,
- initial timing fields from a small run if available.

## Final report

For the final report, include:

- full raw result CSV reference,
- plots listed above,
- speedup and efficiency analysis,
- FLOPs/s analysis,
- discussion of optimal `p`,
- limitations and reproducibility notes,
- AI/source-use justification.

Do not fabricate Khipu measurements. Mark missing runs as TODO until Slurm jobs actually finish.
