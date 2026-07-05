# Status

This folder is the Slurm-ready v2 implementation.

Done:
- v2 training-decomposition algorithm with explicit test broadcast and Top-K tree reduction.
- Benchmark CLI: `--reps`, `--no-warmup`, `--csv`, `--json`, `--pred-out`.
- Sequential baseline with the same CSV schema.
- Local smoke/correctness scripts.
- Khipu environment setup helper.
- Slurm batch script for debug and full benchmark runs.
- CSV validation, aggregation, and plotting scripts.

Remaining external validation:
- Run `slurm/knn_benchmark.sbatch` on Khipu with the debug matrix.
- Run the full Khipu matrix once the debug job is clean.
