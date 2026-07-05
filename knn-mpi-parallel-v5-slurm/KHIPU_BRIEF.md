# Khipu Brief: KNN-MPI v5 Slurm

This is the short flow. Use `README_KHIPU.md` for troubleshooting details.

## 1. Transfer

```bash
rsync -av --exclude ".git" --exclude ".venv" --exclude "results" \
  knn-mpi-parallel-v5-slurm/ USER@KHIPU_HOST:~/knn-mpi-parallel-v5-slurm/
```

## 2. Prepare Environment On Khipu

```bash
ssh USER@KHIPU_HOST
cd ~/knn-mpi-parallel-v5-slurm
module load python3/3.10.2
RECREATE_VENV=1 bash scripts/khipu_setup_env.sh
.venv/bin/python -c "import numpy, sklearn, mpi4py; print('ok')"
mkdir -p logs
```

## 3. Debug Jobs

```bash
sbatch --partition=debug --time=00:05:00 \
  --export=ALL,BENCHMARK_MODE=strong,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",N_TRAINS="25000",N_TEST=500,K=3,REPS=1,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch

sbatch --partition=debug --time=00:05:00 \
  --export=ALL,BENCHMARK_MODE=weak,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",TRAIN_PER_PROCESS=250,N_TEST=50,K=3,REPS=1,TEST_BATCH_SIZE=25 \
  slurm/knn_benchmark.sbatch

sbatch --partition=debug --time=00:05:00 \
  --export=ALL,BENCHMARK_MODE=accuracy,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",N_TOTALS="300",TEST_SIZE=0.2,K=3,REPS=1,TEST_BATCH_SIZE=25,SPLIT_SEED=42,AUGMENT_SEED=123 \
  slurm/knn_benchmark.sbatch
```

## 4. Actual Jobs

Strong scaling:

```bash
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,BENCHMARK_MODE=strong,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",N_TRAINS="10000 25000 50000 100000",N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

Weak scaling:

```bash
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,BENCHMARK_MODE=weak,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",TRAIN_PER_PROCESS=3125,N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

Accuracy/correctness:

```bash
sbatch --partition=standard --time=02:00:00 \
  --export=ALL,BENCHMARK_MODE=accuracy,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 8 32",N_TOTALS="10000 50000 100000",TEST_SIZE=0.2,K=3,REPS=1,TEST_BATCH_SIZE=250,SPLIT_SEED=42,AUGMENT_SEED=123 \
  slurm/knn_benchmark.sbatch
```

## 5. Results

```bash
tail -f logs/knn_mpi_v5_bench_JOBID.out
rsync -av USER@KHIPU_HOST:~/knn-mpi-parallel-v5-slurm/results/ ./khipu_v5_results/
```
