## 1. Transfer the repo to Khipu

From your local machine, adjust the username and path:

```bash
cd /path/to/proyecto-CPD
rsync -av --exclude ".git" --exclude ".venv" --exclude "results" \
  knn-mpi-parallel-v4-slurm/ USER@KHIPU_HOST:~/knn-mpi-parallel-v4-slurm/
```

## 3. Create a project-local virtual environment

Do not install globally.
Khipu's default system Python may be old, so load a Python module first. The Khipu software list includes `python3/3.10.2`; adjust only if `module avail python3` shows a different recommended version.

```bash
module avail python3
module load python3/3.10.2
RECREATE_VENV=1 bash scripts/khipu_setup_env.sh
source .venv/bin/activate
python -c "import numpy, sklearn, mpi4py; print('venv imports ok')"
```

## 5. Submit the strong-scaling benchmark

```bash
mkdir -p logs
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,BENCHMARK_MODE=strong,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",N_TRAINS="10000 25000 50000 100000",N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

## 6. Submit the weak-scaling benchmark

```bash
mkdir -p logs
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,BENCHMARK_MODE=weak,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",TRAIN_PER_PROCESS=3125,N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

## 7. Submit the accuracy/correctness benchmark

```bash
mkdir -p logs
sbatch --partition=standard --time=02:00:00 \
  --export=ALL,BENCHMARK_MODE=accuracy,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 8 32",N_TOTALS="10000 50000 100000",TEST_SIZE=0.2,K=3,REPS=1,TEST_BATCH_SIZE=250,SPLIT_SEED=42,AUGMENT_SEED=123 \
  slurm/knn_benchmark.sbatch
```
