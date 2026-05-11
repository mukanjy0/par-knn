# Khipu and Slurm Workflow

This guide avoids SSH automation. Run these commands manually and do not put passwords in scripts or prompts.

## 1. Transfer the repo to Khipu

From your local machine, adjust the username and path:

```bash
cd /path/to/proyecto-CPD
rsync -av --exclude ".git" --exclude ".venv" --exclude "results" \
  knn-mpi-parallel-v3/ USER@KHIPU_HOST:~/knn-mpi-parallel-v3/
```

## 2. SSH and inspect the environment

```bash
ssh USER@KHIPU_HOST
cd ~/knn-mpi-parallel-v3
pwd
python3 --version
which python3
which srun || true
which mpirun || true
```

If Khipu uses environment modules, inspect available Python/MPI modules:

```bash
module avail 2>&1 | grep -Ei "python|mpi|openmpi|mpich" || true
module list
```

Load the course-recommended Python/MPI modules if your instructor provided them.

## 3. Create a project-local virtual environment

Do not install globally.
Khipu's default system Python may be old, so load a Python module first. The Khipu software list includes `python3/3.10.2`; adjust only if `module avail python3` shows a different recommended version.

```bash
module avail python3
module load python3/3.10.2
bash scripts/khipu_setup_env.sh
source .venv/bin/activate
python -c "import numpy, sklearn, mpi4py; print('venv imports ok')"
```

If `module load python3/3.10.2` is unavailable in your account context, run:

```bash
module spider python3
module avail python
```

Then use the available Python module in both setup and Slurm submissions, for example:

```bash
KHIPU_MODULES="python3/AVAILABLE_VERSION" bash scripts/khipu_setup_env.sh
```

## 4. Very light access-node checks

These checks are intentionally tiny.

```bash
# Optional tiny import/CLI check only. Prefer local smoke tests.
mkdir -p results
source .venv/bin/activate
python -c "import mpi4py, sklearn, numpy; print('imports ok')"
python src/knn_sequential.py --n 120 --k 3 --reps 1 --csv results/access_node_check.csv
python scripts/check_results.py results/access_node_check.csv
```

If MPI launching is allowed only inside Slurm, skip direct `mpirun` on the access node.

## 5. Submit a short debug job

Create the log directory before `sbatch` because Slurm opens log files before the script body runs.
Khipu uses the `debug` partition for short CPU tests.

```bash
mkdir -p logs
sbatch --partition=debug --time=00:05:00 \
  --export=ALL,KHIPU_MODULES="python3/3.10.2",PYTHON="$PWD/.venv/bin/python",MPI_LAUNCHER=mpirun,PROCS="1 2",SAMPLES="120",REPS=1 \
  slurm/knn_benchmark.sbatch
```

This Khipu Open MPI environment should use `mpirun` inside the Slurm allocation. Direct `srun` can fail with a PMI/PMIx support error.

## 6. Submit the main benchmark

Your Khipu account limit is `cpu=32` and `TimeLimit=08:00:00`, so the benchmark matrix uses exactly:

```text
p = 1, 2, 4, 8, 16, 32
```

Start with a one-hour run:

```bash
mkdir -p logs
sbatch --partition=standard --time=01:00:00 \
  --export=ALL,KHIPU_MODULES="python3/3.10.2",PYTHON="$PWD/.venv/bin/python",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",SAMPLES="1797 5000 10000",REPS=3 \
  slurm/knn_benchmark.sbatch
```

If the first run is too short for the selected `n` values, increase the wall time up to your account limit:

```bash
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,KHIPU_MODULES="python3/3.10.2",PYTHON="$PWD/.venv/bin/python",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",SAMPLES="1797 5000 10000",REPS=3 \
  slurm/knn_benchmark.sbatch
```

## 7. Monitor, inspect, and cancel if needed

```bash
squeue -u "$USER"
tail -f logs/knn_mpi_bench_JOBID.out
tail -f logs/knn_mpi_bench_JOBID.err
scancel JOBID
```

Replace `JOBID` with the numeric job id.

## 8. Retrieve results locally

From your local machine:

```bash
rsync -av USER@KHIPU_HOST:~/knn-mpi-parallel-v3/results/ ./khipu_results/
rsync -av USER@KHIPU_HOST:~/knn-mpi-parallel-v3/logs/ ./khipu_logs/
```

## 9. Aggregate and plot locally

If the Slurm job completed, it already produced `summary.csv` and figures. To rerun locally:

```bash
cd knn-mpi-parallel-v3
python scripts/aggregate_results.py path/to/raw_results.csv --out-dir path/to/output_dir
python scripts/plot_results.py path/to/output_dir/summary.csv --out-dir path/to/output_dir/figures
python scripts/check_results.py path/to/raw_results.csv
```

## Notes

- Full benchmarks should run through Slurm, not directly on the access node.
- Use CPU tasks only; this project does not assume or use GPUs.
- Keep raw CSV files. Do not report results that were not produced by actual runs.
