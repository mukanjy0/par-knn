# Khipu and Slurm Workflow for KNN-MPI v4

This guide avoids SSH automation. Run these commands manually and do not put passwords in scripts or prompts.

## 1. Transfer the repo to Khipu

From your local machine, adjust the username and path:

```bash
cd /path/to/proyecto-CPD
rsync -av --exclude ".git" --exclude ".venv" --exclude "results" \
  knn-mpi-parallel-v4-slurm/ USER@KHIPU_HOST:~/knn-mpi-parallel-v4-slurm/
```

## 2. SSH and inspect the environment

```bash
ssh USER@KHIPU_HOST
cd ~/knn-mpi-parallel-v4-slurm
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
RECREATE_VENV=1 bash scripts/khipu_setup_env.sh
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

If `pip` fails with a message like `Failed to establish a new connection` or
`Name or service not known`, the Khipu node cannot reach/resolve PyPI. That is
an environment/network issue, not a project failure. Use one of these options.

### Option A: offline wheelhouse

On your local machine with internet access:

```bash
cd knn-mpi-parallel-v4-slurm
python -m pip download -r requirements.txt -d wheelhouse \
  --platform manylinux2014_x86_64 --python-version 310 --implementation cp \
  --abi cp310 --only-binary=:all:
rsync -av wheelhouse/ USER@KHIPU_HOST:~/knn-mpi-parallel-v4-slurm/wheelhouse/
```

Then on Khipu:

```bash
cd ~/knn-mpi-parallel-v4-slurm
module load python3/3.10.2
PIP_OFFLINE=1 WHEELHOUSE=wheelhouse bash scripts/khipu_setup_env.sh
```

If Khipu already provides `mpi4py` as a module, prefer that module and create
the venv with system packages visible:

```bash
VENV_SYSTEM_SITE_PACKAGES=1 PIP_OFFLINE=1 WHEELHOUSE=wheelhouse \
  KHIPU_MODULES="python3/3.10.2 py3-mpi4py/AVAILABLE_VERSION" \
  bash scripts/khipu_setup_env.sh
```

### Option B: Khipu module packages

If Khipu provides all needed Python packages as modules, use the module stack and
skip `pip` entirely:

```bash
module avail 2>&1 | grep -Ei "numpy|scikit|sklearn|mpi4py|pandas|matplotlib"
VENV_SYSTEM_SITE_PACKAGES=1 SKIP_PIP_INSTALL=1 \
  KHIPU_MODULES="python3/3.10.2 py3-mpi4py/AVAILABLE_VERSION ..." \
  bash scripts/khipu_setup_env.sh
```

The final environment check must import `numpy`, `sklearn`, `pandas`,
`matplotlib`, and `mpi4py`.

## 4. Very light access-node checks

These checks are intentionally tiny.

```bash
# Optional tiny import/CLI check only. Prefer local smoke tests.
mkdir -p results
source .venv/bin/activate
which python
python -m pip show numpy scikit-learn mpi4py
python -c "import mpi4py, sklearn, numpy; print('imports ok')"
python src/knn_sequential.py --n 120 --k 3 --reps 1 --csv results/access_node_check.csv
python scripts/check_results.py results/access_node_check.csv
```

Before every `sbatch`, make sure you submit from the same directory where this
check succeeds:

```bash
cd ~/knn-mpi-parallel-v4-slurm
test -x "$PWD/.venv/bin/python"
"$PWD/.venv/bin/python" -c "import numpy, sklearn, mpi4py; print('slurm python ok')"
```

If the Slurm log says the venv Python is `Python 3.6.8` or `No module named pip`,
the project `.venv` was created with Khipu's old system Python. Recreate it after
loading the Python 3.10 module:

```bash
cd ~/knn-mpi-parallel-v4-slurm
module load python3/3.10.2
RECREATE_VENV=1 bash scripts/khipu_setup_env.sh
.venv/bin/python --version
.venv/bin/python -m pip show numpy scikit-learn mpi4py
```

The Slurm script activates `.venv` from `SLURM_SUBMIT_DIR` automatically. Create
and install that venv on the access node before submitting jobs. Do not install
Python packages inside the Slurm job.

If MPI launching is allowed only inside Slurm, skip direct `mpirun` on the access node.

## 5. Submit the strong-scaling benchmark

This benchmark uses fixed-size train/test loading, where the original digits
dataset is split first and augmentation happens separately inside each split.
For strong scaling, the MPI `p=1` run is the baseline. The aggregation script
uses the median of the three runs for `t_total`, speedup, efficiency, and plots.

Your Khipu account limit is `cpu=32` and `TimeLimit=08:00:00`, so the benchmark matrix uses exactly:

```text
k = 3
n_test = 5000
p = 1, 2, 4, 8, 16, 32
n_train = 10000, 25000, 50000, 100000
runs = 3 or 5
metric = median
```

Create the log directory before `sbatch` because Slurm opens log files before
the script body runs. Khipu uses the `debug` partition for short CPU tests.
This Khipu Open MPI environment should use `mpirun` inside the Slurm allocation.
Direct `srun` can fail with a PMI/PMIx support error.

Debug command:

```bash
mkdir -p logs
sbatch --partition=debug --time=00:05:00 \
  --export=ALL,BENCHMARK_MODE=strong,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",N_TRAINS="25000",N_TEST=500,K=3,REPS=1,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

Actual command:

```bash
mkdir -p logs
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,BENCHMARK_MODE=strong,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",N_TRAINS="10000 25000 50000 100000",N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

Use `REPS=5` in the same command if you want a more stable median for the
strong-scaling plots and the wall time still fits the allocation.

`TEST_BATCH_SIZE=250` only batches the local distance computation over test
rows to reduce memory pressure for the largest `p=1` case. It does not change
the train decomposition, broadcast, reduction tree, or reported metric.

## 6. Submit the weak-scaling benchmark

Weak scaling keeps the training work per MPI rank fixed:

```text
k = 3
n_test = 5000
train_per_process = 3125
p = 1, 2, 4, 8, 16, 32
runs = 3
metric = median
```

That means the job runs `n_train = train_per_process * p`, so the training set
grows from `3125` at `p=1` to `100000` at `p=32`.
Use `summary.csv` for the median weak-scaling timings. The existing
`speedup`/`efficiency` columns are strong-scaling metrics based on same-problem
MPI `p=1` baselines, so they are not the main weak-scaling interpretation.

Debug command:

```bash
mkdir -p logs
sbatch --partition=debug --time=00:05:00 \
  --export=ALL,BENCHMARK_MODE=weak,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",TRAIN_PER_PROCESS=250,N_TEST=50,K=3,REPS=1,TEST_BATCH_SIZE=25 \
  slurm/knn_benchmark.sbatch
```

Actual command:

```bash
mkdir -p logs
sbatch --partition=standard --time=08:00:00 \
  --export=ALL,BENCHMARK_MODE=weak,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",TRAIN_PER_PROCESS=3125,N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 \
  slurm/knn_benchmark.sbatch
```

## 7. Submit the accuracy/correctness benchmark

Accuracy/correctness uses percentage mode and fixed seeds so every process
count sees the same generated dataset for each `n_total`.

```text
test_size = 0.2
k = 3
n_total = 10000, 50000, 100000
p = 1, 8, 32
runs = 1
same seed
```

The job writes prediction vectors under the output directory and compares the
outputs for `p=1`, `p=8`, and `p=32` for each `n_total`.

Debug command:

```bash
mkdir -p logs
sbatch --partition=debug --time=00:05:00 \
  --export=ALL,BENCHMARK_MODE=accuracy,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",N_TOTALS="300",TEST_SIZE=0.2,K=3,REPS=1,TEST_BATCH_SIZE=25,SPLIT_SEED=42,AUGMENT_SEED=123 \
  slurm/knn_benchmark.sbatch
```

Actual command:

```bash
mkdir -p logs
sbatch --partition=standard --time=02:00:00 \
  --export=ALL,BENCHMARK_MODE=accuracy,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 8 32",N_TOTALS="10000 50000 100000",TEST_SIZE=0.2,K=3,REPS=1,TEST_BATCH_SIZE=250,SPLIT_SEED=42,AUGMENT_SEED=123 \
  slurm/knn_benchmark.sbatch
```

## 8. Monitor, inspect, and cancel if needed

```bash
squeue -u "$USER"
tail -f logs/knn_mpi_v4_bench_JOBID.out
tail -f logs/knn_mpi_v4_bench_JOBID.err
scancel JOBID
```

Replace `JOBID` with the numeric job id.

## 9. Retrieve results locally

From your local machine:

```bash
rsync -av USER@KHIPU_HOST:~/knn-mpi-parallel-v4-slurm/results/ ./khipu_v4_results/
rsync -av USER@KHIPU_HOST:~/knn-mpi-parallel-v4-slurm/logs/ ./khipu_v4_logs/
```

## 9. Aggregate and plot locally

If the Slurm job completed, it already produced `summary.csv` and figures. To rerun locally:

```bash
cd knn-mpi-parallel-v4-slurm
python scripts/aggregate_results.py path/to/raw_results.csv --out-dir path/to/output_dir
python scripts/plot_results.py path/to/output_dir/summary.csv --out-dir path/to/output_dir/figures
python scripts/check_results.py path/to/raw_results.csv
```

## Notes

- Full benchmarks should run through Slurm, not directly on the access node.
- Use CPU tasks only; this project does not assume or use GPUs.
- Keep raw CSV files. Do not report results that were not produced by actual runs.
