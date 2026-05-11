#!/usr/bin/env bash
#
# Submit from the project root, after creating logs/:
#   mkdir -p logs
#   sbatch slurm/knn_benchmark.sbatch
#
# Debug example:
#   mkdir -p logs
#   sbatch --partition=debug --time=00:05:00 --export=ALL,MPI_LAUNCHER=mpirun,PROCS="1 2",SAMPLES="120",REPS=1 slurm/knn_benchmark.sbatch
#
# Full benchmark example:
#   mkdir -p logs
#   sbatch --partition=standard --time=01:00:00 --export=ALL,MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",SAMPLES="1797 5000 10000",REPS=3 slurm/knn_benchmark.sbatch
#   # If needed, increase up to the account limit:
#   sbatch --partition=standard --time=08:00:00 --export=ALL,MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",SAMPLES="1797 5000 10000",REPS=3 slurm/knn_benchmark.sbatch

#SBATCH --job-name=knn_mpi_bench        # Short name shown by squeue
#SBATCH --partition=standard            # Khipu CPU partition for general benchmarks
#SBATCH --nodes=1                       # Keep one node unless the course asks multi-node
#SBATCH --ntasks=32                     # Account CPU limit and maximum MPI tasks for this matrix
#SBATCH --cpus-per-task=1               # One CPU core/thread per MPI rank
#SBATCH --time=00:30:00                 # Override with --time, maximum account limit is 08:00:00
#SBATCH --output=logs/%x_%j.out         # stdout log; create logs/ before sbatch
#SBATCH --error=logs/%x_%j.err          # stderr log; create logs/ before sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Khipu's system python can be old and may not have NumPy/scikit-learn.
# Load Python by default, then prefer the project-local venv. If your Khipu
# session needs MPI/compiler modules, pass them with KHIPU_MODULES after
# checking `module avail` or `module spider`.
KHIPU_MODULES="${KHIPU_MODULES:-python3/3.10.2}"
if command -v module >/dev/null 2>&1; then
  for mod in $KHIPU_MODULES; do
    module load "$mod" || {
      echo "Failed to load module '$mod'. Check with: module avail $mod" >&2
      exit 1
    }
  done
fi

if [ -z "${PYTHON:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON="$(pwd)/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    PYTHON="$(command -v python)"
  fi
fi
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
PROCS="${PROCS:-1 2 4 8 16 32}"
SAMPLES="${SAMPLES:-1797 5000 10000}"
K="${K:-3}"
REPS="${REPS:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}}"
OUT_DIR="${OUT_DIR:-results/khipu_${STAMP}}"
CSV="${CSV:-${OUT_DIR}/raw_results.csv}"

mkdir -p "$OUT_DIR" logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node list:   ${SLURM_JOB_NODELIST:-unknown}"
echo "Max tasks:   ${SLURM_NTASKS:-unknown}"
echo "Launcher:    $MPI_LAUNCHER"
echo "Python:      $($PYTHON --version)"
echo "Processes:   $PROCS"
echo "Samples:     $SAMPLES"
echo "Reps:        $REPS"
echo "Output dir:  $OUT_DIR"

echo ""
echo "== Dependency preflight =="
$PYTHON - <<'PY'
import sys
print("executable:", sys.executable)
print("version:", sys.version.replace("\n", " "))
missing = []
for name in ["numpy", "sklearn", "mpi4py"]:
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        missing.append(f"{name} ({exc})")
if missing:
    raise SystemExit("Missing Python dependencies: " + ", ".join(missing))
PY

for n in $SAMPLES; do
  echo ""
  echo "== Sequential n=$n =="
  $PYTHON src/knn_sequential.py --n "$n" --k "$K" --reps "$REPS" --csv "$CSV"

  for p in $PROCS; do
    if [ -n "${SLURM_NTASKS:-}" ] && [ "$p" -gt "$SLURM_NTASKS" ]; then
      echo "Skipping p=$p because allocation has only SLURM_NTASKS=$SLURM_NTASKS"
      continue
    fi

    echo ""
    echo "== MPI n=$n p=$p =="
    if [ "$MPI_LAUNCHER" = "srun" ]; then
      srun --ntasks="$p" --cpus-per-task=1 $PYTHON src/knn_mpi_v3.py \
        --n "$n" --k "$K" --reps "$REPS" --csv "$CSV"
    else
      "$MPI_LAUNCHER" -n "$p" $PYTHON src/knn_mpi_v3.py \
        --n "$n" --k "$K" --reps "$REPS" --csv "$CSV"
    fi
  done
done

$PYTHON scripts/aggregate_results.py "$CSV" --out-dir "$OUT_DIR"
$PYTHON scripts/plot_results.py "${OUT_DIR}/summary.csv" --out-dir "${OUT_DIR}/figures"

echo ""
echo "Finished. Raw results: $CSV"
