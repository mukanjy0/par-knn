#!/usr/bin/env bash
#
# Submit from the project root, after creating logs/:
#   mkdir -p logs
#   sbatch slurm/knn_benchmark.sbatch
#
# Debug example, submitted from the project root after preparing .venv on the access node:
#   mkdir -p logs
#   sbatch --partition=debug --time=00:05:00 --export=ALL,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2",N_TRAINS="250",N_TEST=50,REPS=1 slurm/knn_benchmark.sbatch
#
# Strong-scaling benchmark example:
#   sbatch --partition=standard --time=08:00:00 --export=ALL,BENCHMARK_MODE=strong,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",N_TRAINS="10000 25000 50000 100000",N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 slurm/knn_benchmark.sbatch
#
# Weak-scaling benchmark example:
#   sbatch --partition=standard --time=08:00:00 --export=ALL,BENCHMARK_MODE=weak,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 2 4 8 16 32",TRAIN_PER_PROCESS=3125,N_TEST=5000,K=3,REPS=3,TEST_BATCH_SIZE=250 slurm/knn_benchmark.sbatch
#
# Accuracy/correctness example:
#   sbatch --partition=standard --time=02:00:00 --export=ALL,BENCHMARK_MODE=accuracy,KHIPU_MODULES="python3/3.10.2",MPI_LAUNCHER=mpirun,PROCS="1 8 32",N_TOTALS="10000 50000 100000",TEST_SIZE=0.2,K=3,REPS=1,TEST_BATCH_SIZE=250,SPLIT_SEED=42,AUGMENT_SEED=123 slurm/knn_benchmark.sbatch

#SBATCH --job-name=knn_mpi_v4_bench     # Short name shown by squeue
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
# Load Python by default, then use the project-local venv prepared on the
# access node. Do not install dependencies inside the Slurm job.
if [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
KHIPU_MODULES="${KHIPU_MODULES:-python3/3.10.2}"
if command -v module >/dev/null 2>&1; then
  for mod in $KHIPU_MODULES; do
    module load "$mod" || {
      echo "Failed to load module '$mod'. Check with: module avail $mod" >&2
      exit 1
    }
  done
fi

VENV_DIR="${VENV_DIR:-.venv}"
if [ -x "$VENV_DIR/bin/python" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON="$(pwd)/$VENV_DIR/bin/python"
elif [ -n "${PYTHON:-}" ]; then
  echo "Project venv not found at $VENV_DIR; using exported PYTHON=$PYTHON"
else
  echo "Project venv not found: $(pwd)/$VENV_DIR" >&2
  echo "Prepare it on the access node with: module load python3/3.10.2 && RECREATE_VENV=1 bash scripts/khipu_setup_env.sh" >&2
  exit 1
fi
if [ ! -x "$PYTHON" ]; then
  echo "Python executable not found or not executable: $PYTHON" >&2
  echo "Submit from the project root after creating .venv on the access node, or pass VENV_DIR=/path/to/venv." >&2
  exit 1
fi
BENCHMARK_MODE="${BENCHMARK_MODE:-strong}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
K="${K:-3}"
ORIG_TEST_SIZE="${ORIG_TEST_SIZE:-0.2}"
RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-0}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-250}"
SPLIT_SEED="${SPLIT_SEED:-42}"
AUGMENT_SEED="${AUGMENT_SEED:-123}"
TEST_SIZE="${TEST_SIZE:-0.2}"

case "$BENCHMARK_MODE" in
  strong)
    PROCS="${PROCS:-1 2 4 8 16 32}"
    N_TRAINS="${N_TRAINS:-10000 25000 50000 100000}"
    N_TEST="${N_TEST:-5000}"
    REPS="${REPS:-3}"
    ;;
  weak)
    PROCS="${PROCS:-1 2 4 8 16 32}"
    TRAIN_PER_PROCESS="${TRAIN_PER_PROCESS:-3125}"
    N_TEST="${N_TEST:-5000}"
    REPS="${REPS:-3}"
    ;;
  accuracy)
    PROCS="${PROCS:-1 8 32}"
    N_TOTALS="${N_TOTALS:-10000 50000 100000}"
    REPS="${REPS:-1}"
    ;;
  legacy)
    PROCS="${PROCS:-1 2 4 8 16 32}"
    SAMPLES="${SAMPLES:-1797 5000 10000}"
    REPS="${REPS:-3}"
    ;;
  *)
    echo "Unknown BENCHMARK_MODE='$BENCHMARK_MODE'. Use strong, weak, accuracy, or legacy." >&2
    exit 1
    ;;
esac

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID:-local}}"
OUT_DIR="${OUT_DIR:-results/khipu_v4_${BENCHMARK_MODE}_${STAMP}}"
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
echo "Submit dir:  $(pwd)"
echo "Venv dir:    $VENV_DIR"
echo "Python path: $PYTHON"
echo "Python:      $("$PYTHON" --version)"
echo "Mode:        $BENCHMARK_MODE"
echo "Processes:   $PROCS"
echo "N trains:    ${N_TRAINS:-<not used>}"
echo "N test:      ${N_TEST:-<not used>}"
echo "Train/proc:  ${TRAIN_PER_PROCESS:-<not used>}"
echo "N totals:    ${N_TOTALS:-<not used>}"
echo "Samples:     ${SAMPLES:-<not used>}"
echo "K:           $K"
echo "Reps:        $REPS"
echo "Test batch:  $TEST_BATCH_SIZE"
echo "Test size:   $TEST_SIZE"
echo "Split seed:  $SPLIT_SEED"
echo "Aug seed:    $AUGMENT_SEED"
echo "Metric:      median via scripts/aggregate_results.py"
echo "Output dir:  $OUT_DIR"

echo ""
echo "== Venv diagnostics =="
ls -ld "$VENV_DIR" || true
ls -l "$VENV_DIR/bin/python"* || true
if command -v readlink >/dev/null 2>&1; then
  readlink -f "$VENV_DIR/bin/python" 2>/dev/null || true
  readlink -f "$VENV_DIR/bin/python3" 2>/dev/null || true
fi
cat "$VENV_DIR/pyvenv.cfg" 2>/dev/null || true
echo "PATH: $PATH"

echo ""
echo "== Dependency preflight =="
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is too old for this job. "
        "Recreate the project venv after loading python3/3.10.2: "
        "RECREATE_VENV=1 bash scripts/khipu_setup_env.sh"
    )
PY
"$PYTHON" -m pip --version || true
"$PYTHON" -m pip show numpy scikit-learn mpi4py || true
"$PYTHON" - <<'PY'
import sys
import site
print("executable:", sys.executable)
print("version:", sys.version.replace("\n", " "))
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("site-packages:", site.getsitepackages() if hasattr(site, "getsitepackages") else "unknown")
missing = []
for name in ["numpy", "sklearn", "mpi4py"]:
    try:
        module = __import__(name)
        print(f"{name}: {getattr(module, '__version__', 'ok')}")
    except Exception as exc:
        missing.append(f"{name} ({exc})")
if missing:
    raise SystemExit(
        "Missing Python dependencies: "
        + ", ".join(missing)
        + "\nFix: submit from the project directory that contains .venv, "
          "or pass `--export=ALL,VENV_DIR=/absolute/path/to/venv,...`. "
          "Install with `.venv/bin/python -m pip install -r requirements.txt`."
    )
PY

run_mpi_fixed() {
  local p="$1"
  local n_train="$2"
  local pred_out="${3:-}"
  local cmd=()
  if [ "$MPI_LAUNCHER" = "srun" ]; then
    cmd=(srun --ntasks="$p" --cpus-per-task=1 "$PYTHON" src/knn_mpi_v4.py)
  else
    cmd=("$MPI_LAUNCHER" -n "$p" "$PYTHON" src/knn_mpi_v4.py)
  fi
  cmd+=(
    --n-train "$n_train" --n-test "$N_TEST" --orig-test-size "$ORIG_TEST_SIZE"
    --split-seed "$SPLIT_SEED" --augment-seed "$AUGMENT_SEED"
    --k "$K" --reps "$REPS" --test-batch-size "$TEST_BATCH_SIZE" --csv "$CSV"
  )
  if [ -n "$pred_out" ]; then
    cmd+=(--pred-out "$pred_out")
  fi
  "${cmd[@]}"
}

run_mpi_total() {
  local p="$1"
  local n_total="$2"
  local pred_out="${3:-}"
  local cmd=()
  if [ "$MPI_LAUNCHER" = "srun" ]; then
    cmd=(srun --ntasks="$p" --cpus-per-task=1 "$PYTHON" src/knn_mpi_v4.py)
  else
    cmd=("$MPI_LAUNCHER" -n "$p" "$PYTHON" src/knn_mpi_v4.py)
  fi
  cmd+=(
    --n-total "$n_total" --test-size "$TEST_SIZE"
    --split-seed "$SPLIT_SEED" --augment-seed "$AUGMENT_SEED"
    --k "$K" --reps "$REPS" --test-batch-size "$TEST_BATCH_SIZE" --csv "$CSV"
  )
  if [ -n "$pred_out" ]; then
    cmd+=(--pred-out "$pred_out")
  fi
  "${cmd[@]}"
}

check_p_allocation() {
  local p="$1"
  if [ -n "${SLURM_NTASKS:-}" ] && [ "$p" -gt "$SLURM_NTASKS" ]; then
    echo "Skipping p=$p because allocation has only SLURM_NTASKS=$SLURM_NTASKS"
    return 1
  fi
  return 0
}

if [ "$BENCHMARK_MODE" = "strong" ]; then
  for n_train in $N_TRAINS; do
    n_total=$((n_train + N_TEST))
    if [ "$RUN_SEQUENTIAL" = "1" ]; then
      echo ""
      echo "== Sequential n_train=$n_train n_test=$N_TEST =="
      $PYTHON src/knn_sequential.py \
        --n-train "$n_train" --n-test "$N_TEST" --orig-test-size "$ORIG_TEST_SIZE" \
        --split-seed "$SPLIT_SEED" --augment-seed "$AUGMENT_SEED" \
        --k "$K" --reps "$REPS" --csv "$CSV"
    fi

    for p in $PROCS; do
      check_p_allocation "$p" || continue

      echo ""
      echo "== MPI n_train=$n_train n_test=$N_TEST n=$n_total p=$p =="
      run_mpi_fixed "$p" "$n_train"
    done
  done
elif [ "$BENCHMARK_MODE" = "weak" ]; then
  for p in $PROCS; do
    check_p_allocation "$p" || continue
    n_train=$((TRAIN_PER_PROCESS * p))
    n_total=$((n_train + N_TEST))
    echo ""
    echo "== Weak MPI train_per_process=$TRAIN_PER_PROCESS n_train=$n_train n_test=$N_TEST n=$n_total p=$p =="
    run_mpi_fixed "$p" "$n_train"
  done
elif [ "$BENCHMARK_MODE" = "accuracy" ]; then
  for n_total in $N_TOTALS; do
    pred_paths=()
    for p in $PROCS; do
      check_p_allocation "$p" || continue
      pred_out="${OUT_DIR}/predictions/n${n_total}_p${p}.npy"
      pred_paths+=("$pred_out")
      echo ""
      echo "== Accuracy MPI n_total=$n_total test_size=$TEST_SIZE p=$p =="
      run_mpi_total "$p" "$n_total" "$pred_out"
    done
    if [ "${#pred_paths[@]}" -ge 2 ]; then
      echo ""
      echo "== Prediction comparison n_total=$n_total reference=${pred_paths[0]} =="
      "$PYTHON" scripts/compare_preds.py "${pred_paths[@]}"
    fi
  done
elif [ "$BENCHMARK_MODE" = "legacy" ]; then
  for n in $SAMPLES; do
    echo ""
    echo "== Sequential n=$n =="
    $PYTHON src/knn_sequential.py \
      --n "$n" --test-size "$TEST_SIZE" \
      --split-seed "$SPLIT_SEED" --augment-seed "$AUGMENT_SEED" \
      --k "$K" --reps "$REPS" --csv "$CSV"

    for p in $PROCS; do
      check_p_allocation "$p" || continue

      echo ""
      echo "== MPI n=$n p=$p =="
      run_mpi_total "$p" "$n"
    done
  done
fi

$PYTHON scripts/aggregate_results.py "$CSV" --out-dir "$OUT_DIR"
$PYTHON scripts/plot_results.py "${OUT_DIR}/summary.csv" --out-dir "${OUT_DIR}/figures"

echo ""
echo "Finished. Raw results: $CSV"
