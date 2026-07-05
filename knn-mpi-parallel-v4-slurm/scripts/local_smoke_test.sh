#!/usr/bin/env bash
# Tiny local validation for v4. This does not install MPI and does not run heavy jobs.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MPI_LAUNCHER="${MPI_LAUNCHER:-}"
N="${N:-120}"
K="${K:-3}"
REPS="${REPS:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
PROCS="${PROCS:-1 2 4}"
OUT_DIR="${OUT_DIR:-results/local_smoke_v4_${STAMP}}"
CSV="${OUT_DIR}/smoke_results.csv"

mkdir -p "$OUT_DIR"

echo "== Local smoke test =="
echo "Python: $($PYTHON --version)"
echo "Processes: $PROCS"
echo "Output: $OUT_DIR"

echo ""
echo "== Sequential baseline =="
$PYTHON src/knn_sequential.py --n "$N" --k "$K" --reps "$REPS" --csv "$CSV" --json "${OUT_DIR}/sequential_last.json"

if [ -z "$MPI_LAUNCHER" ]; then
  if command -v mpirun >/dev/null 2>&1; then
    MPI_LAUNCHER="mpirun"
  elif command -v mpiexec >/dev/null 2>&1; then
    MPI_LAUNCHER="mpiexec"
  fi
fi

echo ""
if [ -n "$MPI_LAUNCHER" ]; then
  echo "== MPI smoke tests with $MPI_LAUNCHER =="
  for p in $PROCS; do
    echo "-- p=$p"
    "$MPI_LAUNCHER" -n "$p" $PYTHON src/knn_mpi_v4.py \
      --n "$N" --k "$K" --reps "$REPS" --no-warmup \
      --csv "$CSV" --json "${OUT_DIR}/mpi_p${p}_last.json"
  done
else
  echo "MPI launcher not found; skipped MPI execution."
  echo "Checking whether mpi4py can at least be imported..."
  $PYTHON - <<'PY'
try:
    import mpi4py
    print("mpi4py import: OK")
except Exception as exc:
    print(f"mpi4py import: FAILED ({exc})")
PY
fi

echo ""
echo "== Result check =="
$PYTHON scripts/check_results.py "$CSV"

echo ""
echo "== Aggregate and plot =="
$PYTHON scripts/aggregate_results.py "$CSV" --out-dir "$OUT_DIR"
$PYTHON scripts/plot_results.py "${OUT_DIR}/summary.csv" --out-dir "${OUT_DIR}/figures"
echo "Smoke CSV: $CSV"
