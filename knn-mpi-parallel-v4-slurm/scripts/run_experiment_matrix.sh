#!/usr/bin/env bash
# Run a reproducible KNN-MPI v4 experiment matrix and keep raw rows in one CSV.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
PROCS="${PROCS:-1 2 4 8 16 32}"
SAMPLES="${SAMPLES:-1797 5000 10000}"
K="${K:-3}"
REPS="${REPS:-3}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/v4_${STAMP}}"
CSV="${CSV:-${OUT_DIR}/raw_results.csv}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "KNN-MPI v4 experiment matrix"
echo "Date:       $(date)"
echo "Host:       $(hostname 2>/dev/null || echo unknown)"
echo "Python:     $($PYTHON --version)"
echo "Launcher:   $MPI_LAUNCHER"
echo "Processes:  $PROCS"
echo "Samples:    $SAMPLES"
echo "k:          $K"
echo "Reps:       $REPS (+ warmup unless --no-warmup is added manually)"
echo "CSV:        $CSV"
echo "============================================================"

for n in $SAMPLES; do
  echo ""
  echo "== Sequential n=$n =="
  $PYTHON src/knn_sequential.py --n "$n" --k "$K" --reps "$REPS" --csv "$CSV"

  for p in $PROCS; do
    echo ""
    echo "== MPI n=$n p=$p =="
    # shellcheck disable=SC2086
    "$MPI_LAUNCHER" $MPI_EXTRA_ARGS -n "$p" $PYTHON src/knn_mpi_v4.py \
      --n "$n" --k "$K" --reps "$REPS" --csv "$CSV"
  done
done

echo ""
echo "Aggregating results..."
$PYTHON scripts/aggregate_results.py "$CSV" --out-dir "$OUT_DIR"
echo "Done. Results directory: $OUT_DIR"
