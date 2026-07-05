#!/usr/bin/env bash
# Verifica que la versión MPI produce las MISMAS predicciones que la versión
# secuencial de referencia, y que el resultado es INDEPENDIENTE de p
# (propiedad clave de la reducción de Top-K: el árbol no debe cambiar la salida).
#
# Estrategia:
#   1. Corre la secuencial -> seq.npy  (referencia "verdad")
#   2. Corre la MPI con varios p (1, 2, 4, 8) -> mpi_pP.npy
#   3. Compara todos contra la referencia con compare_preds.py
#
# Uso:   bash scripts/verify_correctness.sh
#        N=10000 K=3 PROCS="1 2 4 8" PYTHON=python bash scripts/verify_correctness.sh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpirun}"
N="${N:-300}"
K="${K:-3}"
PROCS="${PROCS:-1 2 4 8}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/verify_${STAMP}}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "Verificación de correctitud KNN-MPI"
echo "n=$N  k=$K  procesos=$PROCS"
echo "Salida: $OUT_DIR"
echo "============================================================"

echo ""
echo "== Referencia secuencial =="
$PYTHON src/knn_sequential.py --n "$N" --k "$K" \
  --pred-out "$OUT_DIR/seq.npy"

CMP_ARGS=("$OUT_DIR/seq.npy")

echo ""
echo "== MPI para cada p =="
for p in $PROCS; do
  echo "-- p=$p"
  "$MPI_LAUNCHER" -n "$p" $PYTHON src/knn_mpi_v5.py \
    --n "$N" --k "$K" \
    --pred-out "$OUT_DIR/mpi_p${p}.npy"
  CMP_ARGS+=("$OUT_DIR/mpi_p${p}.npy")
done

echo ""
echo "== Comparación de predicciones (referencia = secuencial) =="
$PYTHON scripts/compare_preds.py "${CMP_ARGS[@]}"
