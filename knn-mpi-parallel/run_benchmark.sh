#!/usr/bin/env bash
# run_benchmark.sh
# Corre knn_sequential + knn_mpi_v1 con p=1,2,4,8 y apenda resultados a
# experiments/results/timing_results.txt
#
# Uso:  bash run_benchmark.sh [repeticiones]
# Ej.:  bash run_benchmark.sh 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="$SCRIPT_DIR/experiments/results/timing_results.txt"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
P_VALUES=(1 2 4 8)
REPS="${1:-3}"

# ── Validaciones ─────────────────────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: venv no encontrado en $SCRIPT_DIR/.venv"
  echo "       Ejecuta: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if ! command -v mpirun &>/dev/null; then
  echo "ERROR: mpirun no encontrado. Instala open-mpi: brew install open-mpi"
  exit 1
fi

# ── Cabecera del bloque ───────────────────────────────────────────────────────
{
  echo ""
  echo "============================================================"
  echo "KNN-MPI Benchmark — $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Hardware: $(sysctl -n machdep.cpu.brand_string)"
  echo "P-cores: $(sysctl -n hw.perflevel0.physicalcpu)  E-cores: $(sysctl -n hw.perflevel1.physicalcpu)"
  echo "MPI: $(mpirun --version 2>&1 | head -1)"
  echo "Python: $($PYTHON --version)"
  echo "Repeticiones por configuración: $REPS"
  echo "============================================================"
} | tee -a "$RESULTS_FILE"

# ── Baseline secuencial ───────────────────────────────────────────────────────
{
  echo ""
  echo "--- Sequential baseline ---"
} | tee -a "$RESULTS_FILE"

for i in $(seq 1 "$REPS"); do
  echo "[seq run $i]" | tee -a "$RESULTS_FILE"
  MPLBACKEND=Agg "$PYTHON" "$SCRIPT_DIR/src/knn_sequential.py" 2>&1 \
    | grep -E "Accuracy|Execution" \
    | tee -a "$RESULTS_FILE"
done

# ── Runs MPI ─────────────────────────────────────────────────────────────────
for p in "${P_VALUES[@]}"; do
  {
    echo ""
    echo "--- p=$p ---"
  } | tee -a "$RESULTS_FILE"

  for i in $(seq 1 "$REPS"); do
    echo "[mpi run $i / p=$p]" | tee -a "$RESULTS_FILE"
    mpirun --oversubscribe -n "$p" "$PYTHON" "$SCRIPT_DIR/src/knn_mpi_v1.py" 2>&1 \
      | tee -a "$RESULTS_FILE"
  done
done

# ── Cierre ───────────────────────────────────────────────────────────────────
{
  echo ""
  echo "Benchmark completado: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
} | tee -a "$RESULTS_FILE"

echo ""
echo "Resultados guardados en: $RESULTS_FILE"
