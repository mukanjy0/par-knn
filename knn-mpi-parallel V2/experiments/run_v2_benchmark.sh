#!/bin/bash
# Benchmark de v2 — análogo al que se corrió para v1, ahora con dataset escalado.
#
# Uso:
#   bash experiments/run_v2_benchmark.sh > experiments/results/v2_results.txt
#
# Variables ajustables:
#   PROCS           lista de procesos a probar (default: "1 2 4 8")
#   SAMPLES         lista de tamaños de dataset (default: "1797 5000 10000 40000")
#   REPS            repeticiones por configuración (default: 3)

set -e
cd "$(dirname "$0")/.."

PROCS="${PROCS:-1 2 4 8}"
SAMPLES="${SAMPLES:-1797 5000 10000 40000}"
REPS="${REPS:-3}"

echo "============================================================"
echo "KNN-MPI v2 Benchmark — $(date)"
echo "Hardware: $(uname -srm)"
if command -v sysctl >/dev/null; then
  echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
elif [ -r /proc/cpuinfo ]; then
  echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
fi
echo "MPI: $(mpirun --version | head -1)"
echo "Python: $(python --version)"
echo "Procesos:    $PROCS"
echo "Tamaños n:   $SAMPLES"
echo "Repeticiones: $REPS"
echo "============================================================"

# Secuencial baseline (solo con dataset original, para comparar con v1)
echo ""
echo "--- Sequential baseline (n=1797, dataset original) ---"
for r in $(seq 1 $REPS); do
  echo "[seq run $r]"
  python src/knn_sequential.py 2>&1 | grep -E "Accuracy|Execution"
done

# Barrido paralelo
for n in $SAMPLES; do
  for p in $PROCS; do
    echo ""
    echo "--- n=$n, p=$p ---"
    for r in $(seq 1 $REPS); do
      echo "[v2 run $r / n=$n / p=$p]"
      mpirun --oversubscribe -n $p python src/knn_mpi_v2.py --n $n 2>&1 \
        | grep -vE "WARNING|vader|Local host|help message|orte_base|--------" \
        | grep -E "Dataset|Accuracy|Tiempo|Cómputo|Comunicación|Broadcast|Scatter|Gather|FLOPs"
    done
  done
done

echo ""
echo "Benchmark completado: $(date)"
echo "============================================================"
