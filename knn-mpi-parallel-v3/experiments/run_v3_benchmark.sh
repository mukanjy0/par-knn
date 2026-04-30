#!/bin/bash
# Benchmark agregado de v3 para los experimentos del informe.
#
# Recorre el producto cartesiano (n, p) con repeticiones internas y
# acumula todo en un único CSV listo para los notebooks de análisis.
#
# Uso básico:
#   bash experiments/run_v3_benchmark.sh
#
# Personalización (variables de entorno):
#   PROCS="1 2 4 8 12" SAMPLES="5000 20000" REPS=10 \
#       bash experiments/run_v3_benchmark.sh
#
# Salida:
#   experiments/results/v3_results.csv

set -e
cd "$(dirname "$0")/.."

CSV="${CSV:-experiments/results/v3_results.csv}"
PROCS="${PROCS:-1 2 4 8}"
SAMPLES="${SAMPLES:-1797 5000 10000 40000}"
REPS="${REPS:-}"

mkdir -p "$(dirname "$CSV")"

# Empezamos limpio salvo que se pida lo contrario
if [ "${APPEND:-0}" != "1" ]; then
  rm -f "$CSV"
fi

echo "============================================================"
echo "KNN-MPI v3 benchmark — $(date)"
echo "Hardware: $(uname -srm)"
if command -v sysctl >/dev/null 2>&1; then
  echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
elif [ -r /proc/cpuinfo ]; then
  echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
fi
echo "Procesos:    $PROCS"
echo "Tamaños:     $SAMPLES"
echo "Reps medidas: $REPS  (+ 1 warmup descartado)"
echo "Salida:      $CSV"
echo "============================================================"

total_combos=$(( $(echo $PROCS | wc -w) * $(echo $SAMPLES | wc -w) ))
combo=0
start_time=$SECONDS

for n in $SAMPLES; do
  for p in $PROCS; do
    combo=$((combo + 1))
    echo ""
    echo "[$combo/$total_combos] n=$n  p=$p"
    mpirun --oversubscribe -n $p python src/knn_mpi_v3.py \
        --n $n --reps $REPS --csv "$CSV" 2>&1 \
      | grep -vE "WARNING|vader|Local host|help message|orte_base|---"
  done
done

elapsed=$((SECONDS - start_time))
echo ""
echo "============================================================"
echo "Completado en ${elapsed}s. CSV: $CSV"
echo "Filas escritas: $(($(wc -l < "$CSV") - 1))"
echo "============================================================"
