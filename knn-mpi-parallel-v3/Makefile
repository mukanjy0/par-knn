# Makefile — shortcuts para el proyecto KNN-MPI
#
# Uso típico:
#   make install      # instala dependencias
#   make validate     # corre v1, v2, v3 con dataset chico para verificar
#   make benchmark    # corre el benchmark completo de v3 (varios n × p)
#   make analyze      # abre la notebook con los gráficos del informe
#   make clean        # limpia resultados

PYTHON ?= python
MPIRUN ?= mpirun
PROCS  ?= 4
N      ?= 10000

.PHONY: install validate v1 v2 v3 benchmark benchmark-quick analyze clean help

help:
	@echo "Targets disponibles:"
	@echo "  install         - pip install -r requirements.txt"
	@echo "  validate        - verifica que v1, v2, v3 corren y dan misma accuracy"
	@echo "  v1              - corre v1 con PROCS=$(PROCS)"
	@echo "  v2              - corre v2 con PROCS=$(PROCS) N=$(N)"
	@echo "  v3              - corre v3 con PROCS=$(PROCS) N=$(N), 5 reps"
	@echo "  benchmark       - benchmark completo (PROCS, SAMPLES configurables)"
	@echo "  benchmark-quick - barrido reducido para pruebas (~1 min)"
	@echo "  analyze         - abre Jupyter con el notebook de análisis"
	@echo "  clean           - borra resultados y figuras"

install:
	pip install -r requirements.txt

validate:
	@echo "=== Secuencial ==="
	$(PYTHON) src/knn_sequential.py | grep -E "Accuracy|Execution"
	@echo ""
	@echo "=== v1 (p=$(PROCS)) ==="
	$(MPIRUN) -n $(PROCS) $(PYTHON) src/knn_mpi_v1.py | grep -E "Accuracy|Tiempo total"
	@echo ""
	@echo "=== v2 (p=$(PROCS)) ==="
	$(MPIRUN) -n $(PROCS) $(PYTHON) src/knn_mpi_v2.py | grep -E "Accuracy|Tiempo total"
	@echo ""
	@echo "=== v3 (p=$(PROCS)) ==="
	$(MPIRUN) -n $(PROCS) $(PYTHON) src/knn_mpi_v3.py --reps 3 | tail -5

v1:
	$(MPIRUN) -n $(PROCS) $(PYTHON) src/knn_mpi_v1.py

v2:
	$(MPIRUN) -n $(PROCS) $(PYTHON) src/knn_mpi_v2.py --n $(N)

v3:
	$(MPIRUN) -n $(PROCS) $(PYTHON) src/knn_mpi_v3.py --n $(N) --reps 5

benchmark:
	bash experiments/run_v3_benchmark.sh

benchmark-quick:
	PROCS="1 2 4" SAMPLES="1797 5000" REPS=3 bash experiments/run_v3_benchmark.sh

analyze:
	cd analysis && jupyter notebook 01_results_analysis.ipynb

clean:
	rm -f experiments/results/*.csv
	rm -f docs/figures/*.png
	rm -f docs/summary_table.csv
	@echo "Resultados y figuras borrados."
