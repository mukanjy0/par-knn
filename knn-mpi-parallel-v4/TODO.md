# TODO

El código actual está enfocado **solo en el algoritmo** (correctitud). Toda la
capa de instrumentación y benchmarking se removió deliberadamente para
concentrarnos en la lógica paralela. Pendiente de re-agregar cuando toque medir:

## Instrumentación de tiempos (en `src/knn_mpi_v4.py`)
- [ ] Medir por fase con `MPI.Wtime()` + `comm.Barrier()`:
      `t_scatter` (Fase 1), `t_bcast` (Fase 2), `t_comp` (Fase 3),
      `t_gather`/`t_reduce` (Fase 4) y `t_total`.
- [ ] Estadísticas de cómputo por proceso (`comm.gather` de `t_comp_local`):
      max / mean / min / std para detectar desbalance de carga.
- [ ] Repeticiones medidas (`--reps`) con repetición de **warm-up** descartada
      (mitiga el cold-start de BLAS).

## Salida de resultados
- [ ] CSV append (`--csv`) con una fila por repetición.
- [ ] JSON de la última repetición (`--json`).
- [ ] Metadata: timestamp, hostname, git commit, comando.
- [ ] FLOPs y FLOPs/s según `(3·d + 1)·n_tr·n_te`.
- [ ] Columnas `speedup` / `efficiency` (las calcula el agregador a partir del
      baseline `p=1` o secuencial).

## Scripts de análisis (removidos, re-crear o portar desde otra versión)
- [ ] `aggregate_results.py` — agrupa el CSV crudo y calcula speedup/efficiency.
- [ ] `plot_results.py` — figuras del informe (tiempo/speedup/efficiency/FLOPs vs p).
- [ ] `check_results.py` — validación del CSV.
- [ ] `run_experiment_matrix.sh` — barrido `p × n` a un CSV + agregación.
- [ ] `local_smoke_test.sh` — humo local con salida a CSV.

## Lo que SÍ está listo
- `src/knn_mpi_v4.py` — algoritmo paralelo (4 fases, árboles explícitos).
- `src/knn_sequential.py` — baseline de referencia.
- `src/utils/` — `data_loader.py`, `knn_kernel.py`.
- `scripts/verify_correctness.sh` + `scripts/compare_preds.py` — verificación
  de correctitud (secuencial vs MPI, e independencia respecto a p).
