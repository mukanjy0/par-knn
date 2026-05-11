# AI and External Sources

## AI assistance

Codex was used as a project assistant to inspect the repository, identify reproducibility gaps, add conservative instrumentation, and draft workflow/report documentation. The AI did not generate or claim benchmark results.

Main impacts:

- added CLI and CSV/JSON output to the sequential baseline,
- added metadata fields to MPI result rows,
- added timestamped experiment scripts,
- added Slurm and Khipu workflow documentation,
- added aggregation, plotting, and result-check scripts,
- documented FLOP counting, PRAM/DAG structure, and experiment design.

Validation approach:

- keep raw CSV outputs,
- run local smoke tests,
- run real benchmarks only through Slurm on Khipu,
- compare accuracy across sequential and MPI configurations,
- inspect scripts and logs before using results in the report.

## External sources

The project uses public software/documentation concepts from:

- scikit-learn `load_digits`,
- NumPy vectorized operations,
- mpi4py MPI communication routines,
- Slurm job submission workflow.

When writing the final report, cite the official documentation used by the team and any course material required by the instructor. If additional web sources are consulted later, add them here with a short note about what they influenced.

## Limitations

The generated documentation describes the methodology and workflow. It is not a substitute for measured Khipu results. Any performance claims must come from actual CSV files produced by the scripts.
