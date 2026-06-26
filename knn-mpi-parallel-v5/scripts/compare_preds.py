#!/usr/bin/env python
"""
Compara vectores de predicciones (.npy) y reporta coincidencia.

Uso:
    python scripts/compare_preds.py REF.npy OTHER1.npy [OTHER2.npy ...]

El primer archivo es la REFERENCIA (típicamente la versión secuencial). Cada
uno de los demás se compara contra ella elemento a elemento. Sale con código 0
si todos coinciden al 100%, y 1 si alguno difiere (imprime cuántos y dónde).

Nota: en KNN puede haber diferencias mínimas y legítimas por empates exactos de
distancia en la frontera del Top-K (el desempate de argpartition/argsort es
arbitrario). El script reporta la tasa de coincidencia para que se pueda juzgar.
"""
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Uso: compare_preds.py REF.npy OTHER1.npy [OTHER2.npy ...]")

    ref_path = Path(sys.argv[1])
    ref = np.load(ref_path)
    print(f"Referencia: {ref_path.name}  ({len(ref)} predicciones)")

    all_match = True
    for other_path in sys.argv[2:]:
        other = np.load(other_path)
        name = Path(other_path).name
        if other.shape != ref.shape:
            print(f"  ✗ {name}: shape {other.shape} != ref {ref.shape}")
            all_match = False
            continue
        n_match = int(np.sum(other == ref))
        n_total = len(ref)
        rate = n_match / n_total
        status = "✓" if n_match == n_total else "✗"
        print(f"  {status} {name}: {n_match}/{n_total} coinciden ({rate*100:.2f}%)")
        if n_match != n_total:
            all_match = False
            mismatch = np.where(other != ref)[0]
            preview = ", ".join(f"i={i}(ref={ref[i]},got={other[i]})" for i in mismatch[:10])
            print(f"      primeras discrepancias: {preview}")

    if all_match:
        print("\nRESULTADO: todas las predicciones coinciden al 100%.")
        sys.exit(0)
    else:
        print("\nRESULTADO: hay discrepancias (ver arriba).")
        sys.exit(1)


if __name__ == "__main__":
    main()
