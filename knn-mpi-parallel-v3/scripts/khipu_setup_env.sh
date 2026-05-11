#!/usr/bin/env bash
# Per-project environment setup for Khipu or any Linux MPI host.
# This installs Python packages only inside .venv.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
KHIPU_MODULES="${KHIPU_MODULES:-python3/3.10.2}"

if command -v module >/dev/null 2>&1; then
  for mod in $KHIPU_MODULES; do
    module load "$mod" || {
      echo "Failed to load module '$mod'. Check with: module avail $mod" >&2
      exit 1
    }
  done
fi

echo "Python candidate: $($PYTHON --version)"
echo "Creating/updating virtual environment: $VENV_DIR"
$PYTHON -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo ""
echo "Environment check:"
python --version
python - <<'PY'
import numpy, sklearn, pandas, matplotlib
print("numpy", numpy.__version__)
print("sklearn", sklearn.__version__)
print("pandas", pandas.__version__)
print("matplotlib", matplotlib.__version__)
try:
    import mpi4py
    print("mpi4py", mpi4py.__version__)
except Exception as exc:
    print("mpi4py import failed:", exc)
PY
