#!/usr/bin/env bash
# Per-project environment setup for Khipu or any Linux MPI host.
# This installs Python packages only inside .venv.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
KHIPU_MODULES="${KHIPU_MODULES:-python3/3.10.2}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
UPGRADE_PIP="${UPGRADE_PIP:-0}"
SKIP_PIP_INSTALL="${SKIP_PIP_INSTALL:-0}"
VENV_SYSTEM_SITE_PACKAGES="${VENV_SYSTEM_SITE_PACKAGES:-0}"
PIP_OFFLINE="${PIP_OFFLINE:-0}"
WHEELHOUSE="${WHEELHOUSE:-wheelhouse}"
RECREATE_VENV="${RECREATE_VENV:-0}"

if [ -r /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

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
venv_args=()
if [ "$RECREATE_VENV" = "1" ]; then
  venv_args+=(--clear)
fi
if [ "$VENV_SYSTEM_SITE_PACKAGES" = "1" ]; then
  venv_args+=(--system-site-packages)
fi
"$PYTHON" -m venv "${venv_args[@]}" "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PIP_DISABLE_PIP_VERSION_CHECK=1

python - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Refusing to install into Python {sys.version.split()[0]}. "
        "Load python3/3.10.2 and recreate the venv with RECREATE_VENV=1."
    )
PY

if [ "$SKIP_PIP_INSTALL" = "1" ]; then
  echo "Skipping pip install because SKIP_PIP_INSTALL=1"
else
  pip_args=(--disable-pip-version-check --no-input)

  if [ "$UPGRADE_PIP" = "1" ]; then
    python -m pip "${pip_args[@]}" install --upgrade pip
  else
    echo "Skipping pip self-upgrade by default. Set UPGRADE_PIP=1 if network access is available."
  fi

  if [ "$PIP_OFFLINE" = "1" ]; then
    if [ ! -d "$WHEELHOUSE" ]; then
      echo "Wheelhouse directory not found: $WHEELHOUSE" >&2
      echo "Create it locally with: python -m pip download -r $REQUIREMENTS -d $WHEELHOUSE" >&2
      exit 1
    fi
    python -m pip "${pip_args[@]}" install --no-index --find-links "$WHEELHOUSE" -r "$REQUIREMENTS"
  else
    python -m pip "${pip_args[@]}" install -r "$REQUIREMENTS" || {
      echo "" >&2
      echo "pip install failed. If Khipu cannot resolve/connect to PyPI, use one of:" >&2
      echo "  1) Offline wheelhouse:" >&2
      echo "     python -m pip download -r $REQUIREMENTS -d $WHEELHOUSE" >&2
      echo "     rsync $WHEELHOUSE/ to Khipu, then run:" >&2
      echo "     PIP_OFFLINE=1 WHEELHOUSE=$WHEELHOUSE bash scripts/khipu_setup_env.sh" >&2
      echo "  2) Module-provided packages, if available:" >&2
      echo "     VENV_SYSTEM_SITE_PACKAGES=1 SKIP_PIP_INSTALL=1 KHIPU_MODULES=\"python3/3.10.2 ...\" bash scripts/khipu_setup_env.sh" >&2
      exit 1
    }
  fi
fi

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
