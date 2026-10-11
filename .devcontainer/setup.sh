#!/usr/bin/env bash
# Provision a GPU-less hosted environment (GitHub Codespaces, Binder).
#
# Deliberately does NOT run `uv sync`: the lockfile resolves torch from the
# CUDA index, which installs 5.7 GB of libraries that cannot be used without a
# GPU and dominate build time. Installing torch explicitly from the CPU index
# instead gives a ~1.2 GB environment with the same API.
#
# The real datasets are not bundled (they are 419 GB across two OpenNeuro
# corpora), so the dataset-backed tests skip themselves. Everything else runs.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment"
uv venv --python 3.13

echo "==> Installing PyTorch (CPU build)"
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> Installing project dependencies"
# Editable install without deps, so the CUDA torch above is not pulled back in.
uv pip install \
  mne pyedflib transformers \
  pytest pytest-cov ruff \
  jupyterlab ipykernel
uv pip install -e . --no-deps

echo "==> Registering the Jupyter kernel"
uv run python -m ipykernel install --user --name bci-eeg-to-text \
  --display-name "Python (bci-eeg-to-text)" >/dev/null

echo "==> Smoke test"
uv run python - <<'PY'
import platform

import mne
import torch
import transformers

print(f"python       {platform.python_version()}")
print(f"torch        {torch.__version__}")
print(f"cuda         {torch.cuda.is_available()} (expected False here)")
print(f"mne          {mne.__version__}")
print(f"transformers {transformers.__version__}")
PY

cat <<'EOF'

==> Ready.

Run the tests that need no dataset:

    uv run pytest -q -m "not realdata"

The EEG corpora are not included (hundreds of GB). Fetch them with DataLad and
point the tools at them:

    export EEG_TEXT_BCI_DATA_ROOT=/path/to/bci-eeg
    uv run bci-eeg-to-text split --dataset chineseeeg --link

EOF
