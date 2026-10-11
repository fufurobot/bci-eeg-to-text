#!/usr/bin/env bash
# Provision a GPU-less hosted environment (GitHub Codespaces, Binder).
#
# Deliberately does NOT run `uv sync`: the lockfile resolves torch from the CUDA
# index, which installs 5.7 GB of libraries that cannot be used without a GPU
# and dominate build time. Installing torch from the CPU index instead gives a
# ~1.2 GB environment with the same API.
#
# The `torch` install below uses `--no-config`. This script runs inside the
# repository, and `[tool.uv.sources]` pins torch to the CUDA index for local GPU
# development; that pin overrides `--index-url`, so without `--no-config` the
# "CPU" install silently fetches the CUDA wheels instead -- which is exactly
# what happened the first time this script was run. `--no-sources` does NOT
# suppress the pin; `--no-config` does, because it ignores project config.
#
# The real datasets are not bundled (419 GB across two OpenNeuro corpora), so
# dataset-backed tests skip themselves. Everything else runs.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment"
uv venv --python 3.13

echo "==> Installing PyTorch (CPU build)"
uv pip install --no-config --index-url https://download.pytorch.org/whl/cpu torch

echo "==> Installing project dependencies"
# Editable install without deps, so the CUDA torch is not pulled back in.
uv pip install --no-config \
  mne pyedflib transformers \
  pytest pytest-cov ruff \
  jupyterlab ipykernel
uv pip install --no-config -e . --no-deps

echo "==> Registering the Jupyter kernel"
uv run --no-sync python -m ipykernel install --user --name bci-eeg-to-text \
  --display-name "Python (bci-eeg-to-text)" >/dev/null

echo "==> Smoke test"
uv run --no-sync python - <<'PY'
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

    uv run --no-sync pytest -q -m "not realdata"

The EEG corpora are not included (hundreds of GB). Fetch them with DataLad and
point the tools at them:

    export EEG_TEXT_BCI_DATA_ROOT=/path/to/bci-eeg
    uv run --no-sync bci-eeg-to-text split --dataset chineseeeg --link

EOF
