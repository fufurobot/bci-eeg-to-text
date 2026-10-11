# Binder / repo2docker configuration.
#
# This is a Dockerfile rather than environment.yml because the dependency that
# matters (PyTorch) must come from a specific index: the lockfile's default is a
# CUDA build, which is several GB of libraries that cannot run on Binder and
# would dominate its build budget. Installing torch from the CPU index instead
# keeps the image around 1.2 GB.
#
# Binder expects the built image to start a Jupyter server; the base image
# already provides that, so this only installs the environment.

FROM python:3.13-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# uv installs far faster than pip, which matters against Binder's build limit.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /home/jovyan/bci-eeg-to-text

# Dependencies first, so editing source does not invalidate the cached layer.
COPY pyproject.toml README.md ./

RUN uv venv --python 3.13 && \
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install mne pyedflib transformers jupyterlab ipykernel

COPY . .

# Editable install without dependencies: resolving them here would replace the
# CPU torch above with the CUDA build the lockfile pins.
RUN uv pip install -e . --no-deps

# Run as an unprivileged user, as Binder expects.
RUN useradd -m -u 1000 jovyan && chown -R jovyan:jovyan /home/jovyan
USER jovyan

ENV PATH="/home/jovyan/bci-eeg-to-text/.venv/bin:${PATH}"

# Binder launches Jupyter against the repository root.
WORKDIR /home/jovyan/bci-eeg-to-text
