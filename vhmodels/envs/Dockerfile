# syntax=docker/dockerfile:1
# Per-model OCI image, or the model-free Ray head image without MODEL_DIR and
# REQUIREMENTS_FILE; build context is the repository root.
FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM docker.io/library/ubuntu:24.04

ARG MODEL_DIR=""
ARG REQUIREMENTS_FILE=""
ARG EXCLUDE_FILE=""
ARG TORCH_BACKEND=""
ARG PYTHON_VERSION
ARG RAY_VERSION

COPY --from=uv /uv /usr/local/bin/uv

# System libraries
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates git build-essential libgl1 libglib2.0-0t64 libgomp1 \
        libstdc++6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python and model dependencies
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_INSTALL_BIN=0 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120 \
    UV_HTTP_RETRIES=10

COPY vhmodels/models/${MODEL_DIR}/requirements*.txt /opt/requirements/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install --managed-python "cpython-${PYTHON_VERSION}-linux-x86_64-gnu" \
    && uv venv --managed-python --python "cpython-${PYTHON_VERSION}-linux-x86_64-gnu" /opt/venv \
    && uv pip install --python /opt/venv/bin/python --compile-bytecode \
        ${TORCH_BACKEND:+--torch-backend "$TORCH_BACKEND"} \
        ${EXCLUDE_FILE:+--excludes "/opt/requirements/$EXCLUDE_FILE"} \
        ${REQUIREMENTS_FILE:+--requirements "/opt/requirements/$REQUIREMENTS_FILE"} \
        "ray[serve]==$RAY_VERSION" python-multipart

# vhmodels source
COPY vhmodels /opt/vhmodels-src/vhmodels

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/vhmodels-src

RUN python -c "import vhmodels${MODEL_DIR:+.models.$MODEL_DIR.model}"

CMD ["python", "-m", "vhmodels.vh_checker.worker", "serve"]
