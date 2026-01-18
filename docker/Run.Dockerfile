# syntax=docker/dockerfile:1
# Dockerfile for running tg-signer directly from source code (no PyInstaller)

ARG PY_VERSION=3.11
ARG PIP_INDEX_URL=https://pypi.org/simple

FROM python:${PY_VERSION}-slim-bookworm

# Environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    libssl3 \
    libffi8 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime \
    && echo "${TZ}" > /etc/timezone

WORKDIR /app

# Copy source code
COPY pyproject.toml ./
COPY tg_signer ./tg_signer
COPY assets ./assets

# Install dependencies
ARG PIP_INDEX_URL
RUN python -m pip install --upgrade pip setuptools wheel \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple && \
    python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    -e ".[speedup,gui]"

ENTRYPOINT ["python", "-m", "tg_signer"]
CMD ["--help"]
