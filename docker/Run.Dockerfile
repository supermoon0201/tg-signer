# syntax=docker/dockerfile:1.7
# Dockerfile for running tg-signer directly from source code (no PyInstaller)

ARG PY_VERSION=3.11
ARG PIP_INDEX_URL=https://pypi.org/simple

FROM python:${PY_VERSION}-slim-bookworm

# Environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    TZ=Asia/Shanghai

# Install system dependencies for headless OpenCV/video decoding
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    libssl3 \
    libffi8 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime \
    && echo "${TZ}" > /etc/timezone

WORKDIR /src

# Copy source code
COPY pyproject.toml ./
COPY tg_signer ./tg_signer
COPY assets ./assets

# Install dependencies
ARG PIP_INDEX_URL
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip setuptools wheel \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple && \
    python -m pip install \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    ".[speedup]" playwright && \
    python -m playwright install --with-deps chromium

# Data directory for runtime
WORKDIR /app

ENTRYPOINT ["python", "-m", "tg_signer"]
CMD ["--help"]
