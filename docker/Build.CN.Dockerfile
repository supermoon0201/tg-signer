# syntax=docker/dockerfile:1
# Multi-stage Dockerfile for building tg-signer as a standalone executable
# This version uses Chinese mirrors for faster builds in China

# Build arguments
ARG PY_VERSION=3.11
ARG PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ARG INSTALL_GUI=false
ARG INSTALL_SPEEDUP=true

# ============================================================================
# Builder Stage: Compile Python code to executable using PyInstaller
# ============================================================================
FROM python:${PY_VERSION}-slim-bookworm AS builder

# Configure Debian mirrors (Tsinghua University)
RUN echo "Types: deb\n\
    URIs: https://mirrors.tuna.tsinghua.edu.cn/debian\n\
    Suites: bookworm bookworm-updates bookworm-backports\n\
    Components: main contrib non-free non-free-firmware\n\
    Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\n\
    Types: deb\n\
    URIs: https://security.debian.org/debian-security\n\
    Suites: bookworm-security\n\
    Components: main contrib non-free non-free-firmware\n\
    Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg" \
    > /etc/apt/sources.list.d/debian.sources

# Environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK=on \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source code
COPY . .

# Install Python dependencies and PyInstaller
ARG PIP_INDEX_URL
ARG INSTALL_GUI
ARG INSTALL_SPEEDUP

# Configure pip to use Chinese mirror with official PyPI as fallback
RUN python -m pip install --upgrade pip setuptools wheel \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple && \
    python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    "kurigram<=2.2.7" click pydantic openai croniter json_repair typing-extensions httpx Pillow && \
    if [ "${INSTALL_SPEEDUP}" = "true" ]; then \
    python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    tgcrypto; \
    fi && \
    if [ "${INSTALL_GUI}" = "true" ]; then \
    python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    nicegui; \
    fi && \
    python -m pip install --no-cache-dir \
    --index-url "${PIP_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    pyinstaller && \
    python -m pip install --no-deps --no-cache-dir .

# Build executable with PyInstaller
# Note: --add-data includes assets directory in the executable
# Format: --add-data "source:destination"
ARG INSTALL_GUI
RUN set -ex && \
    PYINSTALLER_ARGS="--onefile \
    --name tg-signer \
    --add-data assets:assets \
    --hidden-import=tg_signer.cli \
    --hidden-import=tg_signer.notification \
    --collect-all kurigram \
    --copy-metadata openai \
    --copy-metadata httpx \
    --copy-metadata click \
    --strip \
    --noupx \
    --clean \
    --workpath /tmp/pyi-build \
    --distpath /build/dist" && \
    if [ "${INSTALL_GUI}" = "true" ]; then \
    PYINSTALLER_ARGS="${PYINSTALLER_ARGS} --hidden-import=tg_signer.webui --collect-all nicegui"; \
    fi && \
    python -m PyInstaller ${PYINSTALLER_ARGS} tg_signer/__main__.py

# ============================================================================
# Runtime Stage: Minimal image with only the executable
# ============================================================================
FROM debian:bookworm-slim AS runtime

# Configure Debian mirrors (Tsinghua University)
RUN echo "Types: deb\n\
    URIs: https://mirrors.tuna.tsinghua.edu.cn/debian\n\
    Suites: bookworm bookworm-updates bookworm-backports\n\
    Components: main contrib non-free non-free-firmware\n\
    Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\n\
    Types: deb\n\
    URIs: https://security.debian.org/debian-security\n\
    Suites: bookworm-security\n\
    Components: main contrib non-free non-free-firmware\n\
    Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg" \
    > /etc/apt/sources.list.d/debian.sources

# Environment variables
ENV TZ=Asia/Shanghai \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    libssl3 \
    libffi8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime \
    && echo "${TZ}" > /etc/timezone

# Create working directory
WORKDIR /app

# Copy executable from builder stage
COPY --from=builder /build/dist/tg-signer /usr/local/bin/tg-signer

# Set executable permissions
RUN chmod +x /usr/local/bin/tg-signer

# Default entrypoint
ENTRYPOINT ["/usr/local/bin/tg-signer"]

# Default command (can be overridden)
CMD ["--help"]
