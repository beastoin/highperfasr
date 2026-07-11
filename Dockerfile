# HighPerfASR — GPU ASR server.
#
# Batch:   docker build --target batch -t highperfasr-batch .
# Stream:  docker build --target stream -t highperfasr-stream .
# Both:    docker compose up -d
#
# Includes NeMo fork patches (github.com/beastoin/NeMo) for thread-safety
# and streaming fixes not yet merged upstream.

# --- Stage: clone NeMo fork patches ---
FROM alpine/git:latest AS nemo-fork
ARG NEMO_FORK_REF=3c736deb8b3b5fec7029e88af9c59e84a48b4294
RUN git clone --filter=blob:none --no-checkout \
    https://github.com/beastoin/NeMo.git /nemo-fork \
    && cd /nemo-fork \
    && test -n "$NEMO_FORK_REF" \
    && git checkout --detach "$NEMO_FORK_REF"

# --- Stage: base image ---
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS base

WORKDIR /app

# Build tools needed for numba JIT compilation at first inference
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install NeMo from fork (includes thread-safety + streaming patches)
COPY --from=nemo-fork /nemo-fork/ /tmp/nemo-fork/
RUN pip install --no-cache-dir \
    "/tmp/nemo-fork[asr]" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "python-multipart>=0.0.9" \
    "pyyaml>=6.0" \
    "numpy>=1.24" \
    && rm -rf /tmp/nemo-fork \
    && pip cache purge

# Install highperfasr
COPY labs/nemo-fastapi/pyproject.toml ./pyproject.toml
COPY labs/nemo-fastapi/src/ ./src/
COPY labs/nemo-fastapi/configs/ ./configs/
COPY LICENSE ./
RUN pip install --no-cache-dir --no-deps -e . \
    && pip cache purge

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/app/.cache/huggingface \
    NUMBA_CACHE_DIR=/tmp/numba_cache

EXPOSE 8000
ENTRYPOINT ["highperfasr"]

# --- Batch target: Parakeet TDT 0.6B ---
FROM base AS batch
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["serve", "--config", "/app/configs/serving-batch.yaml"]

# --- Stream target: Nemotron Streaming 0.6B ---
FROM base AS stream
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["serve", "--config", "/app/configs/serving-stream.yaml"]
