# HighPerfASR — GPU ASR server.
#
# Batch:   docker build --target batch -t highperfasr-batch .
# Stream:  docker build --target stream -t highperfasr-stream .
# Both:    docker compose up -d

FROM pytorch/pytorch:2.6.0-cuda12.8-cudnn9-runtime AS base

WORKDIR /app

COPY labs/nemo-fastapi/pyproject.toml ./pyproject.toml
COPY labs/nemo-fastapi/src/ ./src/
COPY labs/nemo-fastapi/configs/ ./configs/
COPY LICENSE ./

RUN pip install --no-cache-dir \
    "nemo-toolkit[asr]>=2.5,<2.7" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "python-multipart>=0.0.9" \
    "pyyaml>=6.0" \
    "numpy>=1.24" \
    && pip install --no-cache-dir --no-deps -e . \
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
