# HighPerfASR — Claude Code Instructions

## What This Project Is

HighPerfASR is "vLLM for ASR" — serving optimization for existing open-source NVIDIA ASR models. We do NOT train models. We take published models (parakeet-tdt-0.6b for batch, nemotron-3.5-asr-streaming-0.6b for streaming) and maximize serving performance while preserving their published WER.

Three layers of value:
1. **Config tuning** — optimal server config for max throughput/concurrency
2. **Framework patching** — fix NeMo bugs (e.g., RNNT thread-safety that unlocked 512 streams)
3. **Serving optimization** — FastAPI server wrapping models efficiently

## Repository Layout

```
beastoin/highperfasr (main branch)
├── Dockerfile                          # multi-target: batch + stream
├── compose.yaml                        # docker compose up -d
├── gke-l4.yaml                         # GKE L4 GPU deployment
├── labs/nemo-fastapi/
│   ├── configs/                        # serving-batch.yaml, serving-stream.yaml, cache_aware_rnnt.yaml
│   ├── src/highperfasr/                # server.py, gpu_worker.py, stream_engine.py, batch_engine.py, compat.py, config.py, cli.py
│   └── pyproject.toml
├── benchmarks/
│   ├── scripts/                        # bench_batch.py, bench_stream.py, bench_stream_soak.py, tune_gpu.py, gates.py, wer_utils.py
│   ├── results/                        # published benchmark reports ({gpu}-{mode}-{timestamp}/)
│   ├── config/quality-gates.json       # gate thresholds
│   └── baselines/registry.json         # baseline report paths
├── docs/                               # GitHub Pages site
│   ├── index.html                      # main results page
│   ├── dashboard.html                  # interactive dashboard
│   ├── runs/                           # per-run HTML pages
│   └── scripts/gen-run-pages.py        # page generator
└── spec/                               # protocol.md, openapi.yaml, asyncapi.yaml
```

## Critical Rules

### Infrastructure
- **NGC base:** `nvcr.io/nvidia/nemo:26.02` (PyTorch 2.6 + CUDA 12.8)
- **DO NOT** use PyTorch 2.12 + CUDA 13.x — CachingHostAllocator crash
- **GHCR images:** `ghcr.io/beastoin/highperfasr-{batch,stream}:<semver>` — never `:latest` for benchmarks
- **GKE:** bench pods in `highperfasr-bench` namespace, L4 GPUs

### Benchmarks
- **Always use repo bench scripts** (`bench_batch.py`, `bench_stream.py`) — never compute WER manually
- **Whisper normalization** required for WER comparability
- **Real speech only** for WER measurement (LibriSpeech test-clean) — silence/tone WAVs inflate RNNT throughput
- **`--image-tag <semver>`** required — records container version in reports
- **Results naming:** `{gpu}-{mode}-{timestamp}/` (e.g., `l4-batch-20260725T055248/`)
- **Both-mode** degrades stream WER (~3.21% vs ~1.87% stream-only) due to GPU sharing — always note mode

### Serving
- Load ALL models on the GPU worker thread — cross-thread CUDA tensor ownership causes segfaults
- `gc.disable()` at module level, `gc.collect(0)` per batch on GPU thread
- uvicorn: `ws_ping_interval=None, ws_ping_timeout=None` for sustained WebSocket workloads
- RNNT `num_slots` must match `max_concurrent_streams`

### Workflow
- Never merge PRs — present link and wait for manager approval
- Use REST API for GitHub operations (`GH_TOKEN_CLASSIC` has `public_repo` scope only)
- Source credentials: `source ~/.config/claudecode-telegram/agent.env`
- No real dollar amounts on GitHub — percentages only
- Run tests before any commit: `python3 -m pytest benchmarks/scripts/tests/ -q`
- Hardcoded HTML data (charts, stats) must match source JSON — cross-check before committing

## Current State (v0.3.0)

### Baselines (L4 GPU)
| Mode | WER | Peak Throughput | Max Concurrency |
|------|-----|-----------------|-----------------|
| Batch (dedicated) | 1.57% | 19.5 RPS / 178x RTFx | c=512, 0 failures |
| Batch (v0.3.0, both) | 1.92% | 13.49 RPS / 100x RTFx | c=32, 0 failures |
| Stream (dedicated) | 3.21% | 297 sess/min / 38.69x RTFx | c=512, 0 failures |
| Stream (v0.3.0, both) | 3.21% | 87.4 sess/min / 11.48x RTFx | c=32, 0 failures |

### Quality Gates
- `max_wer_pct`: batch 2.5%, stream 4.0%
- `max_failure_rate`: 0.0
- `wer_delta`: 0.3% (vs reference baseline)
- `min_rtfx`: 1.0

## Running Benchmarks

```bash
# Batch
python3 benchmarks/scripts/bench_batch.py \
  --server http://localhost:8000 \
  --concurrency 1,8,16,32,64 \
  --sustained-rounds 4 \
  --image-tag v0.3.0 \
  --output /tmp/bench_batch.json

# Streaming
python3 benchmarks/scripts/bench_stream.py \
  --server ws://localhost:8000 \
  --endpoint /v1/stream \
  --concurrency 1,16,32 \
  --sustained-rounds 2 \
  --image-tag v0.3.0 \
  --output /tmp/bench_stream.json

# Quick validation (200 samples)
python3 benchmarks/scripts/bench_batch.py --quick ...

# Generate HTML run pages after publishing results
python3 docs/scripts/gen-run-pages.py
```

## GKE Setup

```bash
gcloud container clusters get-credentials dev-omi-gke --region us-central1 --project based-hardware-dev
kubectl get pods -n highperfasr-bench
kubectl port-forward -n highperfasr-bench svc/bench-l4 10320:8000
```

**Warning:** kubectl port-forwards degrade after 50+ min of WebSocket traffic. For long benchmarks, run scripts directly on the pod.
