# highperfasr

Production ASR serving for NeMo models: low-latency streaming, high-throughput batch transcription, and GPU-safe inference defaults in one FastAPI server.

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v highperfasr-cache:/root/.cache \
  ghcr.io/beastoin/highperfasr:latest
```

## Benchmarked on NVIDIA L4

| Metric | Streaming | Batch |
|--------|-----------|-------|
| Concurrency | 512 simultaneous streams | 64 concurrent requests |
| WER | 3.21% | 1.57% |
| Throughput | 297 sessions/min | 20 RPS (178x real-time) |
| VRAM | 8.7 GB | 4 GB |
| Stability | 512/512 for 10 min, 0 failures | 0 failures sustained |

All benchmarks use LibriSpeech test-clean with Whisper text normalization.

## Install

```bash
pip install highperfasr
```

With NeMo (requires CUDA):

```bash
pip install "highperfasr[nemo]"
```

## Quick Start

**Streaming ASR** (512 concurrent WebSocket streams):

```bash
highperfasr serve --config configs/serving-stream.yaml
```

**Batch ASR** (file upload with dynamic batching):

```bash
highperfasr serve --config configs/serving-batch.yaml
```

**Both modes** on one GPU:

```bash
highperfasr serve --config configs/serving.yaml
```

## API

### REST — Batch Transcription

```bash
curl -F "file=@audio.wav" http://localhost:8000/v1/transcribe
```

Response:

```json
{"text": "the transcribed text", "audio_path": "/tmp/..."}
```

### WebSocket — Streaming

Connect to `ws://localhost:8000/v1/stream`:

1. Server sends `{"stream_id": "...", "status": "opened"}`
2. Client sends raw PCM16 audio chunks (binary frames)
3. Server sends `{"partial_transcript": "...", "final_transcript": "...", "is_final": false}`
4. Client sends `{"action": "close"}`
5. Server sends final transcript and closes

### Health

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

## Configuration

Configs live in `configs/`. Precedence: CLI args > env vars > YAML > defaults.

| Config | Mode | Model | Use case |
|--------|------|-------|----------|
| `serving-stream.yaml` | stream | Nemotron 3.5 Streaming 0.6B | Real-time transcription |
| `serving-batch.yaml` | batch | Parakeet TDT 0.6B v3 | File upload / offline |
| `serving.yaml` | both | Both models | Full deployment |

Key tuning parameters:

- `stream.max_concurrent_streams` — decoder slots (each costs ~3 MB VRAM)
- `stream.max_stream_drain` — chunks per GPU loop (keep at 16, higher causes VRAM explosion)
- `batcher.max_batch_size` — max files per GPU batch
- `batcher.vram_safety_factor` — fraction of VRAM available for batching (0.8 = 80%)
- `batch_model.attention_mode` — `full` (best quality), `local` (long files), `auto` (switches per batch)

## GPU Compatibility

| GPU | Streaming | Batch | Notes |
|-----|-----------|-------|-------|
| L4 24GB | 512 streams | c=64 | Primary test target |
| A10 24GB | 512 streams | c=64 | Production recommended |
| A100 40/80GB | 1024+ streams | c=128+ | Set `max_inflight: 3` |
| T4 16GB | 256 streams | c=32 | Set `cuda_graphs: false` |

## Project Structure

```
highperfasr/
  src/highperfasr/       # Product code
    server.py            # FastAPI app, routes
    gpu_worker.py        # GPU inference thread
    batch_engine.py      # Dynamic batching
    stream_engine.py     # WebSocket streaming
    config.py            # Config loading
    compat.py            # NeMo patches (version-gated, logged)
    cli.py               # CLI entry point
  configs/               # YAML configs
  benchmarks/            # Benchmark scripts and reports
  labs/                  # NeMo patches, experiments, evidence
  deploy/docker/         # Dockerfiles
  tests/                 # Test suite
```

## License

Apache-2.0
