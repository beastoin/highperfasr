# highperfasr

Serving optimization for existing open-source ASR models.

highperfasr does not train models or change recognition quality. It tunes server configuration and patches framework bottlenecks to maximize throughput and concurrency while preserving the model's published WER.

Run batch and streaming transcription on a single GPU with a simple, framework-agnostic protocol:

- REST endpoint for file transcription
- WebSocket endpoint for real-time PCM16 streams
- Docker Compose for local GPU serving
- GKE L4 manifest for single-GPU deployment
- Published benchmarks with reproducible results

Measured on one GKE L4 GPU with model quality preserved:

| Workload | Result |
|----------|--------|
| Streaming concurrency | 512 WebSocket streams, 0 failures |
| Streaming quality | 3.21% WER on LibriSpeech test-clean |
| Streaming throughput | 297 sessions/min, 38.69x realtime |
| Batch throughput | 178x realtime, about 3 hours of audio per minute |
| Batch quality | 1.57% WER on LibriSpeech test-clean |
| Cost | About $0.70/hr on GKE L4 |

```bash
git clone https://github.com/beastoin/highperfasr
cd highperfasr
docker compose up -d
curl -F "file=@audio.wav" http://localhost:8000/v1/transcriptions
```

## Performance

### Streaming (Nemotron 3.5 ASR 0.6B)

![RTFx Scaling](benchmarks/results/2026-l4-nemo-512-streams/rtfx-scaling.png)

![Throughput Scaling](benchmarks/results/2026-l4-nemo-512-streams/throughput-scaling.png)

![Cost Efficiency](benchmarks/results/2026-l4-nemo-512-streams/cost-efficiency.png)

512 persistent WebSocket streams, 10-minute real-time soak, all 2,620 LibriSpeech
test-clean files. WER 3.21%, 297 sessions/min, 8672 MB VRAM (38%), 0 failures.

Full report: [benchmarks/results/2026-l4-nemo-512-streams/](benchmarks/results/2026-l4-nemo-512-streams/)

### Batch (Parakeet TDT 0.6B)

![Batch RTFx Scaling](benchmarks/results/2026-l4-nemo-batch/rtfx-scaling.png)

![Batch Throughput](benchmarks/results/2026-l4-nemo-batch/throughput-scaling.png)

![Batch Latency](benchmarks/results/2026-l4-nemo-batch/latency-scaling.png)

![Batch Cost Efficiency](benchmarks/results/2026-l4-nemo-batch/cost-efficiency.png)

REST concurrency sweep c=1..512, LibriSpeech test-clean (200 files).
WER 1.57%, peak 19.5 RPS (178x realtime), 0 failures at every level.

Full report: [benchmarks/results/2026-l4-nemo-batch/](benchmarks/results/2026-l4-nemo-batch/)

### Methodology

Quality rubric: real speech corpus, standard WER normalization (Whisper
EnglishTextNormalizer), sustained concurrent load, reproducible artifacts.
Verify: [report schema](benchmarks/report-schema.json),
[streaming result.json](benchmarks/results/2026-l4-nemo-512-streams/result.json),
[batch result.json](benchmarks/results/2026-l4-nemo-batch/result.json).

## Deploy

Prerequisites: Docker, NVIDIA Container Toolkit, and a CUDA-capable GPU. The
first run downloads the ASR models and caches them in Docker volumes.

| Command | What |
|---------|------|
| `docker compose up -d` | Start batch (:8000) + streaming (:8001) |
| `docker compose up -d stream` | Start streaming only |
| `docker compose up -d batch` | Start batch only |
| `make health` | Check server readiness |
| `make smoke` | Run a quick transcription test |
| `docker compose logs -f` | Tail server logs |

### GKE L4

```bash
docker build --target stream -t $REGISTRY/highperfasr-stream:v0.1.0 .
docker push $REGISTRY/highperfasr-stream:v0.1.0
kubectl apply -f gke-l4.yaml
```

## Model Caching

The first start downloads ASR models from HuggingFace (~2.3 GB per model).
Subsequent starts are instant because models are cached in persistent volumes.

The `HF_HOME` environment variable controls the cache directory inside the
container (default: `/app/.cache/huggingface`). Both `compose.yaml` and
`gke-l4.yaml` set this automatically.

### Cache locations

| Setup | Cache Location | Survives Restart |
|-------|---------------|------------------|
| Docker Compose | Named volumes `model-cache-batch`, `model-cache-stream` | Yes (`docker compose down`). Cleared by `docker compose down -v`. |
| GKE | PVCs `highperfasr-batch-model-cache`, `highperfasr-stream-model-cache` (5 Gi, standard-rwo) | Yes, managed by Kubernetes. |
| Plain Docker | Mount any volume to the `HF_HOME` path | Yes, if the same volume is reused. |

### Pre-fetch models

Download models before the first real request so cold start is handled during
deployment, not when traffic arrives:

```bash
# Batch model (Parakeet TDT 0.6B v3)
docker compose run --rm batch python -c \
  "from nemo.collections.asr.models import EncDecRNNTBPEModel; EncDecRNNTBPEModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"

# Streaming model (Nemotron 3.5 ASR 0.6B)
docker compose run --rm stream python -c \
  "from nemo.collections.asr.models import EncDecRNNTBPEModel; EncDecRNNTBPEModel.from_pretrained('nvidia/nemotron-3.5-asr-streaming-0.6b')"
```

Or use the Makefile shortcut:

```bash
make prefetch
```

### Air-gapped deployment

For environments without internet access, populate the cache volume on a
connected machine and copy it to the target:

```bash
# 1. Pre-fetch on a connected machine
docker compose up -d && docker compose down

# 2. Export the cache volumes
docker run --rm -v model-cache-batch:/data -v $(pwd):/backup alpine \
  tar czf /backup/model-cache-batch.tar.gz -C /data .
docker run --rm -v model-cache-stream:/data -v $(pwd):/backup alpine \
  tar czf /backup/model-cache-stream.tar.gz -C /data .

# 3. Transfer archives to the air-gapped machine, then import
docker volume create model-cache-batch
docker volume create model-cache-stream
docker run --rm -v model-cache-batch:/data -v $(pwd):/backup alpine \
  tar xzf /backup/model-cache-batch.tar.gz -C /data
docker run --rm -v model-cache-stream:/data -v $(pwd):/backup alpine \
  tar xzf /backup/model-cache-stream.tar.gz -C /data
```

For GKE air-gapped clusters, build a custom image with models baked in or use
an init container that copies from a GCS bucket into the PVC before the main
container starts.

## Protocol (v1alpha1, draft)

highperfasr uses a framework-agnostic protocol: REST for files, WebSocket for
live audio, and health checks for orchestration.

| Endpoint | Mode | Input |
|----------|------|-------|
| `POST /v1/transcriptions` | Batch | Multipart file upload |
| `WebSocket /v1/stream` | Streaming | Raw PCM16 audio frames |
| `GET /health` | Health | Readiness and server mode |

Full spec: [spec/protocol.md](spec/protocol.md) | [OpenAPI](spec/openapi.yaml) | [AsyncAPI](spec/asyncapi.yaml)

## Structure

```text
Dockerfile           # multi-target image: batch + stream
compose.yaml         # docker compose up -d
gke-l4.yaml          # GKE L4 GPU deployment
labs/nemo-fastapi/   # NeMo serving + framework patches (fork: github.com/beastoin/NeMo)
spec/                # REST + WebSocket protocol
benchmarks/scripts/  # reproducible benchmark scripts (batch, streaming, WER)
benchmarks/results/  # published benchmark reports (JSON + markdown)
```

## Q&A

| Question | Batch (Parakeet TDT 0.6B v3) | Streaming (Nemotron 3.5 ASR 0.6B) |
|----------|------|-----------|
| What languages are supported? | 25 European languages, auto-detect | 36 languages / 40 locales, auto-detect |
| Punctuation & capitalization? | Yes | Yes |
| Word-level timestamps? | Yes | Partial transcripts (real-time) |
| Maximum audio length? | 24 min (full attention), 3 hr (local attention) | Indefinite (persistent WebSocket) |
| Speaker diarization? | Roadmap | Roadmap |
| Inverse text normalization (ITN)? | Roadmap | Roadmap |

**Languages — Batch:** bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt,
lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk (25 European).
**Languages — Streaming:** all batch languages plus ar, ja, ko, zh, hi, th, and
14 more locales (40 total). Set `target_lang` in config or use `auto`.

## Users & Sponsors

- **[Omi](https://omi.me)** uses highperfasr in production for an AI wearable
  workload with thousands of concurrent streams. Omi also sponsors the GPU
  benchmark work published in this repository.

## Mission

Speech recognition infrastructure should be something teams can run, measure,
and control.

highperfasr exists to help companies keep audio inside their own pipeline, on
their own infrastructure, without depending on third-party ASR APIs.

## License

MIT
