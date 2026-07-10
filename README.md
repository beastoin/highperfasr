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

The published L4 streaming benchmark runs 512 persistent WebSocket streams for a
10-minute real-time soak across all 2,620 files in LibriSpeech test-clean.

| Metric | Value |
|--------|-------|
| GPU | NVIDIA L4 24GB |
| Concurrent streams | 512 |
| Failures | 0 / 512 |
| WER | 3.21% (LibriSpeech test-clean) |
| Throughput | 297 sessions/min |
| Realtime factor | 38.69x |
| VRAM | 8672 MB (38%) |

Quality rubric: high quality means real speech corpus, standard WER normalization,
sustained concurrent load, and reproducible artifacts. This run uses LibriSpeech
test-clean (2,620 files, 5.4 hours, CC-BY-4.0), Whisper EnglishTextNormalizer,
`nvidia/nemotron-3.5-asr-streaming-0.6b`, and a 512-stream realtime-paced 10-minute soak.
Verify [result.json](benchmarks/results/2026-l4-nemo-512-streams/result.json),
[c=32..512 sweep](benchmarks/results/2026-l4-nemo-512-streams/concurrency-sweep.json),
and [report schema](benchmarks/report-schema.json).

Full report: [benchmarks/results/2026-l4-nemo-512-streams/](benchmarks/results/2026-l4-nemo-512-streams/)

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

## Protocol (v1alpha1)

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
labs/nemo-fastapi/   # NeMo serving + framework patches
spec/                # REST + WebSocket protocol
benchmarks/          # benchmark reports and tooling
```

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

Apache-2.0
